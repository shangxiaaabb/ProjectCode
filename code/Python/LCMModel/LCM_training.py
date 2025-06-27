import os
import gc
import shutil
import math
import random
import functools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import accelerate
from tqdm import tqdm
from pathlib import Path
from packaging import version
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from contextlib import nullcontext
from transformers import AutoTokenizer, CLIPTextModel, PretrainedConfig
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    LCMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler

from LCM_config import LCMTrainingConfig
from LCM_dataloader import SDText2ImageDataset

def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def scalings_for_boundary_conditions(timestep, sigma_data=0.5, timestep_scaling=10.0):
    scaled_timestep = timestep_scaling * timestep
    c_skip = sigma_data**2 / (scaled_timestep**2 + sigma_data**2)
    c_out = scaled_timestep / (scaled_timestep**2 + sigma_data**2) ** 0.5
    return c_skip, c_out

def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f"input has {x.ndim} dims but target_dims is {target_dims}, which is less")
    return x[(...,) + (None,) * dims_to_append]

@torch.no_grad()
def update_ema(target_params, source_params, rate=0.99):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.

    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)

def guidance_scale_embedding(w, embedding_dim=512, dtype=torch.float32):
    """
    See https://github.com/google-research/vdm/blob/dc27b98a554f65cdc654b800da5aa1846545d41b/model_vdm.py#L298

    Args:
        timesteps (`torch.Tensor`):
            generate embedding vectors at these timesteps
        embedding_dim (`int`, *optional*, defaults to 512):
            dimension of the embeddings to generate
        dtype:
            data type of the generated embeddings

    Returns:
        `torch.Tensor`: Embedding vectors with shape `(len(timesteps), embedding_dim)`
    """
    assert len(w.shape) == 1
    w = w * 1000.0

    half_dim = embedding_dim // 2
    emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=dtype) * -emb)
    emb = w.to(dtype)[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1))
    assert emb.shape == (w.shape[0], embedding_dim)
    return emb

def get_predicted_original_sample(model_output, timesteps, sample, prediction_type, alphas, sigmas):
    alphas = extract_into_tensor(alphas, timesteps, sample.shape)
    sigmas = extract_into_tensor(sigmas, timesteps, sample.shape)
    if prediction_type == "epsilon":
        pred_x_0 = (sample - sigmas * model_output) / alphas
    elif prediction_type == "sample":
        pred_x_0 = model_output
    elif prediction_type == "v_prediction":
        pred_x_0 = alphas * sample - sigmas * model_output
    else:
        raise ValueError(
            f"Prediction type {prediction_type} is not supported; currently, `epsilon`, `sample`, and `v_prediction`"
            f" are supported."
        )

    return pred_x_0

def get_predicted_noise(model_output, timesteps, sample, prediction_type, alphas, sigmas):
    alphas = extract_into_tensor(alphas, timesteps, sample.shape)
    sigmas = extract_into_tensor(sigmas, timesteps, sample.shape)
    if prediction_type == "epsilon":
        pred_epsilon = model_output
    elif prediction_type == "sample":
        pred_epsilon = (sample - alphas * model_output) / sigmas
    elif prediction_type == "v_prediction":
        pred_epsilon = alphas * model_output + sigmas * sample
    else:
        raise ValueError(
            f"Prediction type {prediction_type} is not supported; currently, `epsilon`, `sample`, and `v_prediction`"
            f" are supported."
        )

    return pred_epsilon

def encode_prompt(prompt_batch, text_encoder, tokenizer, proportion_empty_prompts, is_train=True):
    captions = []
    for caption in prompt_batch:
        if random.random() < proportion_empty_prompts:
            captions.append("")
        elif isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            captions.append(random.choice(caption) if is_train else caption[0])

    with torch.no_grad():
        text_inputs = tokenizer(
            captions,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        prompt_embeds = text_encoder(text_input_ids.to(text_encoder.device))[0]

    return prompt_embeds

def log_validation(vae, unet, config, accelerator, weight_dtype, step, name="target"):
    unet = accelerator.unwrap_model(unet)
    pipeline = StableDiffusionPipeline.from_pretrained(
        config.teacher_model,
        vae=vae,
        unet=unet,
        scheduler=LCMScheduler.from_pretrained(config.teacher_model, subfolder="scheduler"),
        torch_dtype=weight_dtype,
        cache_dir = config.cache_dir
    )
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    if config.xformer:
        pipeline.enable_xformers_memory_efficient_attention()

    if config.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(config.seed)

    validation_prompts = [
        "This is a selfie photo of Doc Brown and Marty McFly in a typical 1910s nobility with the Titanic just before the ship departed. The dock is filled with people boarding the famous ocean liner, and cranes are lifting passengers' baggage.",
        "Elven warrior in silver armor, standing in enchanted forest, cinematic lighting", "Close-up portrait of a tiger with glowing eyes, hyper-realistic"
    ]

    image_logs = []

    for _, prompt in enumerate(validation_prompts):
        images = []
        if torch.backends.mps.is_available():
            autocast_ctx = nullcontext()
        else:
            autocast_ctx = torch.autocast(accelerator.device.type)

        with autocast_ctx:
            images = pipeline(
                prompt=prompt,
                num_inference_steps=4,
                num_images_per_prompt=4,
                generator=generator,
            ).images
        image_logs.append({"validation_prompt": prompt, "images": images})

    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            for log in image_logs:
                images = log["images"]
                validation_prompt = log["validation_prompt"]
                formatted_images = []
                for image in images:
                    formatted_images.append(np.asarray(image))

                formatted_images = np.stack(formatted_images)

                tracker.writer.add_images(validation_prompt, formatted_images, step, dataformats="NHWC")
        del pipeline
        gc.collect()
        torch.cuda.empty_cache()

        return image_logs
    
class DDIMSolver:
    def __init__(self, alpha_cumprods, timesteps=1000, ddim_timesteps=50):
        # DDIM sampling parameters
        step_ratio = timesteps // ddim_timesteps
        self.ddim_timesteps = (np.arange(1, ddim_timesteps + 1) * step_ratio).round().astype(np.int64) - 1
        self.ddim_alpha_cumprods = alpha_cumprods[self.ddim_timesteps]
        self.ddim_alpha_cumprods_prev = np.asarray(
            [alpha_cumprods[0]] + alpha_cumprods[self.ddim_timesteps[:-1]].tolist()
        )
        # convert to torch tensors
        self.ddim_timesteps = torch.from_numpy(self.ddim_timesteps).long()
        self.ddim_alpha_cumprods = torch.from_numpy(self.ddim_alpha_cumprods)
        self.ddim_alpha_cumprods_prev = torch.from_numpy(self.ddim_alpha_cumprods_prev)

    def to(self, device):
        self.ddim_timesteps = self.ddim_timesteps.to(device)
        self.ddim_alpha_cumprods = self.ddim_alpha_cumprods.to(device)
        self.ddim_alpha_cumprods_prev = self.ddim_alpha_cumprods_prev.to(device)
        return self

    def ddim_step(self, pred_x0, pred_noise, timestep_index):
        alpha_cumprod_prev = extract_into_tensor(self.ddim_alpha_cumprods_prev, timestep_index, pred_x0.shape)
        dir_xt = (1.0 - alpha_cumprod_prev).sqrt() * pred_noise
        x_prev = alpha_cumprod_prev.sqrt() * pred_x0 + dir_xt
        return x_prev
    
def main():
    config = LCMTrainingConfig()
    set_seed(config.seed)

    # store path
    accelerator = Accelerator(gradient_accumulation_steps= config.gradient_accumulation_steps,
                              mixed_precision= config.mixed_precision,
                              log_with= config.log_with,
                              project_config=ProjectConfiguration(project_dir= config.out_dir, 
                                                                  logging_dir=Path(config.out_dir, "logs")))
    
    if accelerator.is_main_process:
        os.makedirs(config.out_dir, exist_ok= True)
    
    # 1. Create the noise scheduler and the desired noise schedule.
    noise_scheduler = DDPMScheduler.from_pretrained(config.teacher_model, subfolder="scheduler")
    alpha_schedule = torch.sqrt(noise_scheduler.alphas_cumprod)
    sigma_schedule = torch.sqrt(1 - noise_scheduler.alphas_cumprod)
    solver = DDIMSolver(noise_scheduler.alphas_cumprod.numpy(),
                        timesteps= noise_scheduler.config.num_train_timesteps,
                        ddim_timesteps= config.num_ddim_timesteps,)
    
    # 2. Load teacher model freeze teatcher model parameter
    tokenizer = AutoTokenizer.from_pretrained(config.teacher_model, subfolder="tokenizer", 
                                              use_fast=False, cache_dir= config.cache_dir)
    text_encoder = CLIPTextModel.from_pretrained(config.teacher_model, subfolder="text_encoder",
                                                 cache_dir= config.cache_dir)
    vae = AutoencoderKL.from_pretrained(config.teacher_model, subfolder="vae", cache_dir= config.cache_dir)
    teacher_model = UNet2DConditionModel.from_pretrained(config.teacher_model, subfolder="unet",
                                                         cache_dir= config.cache_dir)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    teacher_model.requires_grad_(False)

    # 3. Build Student Model
    time_cond_proj_dim = (teacher_model.config.time_cond_proj_dim
                          if teacher_model.config.time_cond_proj_dim is not None
                          else 256
    )
    unet = UNet2DConditionModel.from_config(teacher_model.config, time_cond_proj_dim=time_cond_proj_dim)
    unet.load_state_dict(teacher_model.state_dict(), strict=False)
    unet.train()

    student_model = UNet2DConditionModel.from_config(unet.config)
    student_model.load_state_dict(unet.state_dict(), strict=False)
    student_model.train()
    student_model.requires_grad_(False)

    # 4. Set weight dtype Move device
    weight_dtype = torch.float16 if config.mixed_precision== 'fp16' else torch.bfloat16
    vae.to(accelerator.device)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    student_model.to(accelerator.device)
    teacher_model.to(accelerator.device)
    teacher_model.to(weight_dtype)

    alpha_schedule = alpha_schedule.to(accelerator.device)
    sigma_schedule = sigma_schedule.to(accelerator.device)
    solver = solver.to(accelerator.device)

    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # `accelerate` 0.16.0 will have better support for customized saving
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                student_model.save_pretrained(os.path.join(output_dir, "unet_target"))

                for i, model in enumerate(models):
                    model.save_pretrained(os.path.join(output_dir, "unet"))
                    weights.pop()

        def load_model_hook(models, input_dir):
            load_model = UNet2DConditionModel.from_pretrained(os.path.join(input_dir, "unet_target"))
            student_model.load_state_dict(load_model.state_dict())
            student_model.to(accelerator.device)
            del load_model

            for i in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)
    
    if config.xformer:
        import xformers
        unet.enable_xformers_memory_efficient_attention()
        teacher_model.enable_xformers_memory_efficient_attention()
        student_model.enable_xformers_memory_efficient_attention()
    if config.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
    # optimizer config
    if config.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    # 5. optimizer, dataset, lr_sch
    optimizer = optimizer_class(unet.parameters(),
                                lr= config.lr_rate,
                                betas= (config.adam_beta1, config.adam_beta2),
                                weight_decay= config.weight_decay,
                                eps= config.epsilon)
    def compute_embeddings(prompt_batch, proportion_empty_prompts, text_encoder, tokenizer, is_train=True):
        prompt_embeds = encode_prompt(prompt_batch, text_encoder, tokenizer, proportion_empty_prompts, is_train)
        return {"prompt_embeds": prompt_embeds}

    dataset = SDText2ImageDataset(
        train_shards_path_or_url= config.train_data_path,
        num_train_examples= config.num_train_samples,
        per_gpu_batch_size= config.train_batch_size,
        global_batch_size= config.train_batch_size * accelerator.num_processes,
        num_workers= config.num_workers,
        resolution= config.resolution,
    )
    train_dataloader = dataset.train_dataloader

    compute_embeddings_fn = functools.partial(compute_embeddings,
                                              proportion_empty_prompts=0,
                                              text_encoder=text_encoder,
                                              tokenizer=tokenizer,
    )
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(train_dataloader.num_batches / config.gradient_accumulation_steps)
    if config.max_train_steps is None:
        config.max_train_steps = config.epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(config.lr_scheduler,
                                 optimizer= optimizer,
                                 num_warmup_steps= config.lr_warmup_steps,
                                 num_training_steps= config.max_train_steps)
    unet, optimizer, lr_scheduler = accelerator.prepare(unet, optimizer, lr_scheduler)

    num_update_steps_per_epoch = math.ceil(train_dataloader.num_batches / config.gradient_accumulation_steps)
    if overrode_max_train_steps:
        config.max_train_steps = config.epochs * num_update_steps_per_epoch
    config.epochs = math.ceil(config.max_train_steps / num_update_steps_per_epoch)

    if accelerator.is_main_process:
        accelerator.init_trackers(config.project_name)

    uncond_input_ids = tokenizer(
        [""] * config.train_batch_size, return_tensors="pt", padding="max_length", max_length=77
    ).input_ids.to(accelerator.device)
    uncond_prompt_embeds = text_encoder(uncond_input_ids)[0]

    # 6. training
    global_step, first_epoch = 0, 0

    # 直接再训练
    if config.resume_from_checkpoint:
        if config.resume_from_checkpoint != "latest":
            path = os.path.basename(config.resume_from_checkpoint)

        if path is None:
            config.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(config.out_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, config.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(first_epoch, config.epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # train-1 data
                image, text = batch
                image = image.to(accelerator.device, non_blocking= True)
                encoded_text = compute_embeddings_fn(text)

                pixel_values = image.to(dtype=weight_dtype)
                if vae.dtype != weight_dtype:
                    vae.to(dtype=weight_dtype)

                latents = []
                for i in range(0, pixel_values.shape[0], config.vae_encode_batch_size):
                    latents.append(vae.encode(pixel_values[i : i + config.vae_encode_batch_size]).latent_dist.sample())
                latents = torch.cat(latents, dim=0)

                latents = latents * vae.config.scaling_factor
                latents = latents.to(weight_dtype)
                bsz = latents.shape[0]

                # train-2 sample
                topk = noise_scheduler.config.num_train_timesteps // config.num_ddim_timesteps
                index = torch.randint(0, config.num_ddim_timesteps, (bsz,), device=latents.device).long()
                start_timesteps = solver.ddim_timesteps[index]
                timesteps = start_timesteps - topk
                timesteps = torch.where(timesteps < 0, torch.zeros_like(timesteps), timesteps)

                # train-3
                c_skip_start, c_out_start = scalings_for_boundary_conditions(
                        start_timesteps, timestep_scaling= config.timestep_scaling_factor
                    )
                c_skip_start, c_out_start = [append_dims(x, latents.ndim) for x in [c_skip_start, c_out_start]]
                c_skip, c_out = scalings_for_boundary_conditions(
                    timesteps, timestep_scaling= config.timestep_scaling_factor
                )
                c_skip, c_out = [append_dims(x, latents.ndim) for x in [c_skip, c_out]]

                # train-3 noise
                noise = torch.randn_like(latents)
                noisy_model_input = noise_scheduler.add_noise(latents, noise, start_timesteps)

                # train-4 guidance scale
                w = (config.w_max - config.w_min) * torch.rand((bsz,)) + config.w_min
                w_embedding = guidance_scale_embedding(w, embedding_dim=time_cond_proj_dim)
                w = w.reshape(bsz, 1, 1, 1)
                w = w.to(device=latents.device, dtype=latents.dtype)
                w_embedding = w_embedding.to(device=latents.device, dtype=latents.dtype)

                # train-5 prompt embeds model out
                prompt_embeds = encoded_text.pop("prompt_embeds")
                noise_pred = unet(
                    noisy_model_input,
                    start_timesteps,
                    timestep_cond=w_embedding,
                    encoder_hidden_states=prompt_embeds.float(),
                    added_cond_kwargs=encoded_text,
                ).sample

                pred_x_0 = get_predicted_original_sample(
                    noise_pred,
                    start_timesteps,
                    noisy_model_input,
                    noise_scheduler.config.prediction_type,
                    alpha_schedule,
                    sigma_schedule,
                )

                model_pred = c_skip_start * noisy_model_input + c_out_start * pred_x_0

                # train-6 condiction uncondiction 
                with torch.no_grad():
                    if torch.backends.mps.is_available():
                        autocast_ctx = nullcontext()
                    else:
                        autocast_ctx = torch.autocast(accelerator.device.type)

                    with autocast_ctx:
                        # 1. Get teacher model prediction on noisy_model_input z_{t_{n + k}} and conditional embedding c
                        cond_teacher_output = teacher_model(
                            noisy_model_input.to(weight_dtype),
                            start_timesteps,
                            encoder_hidden_states=prompt_embeds.to(weight_dtype),
                        ).sample
                        cond_pred_x0 = get_predicted_original_sample(
                            cond_teacher_output,
                            start_timesteps,
                            noisy_model_input,
                            noise_scheduler.config.prediction_type,
                            alpha_schedule,
                            sigma_schedule,
                        )
                        cond_pred_noise = get_predicted_noise(
                            cond_teacher_output,
                            start_timesteps,
                            noisy_model_input,
                            noise_scheduler.config.prediction_type,
                            alpha_schedule,
                            sigma_schedule,
                        )

                        # 2. Get teacher model prediction on noisy_model_input z_{t_{n + k}} and unconditional embedding 0
                        uncond_teacher_output = teacher_model(
                            noisy_model_input.to(weight_dtype),
                            start_timesteps,
                            encoder_hidden_states=uncond_prompt_embeds.to(weight_dtype),
                        ).sample
                        uncond_pred_x0 = get_predicted_original_sample(
                            uncond_teacher_output,
                            start_timesteps,
                            noisy_model_input,
                            noise_scheduler.config.prediction_type,
                            alpha_schedule,
                            sigma_schedule,
                        )
                        uncond_pred_noise = get_predicted_noise(
                            uncond_teacher_output,
                            start_timesteps,
                            noisy_model_input,
                            noise_scheduler.config.prediction_type,
                            alpha_schedule,
                            sigma_schedule,
                        )

                        # 3. Calculate the CFG estimate of x_0 (pred_x0) and eps_0 (pred_noise)
                        pred_x0 = cond_pred_x0 + w * (cond_pred_x0 - uncond_pred_x0)
                        pred_noise = cond_pred_noise + w * (cond_pred_noise - uncond_pred_noise)
                        # 4. Run one step of the ODE solver to estimate the next point x_prev on the
                        # augmented PF-ODE trajectory (solving backward in time)
                        # Note that the DDIM step depends on both the predicted x_0 and source noise eps_0.
                        x_prev = solver.ddim_step(pred_x0, pred_noise, index)

                # train-7 LCM
                with torch.no_grad():
                    if torch.backends.mps.is_available():
                        autocast_ctx = nullcontext()
                    else:
                        autocast_ctx = torch.autocast(accelerator.device.type, dtype=weight_dtype)

                    with autocast_ctx:
                        target_noise_pred = student_model(
                            x_prev.float(),
                            timesteps,
                            timestep_cond=w_embedding,
                            encoder_hidden_states=prompt_embeds.float(),
                        ).sample
                    pred_x_0 = get_predicted_original_sample(
                        target_noise_pred,
                        timesteps,
                        x_prev,
                        noise_scheduler.config.prediction_type,
                        alpha_schedule,
                        sigma_schedule,
                    )
                    target = c_skip * x_prev + c_out * pred_x_0
                
                # train-8 com loss backward
                if config.loss_type == "l2":
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                elif config.loss_type == "huber":
                    loss = torch.mean(
                        torch.sqrt((model_pred.float() - target.float()) ** 2 + config.huber_c**2) - config.huber_c
                    )
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), config.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                # train-9
                if accelerator.sync_gradients:
                    update_ema(student_model.parameters(), unet.parameters(), config.ema_decay)
                    progress_bar.update(1)
                    global_step += 1

                    if accelerator.is_main_process:
                        if global_step % config.checkpointing_steps == 0:
                            if config.checkpoints_total_limit is not None:
                                checkpoints = os.listdir(config.out_dir)
                                checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                                checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                                # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                                if len(checkpoints) >= config.checkpoints_total_limit:
                                    num_to_remove = len(checkpoints) - config.checkpoints_total_limit + 1
                                    removing_checkpoints = checkpoints[0:num_to_remove]

                                    for removing_checkpoint in removing_checkpoints:
                                        removing_checkpoint = os.path.join(config.out_dir, removing_checkpoint)
                                        shutil.rmtree(removing_checkpoint)

                            save_path = os.path.join(config.out_dir, f"checkpoint-{global_step}")
                            accelerator.save_state(save_path)

                        if global_step % config.validation_steps == 0:
                            log_validation(vae, student_model, config, accelerator, 
                                           weight_dtype, global_step, "target")
                            log_validation(vae, unet, config, accelerator, 
                                           weight_dtype, global_step, "online")

                logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)

                # if global_step >= config.max_train_steps:
                #     break
                
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)
        unet.save_pretrained(os.path.join(config.out_dir, "unet"))

        student_model = accelerator.unwrap_model(student_model)
        student_model.save_pretrained(os.path.join(config.out_dir, "unet_target"))
    accelerator.end_training()

if __name__ == "__main__":
    # CUDA_VISIBLE_DEVICES=1 accelerate launch --num_processes=1 LCM_training.py
    main()