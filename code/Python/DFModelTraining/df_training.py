import os
import re
import imageio
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from diffusers import DDPMScheduler, AutoencoderKL
from diffusers.optimization import get_cosine_schedule_with_warmup
from tqdm.auto import tqdm

from config import dit_config, vae_dit_config
from df_model import DiffusionModel
from df_dataloader import DFDataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'
config = dit_config.DitConfig()
# config = vae_dit_config.DitConfig()

noise_scheduler = DDPMScheduler(num_train_timesteps= config.num_train_timesteps,
                            beta_start= config.beta_start,
                            beta_end= config.beta_end,
                            beta_schedule= 'scaled_linear')
model = DiffusionModel(df_model_name= config.training,
                        params= config.params_dit,
                        text_model_name= config.text_model,
                        cache_dir= config.cache_dir)
if config.vae:
    vae_model = AutoencoderKL.from_pretrained(config.vae,
                                        subfolder="vae",
                                        cache_dir= config.cache_dir).to(config.device)
    vae_model.eval()
    for param in vae_model.parameters():
        param.requires_grad = False

optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

def generate(model, noise_image, scheduler, labels=None, vae_model=None):
    def save_samples(samples, title="Generated Samples"):
        fig, axes = plt.subplots(1, len(samples), figsize=(len(samples) * 2, 2))
        for i, img in enumerate(samples):
            axes[i].imshow(img.permute(1, 2, 0).numpy())  # RGB 图像，[C, H, W] -> [H, W, C]
            axes[i].axis('off')
        plt.title(title)
        plt.savefig(f"{title.replace(' ', '_')}.png")
        plt.close()
    
    def generate_gif(path_dir):
        '''生成gif'''
        images = []
        path_list = os.listdir(path_dir)
        def sort_key(filename):
            match = re.search(r'(\d+)', filename)
            return int(match.group(1)) if match else filename
        # 按数字排序
        path_list = sorted(path_list, key=sort_key, reverse= True)
        for path in path_list:
            img_path = os.path.join(path_dir, path)
            if os.path.exists(img_path):
                images.append(imageio.v2.imread(img_path))
        if images:
            imageio.mimsave(f'{config.output_dir}/Generate_image.gif', images, fps=10)
        print(f"The GIF store in: {config.output_dir}/Generate_image.gif")
    
    os.makedirs(config.store_path, exist_ok= True)
    model.eval()
    model.to(noise_image.device)
    with torch.no_grad():
        x = noise_image.to(noise_image.device)
        with tqdm(total= len(scheduler.timesteps)) as pbar:
            for t in scheduler.timesteps:
                t_tensor = torch.full((x.shape[0],), 
                                      t, 
                                      dtype=torch.long, 
                                      device=noise_image.device)
   
                predicted_noise = model(x, t_tensor, labels)
                x = scheduler.step(predicted_noise, t, x).prev_sample
                if t% 20==0:
                    tmp_x = (x.clamp(-1, 1) + 1) / 2
                    if vae_model is not None:
                        tmp_x = vae_model.decode(tmp_x).sample
                    save_samples(tmp_x.cpu(), 
                                 f'{config.store_path}/{t.item()}')
                pbar.update(1)
        x = (x.clamp(-1, 1) + 1) / 2
        if vae_model is not None:
            x = vae_model.decode(x).sample
        generate_gif(path_dir= config.store_path)
        return x.cpu()

def evaluate(config, 
             epoch, 
             model= None, 
             noise_scheduler= None,
             accelerator= None,
             text_label=None,
             vae_model= None):
    def make_image_grid(images, rows, cols):
        from PIL import Image
        assert len(images) >= rows * cols, "Not enough images to create grid"
        images = images[:rows * cols]
        images = [Image.fromarray((img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)) for img in images]
        
        w, h = images[0].size
        grid = Image.new('RGB', size=(cols * w, rows * h))
        for i, img in enumerate(images):
            grid.paste(img, box=(i % cols * w, i // cols * h))
        return grid
    
    test_dir = os.path.join(config.output_dir, "samples")
    os.makedirs(test_dir, exist_ok=True)

    device = accelerator.device
    model = accelerator.unwrap_model(model).to(device)
    model.eval()

    noise = torch.randn(
        (config.eval_batch_size, config.channel, config.image_size, config.image_size),
        generator=torch.Generator(device=device).manual_seed(config.seed),
        device=device
    )
    
    with torch.no_grad():
        with tqdm(total= len(noise_scheduler.timesteps), desc= "Eva") as pbar:
            for t in noise_scheduler.timesteps:
                t_tensor = torch.full((noise.shape[0],), 
                                        t, 
                                        dtype=torch.long, 
                                        device= device)
                predicted_noise = model(noise, t_tensor, text_label)
                noise = noise_scheduler.step(predicted_noise, t, noise).prev_sample
                pbar.update(1)
        images = (noise.clamp(-1, 1) + 1) / 2
        if vae_model is not None:
            images = vae_model.decode(images).sample
    image_grid = make_image_grid(images, rows=4, cols=4)
    image_grid.save(f"{test_dir}/{epoch:04d}.png")
    model.train()


def train(model, noise_scheduler, optimizer, vae_model= None):
    # data loader
    train_dataset = DFDataset(data_name= config.dataset_name,
                                 cache_dir= config.cache_dir,
                                 text_dir= config.text_dir,
                                 image_dir= config.image_dir,
                                 sub_dataset= config.sub_dataset)
    train_dataloader = DataLoader(train_dataset, config.batch_size, shuffle= True,
                                  num_workers= 16)
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer, 
                                                   num_warmup_steps=config.lr_warmup_steps, 
                                                   num_training_steps=len(train_dataloader) * config.epochs)
    kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)]
    accelerator = Accelerator(mixed_precision=config.mixed_precision, 
                              gradient_accumulation_steps=config.gradient_accumulation_steps,
                              log_with= "tensorboard",
                              project_dir=os.path.join(config.output_dir, "logs"),
                              kwargs_handlers= kwargs_handlers
                              )

    if accelerator.is_main_process:
        os.makedirs(config.output_dir, exist_ok=True)
        accelerator.init_trackers(f"Train-{config.training}")
    
    if config.vae:
        model, optimizer, train_dataloader, lr_scheduler, vae_model = accelerator.prepare(model, optimizer, 
                                                                               train_dataloader, 
                                                                               lr_scheduler,
                                                                               vae_model)
    else:
        model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(model, optimizer, 
                                                                               train_dataloader, 
                                                                               lr_scheduler,)
        
    if accelerator.is_main_process:
        print("*"*20, f"\nModle Name: {config.training} Batch Size: {config.batch_size} Data Nums: {len(train_dataloader)} VAE: {config.vae}\n", "*"*20)

    global_step = 0
    for epoch in range(config.epochs):
        progress_bar = tqdm(total=len(train_dataloader), 
                            disable=not accelerator.is_local_main_process, desc=f"Epoch {epoch}")
        # train model
        for i, batch in enumerate(train_dataloader):
            if config.dataset_name in ["Flicker30k", "CIFAR-100"]:
                image, text_label = batch
            elif config.dataset_name == "CelebA-HQ":
                image = batch["images"]
                text_label = None
            image = image.to(accelerator.device)

            timesteps = torch.randint(0, 
                                      noise_scheduler.config.num_train_timesteps, 
                                      (image.shape[0],), 
                                      device=image.device, 
                                      dtype=torch.int64)
            if config.vae:
                image = vae_model.encode(image).latent_dist.sample()
            
            noise = torch.randn(image.shape, device= accelerator.device)
            noise_image = noise_scheduler.add_noise(image, noise, timesteps)

            with accelerator.accumulate(model):
                if text_label is None:
                    noise_pred = model(noise_image, timesteps)
                else:
                    noise_pred = model(noise_image, timesteps, text_label)
                loss = F.mse_loss(noise_pred, noise)

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 
                                                1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), 
                    "lr": lr_scheduler.get_last_lr()[0], 
                    "step": global_step}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1
        
        # evaluate save model
        if accelerator.is_main_process:
            if (epoch+ 1)%10 ==0 or epoch== config.epochs-1:
                text_label = None
                if 'text' in config.training:
                    text_label = "An asian man wearing a black suit stands near a dark-haired woman and a brown-haired woman ."
                evaluate(config, 
                         epoch, 
                         model= model, 
                         noise_scheduler= noise_scheduler,
                         accelerator= accelerator,
                         text_label= text_label,
                         vae_model= vae_model)
            if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.epochs - 1:
                model_to_save = accelerator.unwrap_model(model)
                torch.save({
                    'model_state_dict': model_to_save.state_dict(),
                    'epoch': epoch,
                    'config': vars(config)
                }, f"{config.output_dir}/model_checkpoint_{epoch}.pth")
                noise_scheduler.save_config(config.output_dir)
                print(f"Saved state_dict to {config.output_dir}/model_checkpoint.pth")
    
    # generate
    if accelerator.is_main_process:
        n_samples, image_size = 5, config.image_size
        text_label = None
        scheduler = DDPMScheduler.from_pretrained(f"{config.output_dir}")
        scheduler.set_timesteps(config.num_train_timesteps)
        checkpoint = torch.load(f"{config.output_dir}/model_checkpoint_{epoch}.pth", 
                                map_location=device,
                                weights_only= False)
        model.module.load_state_dict(checkpoint['model_state_dict'], strict=False)
        
        clean_images = torch.randn(n_samples, config.channel, image_size, image_size).to(config.device)
        if 'text' in config.training:
            text_label = "An asian man wearing a black suit stands near a dark-haired woman and a brown-haired woman ."
        noisy_images = torch.randn(n_samples, 
                                   config.channel, 
                                   image_size, 
                                   image_size).to(clean_images.device)
        
        generate(model, 
                 noisy_images, 
                 scheduler, 
                 labels= text_label,
                 vae_model= vae_model)


if __name__ == '__main__': 
    # CUDA_VISIBLE_DEVICES=2,3 accelerate launch --num_processes=2 df_training.py
    if config.vae:
        train(model, noise_scheduler, optimizer, vae_model)
    else:
        train(model, noise_scheduler, optimizer)