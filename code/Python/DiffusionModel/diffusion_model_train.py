import os
import re
import imageio
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from dataclasses import dataclass
import torchvision
from torch.utils.data import DataLoader, ConcatDataset
from datasets import load_dataset
from torchvision import transforms
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from diffusers import UNet2DModel, AutoencoderKL, DDPMScheduler, DDPMPipeline
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers.utils import make_image_grid
from tqdm.auto import tqdm

@dataclass
class TrainingConfig:
    latent_size = 16
    eval_batch_size= 16 # 测试生成效果
    learning_rate = 1e-4
    lr_warmup_steps = 500 # 选择10-20% 50*100*0.1

    num_epochs = 100
    sub_datasets = True if num_epochs==1 else False

    gradient_accumulation_steps = 1
    save_model_epochs = 20
    mixed_precision = "fp16"
    seed = 0
    
    # 配置线性方差调度
    num_train_timesteps = 1000
    beta_start = 1e-4
    beta_end = 2e-2

    # mask
    session= 'normal' # 训练任务 normal mask condiction
    mask_ratio= 0.8
    min_mask_size = 0.1
    max_mask_size = 0.5
    irregular_prob = 0.5

    # data
    dataset_name = "saitsharipov/CelebA-HQ" #"huggan/smithsonian_butterflies_subset"  "saitsharipov/CelebA-HQ"  "CIFAR-100"
    image_size = 128 if dataset_name != "CIFAR-100" else 32

    # model
    vae= "stabilityai/stable-diffusion-2-1" #latent (batch_size, 4, height/8, width/8)
    # vae = None

    unet_model= 'dit'
    batch_size_dict = {"dit": 84,
                       "hf": 16,
                       "sd": 16}
    batch_size_dict['dit'] = 520 if vae else 84
    batch_size = batch_size_dict[unet_model]

    params_hf = {
        'sample_size': latent_size if vae else image_size, 
        'in_channels': 4 if vae else 3,
        'out_channels': 4 if vae else 3,
        'class_embed_type': 'timestep' if session== 'condiction' else None,
        'layers_per_block': 2,
        'block_out_channels': (128, 128, 256, 256, 512, 512),
        'down_block_types': ("DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D", 
                             "AttnDownBlock2D", "DownBlock2D"),
        'up_block_types': ("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D", 
                           "UpBlock2D", "UpBlock2D")
    } # 使用hugging face model

    params_sd = {
        'ch': 128,
        'out_ch': 4 if vae else 3,
        'ch_mult': (1, 2, 4),
        'num_res_blocks': 2,
        'attn_resolutions': (16, 8),
        'dropout': 0.1,
        'resamp_with_conv': True,
        'in_channels': 4 if vae else 3,
        'resolution': latent_size if vae else image_size,
        'use_timestep': True,
        'use_class': True if session== 'condiction' else False,
        'use_linear_attn': False,
        'attn_type': "vanilla"
    } # 使用stable diffusion model

    params_dit = {
        'depth': 28,
        'hidden_size': 1152, 
        'patch_size': 8, 
        'num_heads': 16,
        'input_size': latent_size if vae else image_size,
        'in_channels': 4 if vae else 3,
        'learn_sigma': False
    } # 使用dit模型

    # store
    output_dir = f"ddpm-{unet_model}-{session}-VAE" if vae else f"ddpm-{unet_model}-{session}"
    store_path = f"{output_dir}/Image/"
    cache_dir = '/data/huangjie'

    device= 'cuda' if torch.cuda.is_available() else 'cpu'

def generate(model, nois_image, scheduler, masks= None, clean_images=None, labels=None):
    '''无条件生成器：从噪声图像生成最终图像'''
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
    model.to(nois_image.device)
    with torch.no_grad():
        if config.vae:
            latent_channels = 4  # VAE latent 空间的通道数
            with torch.no_grad():
                if clean_images is not None:
                    clean_images = vae.encode(clean_images).latent_dist.sample()  # [B, 4, H/8, W/8]
                nois_image = torch.randn(
                    nois_image.shape[0], latent_channels, 
                    nois_image.shape[2], nois_image.shape[3],
                    device= nois_image.device
                )
        else:
            latent_channels = 3  # 原始 RGB 图像
    
        x = nois_image.to(nois_image.device)
        if masks is not None and clean_images is not None:
            masks = masks.to(nois_image.device)
            clean_images = clean_images.to(nois_image.device)
            x = masks * clean_images + (1 - masks) * x

        # 采样循环
        with tqdm(total= len(scheduler.timesteps)) as pbar:
            for t in scheduler.timesteps:
                t_tensor = torch.full((x.shape[0],), 
                                    t, 
                                    dtype=torch.long, 
                                    device=nois_image.device)
                try:
                    pred_noise = model(x, 
                                       t_tensor, 
                                       labels).sample if labels is not None else model(x, t_tensor).sample
                except Exception:
                    pred_noise = model(x, 
                                       t_tensor, 
                                       labels) if labels is not None else model(x, t_tensor)
                x = scheduler.step(pred_noise, t, x).prev_sample
                if t% 20==0:
                    tmp_x = x
                    if config.vae:
                        # 解码 latent 回像素空间
                        tmp_x = vae.decode(tmp_x).sample
                    tmp_x = (x.clamp(-1, 1) + 1) / 2
                    save_samples(tmp_x.cpu(), f'{config.store_path}/{t.item()}')
                pbar.update(1)
        # 反归一化到 [0, 1]
        if config.vae:
            x = vae.decode(x).sample  # [B, 3, H, W]
        x = (x.clamp(-1, 1) + 1) / 2
        generate_gif(path_dir= config.store_path)
        return x.cpu()

def load_clean_images(image_path, n_samples=5, device='cuda'):
    import cv2
    trans = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((config.image_size, config.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)    
    img_tensor = trans(img)  # 形状: (C, H, W)
    clean_images = img_tensor.unsqueeze(0).repeat(n_samples, 1, 1, 1)  # 形状: (n_samples, C, H, W)
    return clean_images.to(device)

def load_datasets(config: TrainingConfig):
    def _transform(examples, dataset_name="CelebA"):
        if dataset_name == "CIFAR-100":
            preprocess = transforms.Compose([
                transforms.Resize((config.image_size, config.image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), 
                                   (0.2675, 0.2565, 0.2761))
            ])
            images = [preprocess(_[0]) for _ in examples]
            labels = [_[1] for _ in examples]
            return {"images": images, "labels": labels}
        else:
            preprocess = transforms.Compose([
                transforms.Resize((config.image_size, config.image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
            return {"images": [preprocess(image.convert("RGB")) for image in examples["image"]]}

    def point_in_polygon(x, y, poly_x, poly_y):
        """Simplified point-in-polygon test for mask generation"""
        n = len(poly_x)
        inside = torch.zeros_like(x, dtype=torch.bool, device=x.device)
        for i in range(n):
            j = (i + 1) % n
            cond = ((poly_y[i] > y) != (poly_y[j] > y)) & \
                   (x < (poly_x[j] - poly_x[i]) * (y - poly_y[i]) / (poly_y[j] - poly_y[i] + 1e-10) + poly_x[i])
            inside ^= cond
        return inside

    def collate_fn(batch):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if config.session == 'mask':
            if isinstance(batch[0], dict):
                images = torch.stack([item['images'] for item in batch]).to(device)
                labels = [item['labels'] for item in batch if item['labels'] is not None]
                labels = torch.tensor(labels, device=device) if labels else None
            else:  # For CIFAR-100 tuple-style data
                images = torch.stack([item[0]['images'] for item in batch]).to(device)
                labels = torch.tensor([item[1] for item in batch], device=device)

            batch_size, channels, height, width = images.shape

            # Initialize masks
            if config.vae:
                masks = torch.ones(batch_size, 1, 
                                 config.latent_size,
                                 config.latent_size,
                                 device=device)
            else:
                masks = torch.ones(batch_size, 1, 
                                 height, width, 
                                 device=device)

            # Configuration parameters
            mask_ratio = config.mask_ratio
            min_mask_size = config.min_mask_size
            max_mask_size = config.max_mask_size
            irregular_prob = config.irregular_prob

            # Vectorized random parameters
            apply_mask = torch.rand(batch_size, device=device) <= mask_ratio
            mask_heights = (min_mask_size + (max_mask_size - min_mask_size) * torch.rand(batch_size, device=device)) * height
            mask_widths = (min_mask_size + (max_mask_size - min_mask_size) * torch.rand(batch_size, device=device)) * width
            ys = (torch.rand(batch_size, device=device) * (height - mask_heights)).clamp(0, height - 1).int()
            xs = (torch.rand(batch_size, device=device) * (width - mask_widths)).clamp(0, width - 1).int()

            # Clamp mask sizes
            mask_heights = torch.clamp(mask_heights, 0, height).int()
            mask_widths = torch.clamp(mask_widths, 0, width).int()

            for i in range(batch_size):
                if not apply_mask[i]:
                    continue

                mask_h, mask_w = mask_heights[i].item(), mask_widths[i].item()
                x, y = xs[i].item(), ys[i].item()

                # Ensure mask doesn't go out of bounds
                mask_h = min(mask_h, height - y)
                mask_w = min(mask_w, width - x)
                if mask_h <= 0 or mask_w <= 0:
                    continue

                if torch.rand(1, device=device).item() <= irregular_prob:
                    # Irregular mask (circle/ellipse/polygon)
                    center_y, center_x = y + mask_h // 2, x + mask_w // 2
                    radius = min(mask_h, mask_w) // 2

                    yy, xx = torch.meshgrid(
                        torch.arange(height, device=device),
                        torch.arange(width, device=device),
                        indexing='ij'
                    )

                    shape_type = torch.rand(1, device=device).item()
                    if shape_type < 0.3:
                        # Ellipse
                        a = mask_w / 2 * (0.8 + torch.rand(1, device=device).item() * 0.4)
                        b = mask_h / 2 * (0.8 + torch.rand(1, device=device).item() * 0.4)
                        mask_area = ((xx - center_x) / a) ** 2 + ((yy - center_y) / b) ** 2 <= 1
                    else:
                        # Polygon approximation
                        num_vertices = torch.randint(3, 8, (1,), device=device).item()
                        angles = torch.linspace(0, 2 * np.pi, num_vertices + 1, device=device)[:-1]
                        radius_variation = 0.7 + torch.rand(num_vertices, device=device) * 0.6
                        vertices_x = center_x + radius * radius_variation * torch.cos(angles)
                        vertices_y = center_y + radius * radius_variation * torch.sin(angles)
                        mask_area = point_in_polygon(xx, yy, vertices_x, vertices_y)

                    masks[i, 0] = masks[i, 0] * (~mask_area)
                else:
                    # Rectangle mask
                    masks[i, 0, y:y + mask_h, x:x + mask_w] = 0

            return {'images': images, 'masks': masks, 'labels': labels}

    dataset_list = []
    if config.dataset_name == "CIFAR-100":
        transforms_cifar = transforms.Compose([
            transforms.Resize((config.image_size, config.image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), 
                                   (0.2675, 0.2565, 0.2761))])
        dataset_list_train = torchvision.datasets.CIFAR100(
            root=config.cache_dir,
            train=True,
            download=True,
            transform= transforms_cifar)
        dataset_list_test = torchvision.datasets.CIFAR100(
            root=config.cache_dir,
            train=False,
            download=True,
            transform= transforms_cifar)        
        dataset_list = ConcatDataset([dataset_list_train, dataset_list_test])
    else:
        dataset = load_dataset(config.dataset_name, 
                             split="train", 
                             cache_dir=config.cache_dir)
        dataset_list = dataset.with_transform(lambda x: _transform(x, dataset_name="CelebA"))

    if config.sub_datasets:
        dataset_list = torch.utils.data.Subset(
            dataset_list, 
            indices=range(min(500, len(dataset_list)))
        )

    train_dataloader = DataLoader(
        dataset_list,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn if config.session == 'mask' else None,
        num_workers=4
    )
    return train_dataloader

config = TrainingConfig()
if config.unet_model== 'sd':
    from SDFModel import DFUNetModel
    model = DFUNetModel(**config.params_sd).to(config.device)
elif config.unet_model== 'hf':
    model = UNet2DModel(**config.params_hf).to(config.device)
elif config.unet_model== 'dit':
    from Dit import DiT
    model = DiT(**config.params_dit)

if config.vae:
    vae = AutoencoderKL.from_pretrained(config.vae,
                                        subfolder="vae",
                                        cache_dir= config.cache_dir).to(config.device)
    vae.eval()
    config.image_size = int(config.image_size/8)

noise_scheduler = DDPMScheduler(num_train_timesteps= config.num_train_timesteps,
                                beta_start= config.beta_start,
                                beta_end= config.beta_end,
                                beta_schedule= 'scaled_linear')
optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

def evaluate(config, epoch, 
             pipeline=None, 
             model= None, 
             noise_scheduler= None, 
             accelerator= None):
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
    if model is None:
        images = pipeline(
            batch_size=config.eval_batch_size,
            generator=torch.Generator(device='cpu').manual_seed(config.seed),
        ).images
    else:
        device = accelerator.device
        channel = 4 if config.vae else 3
        model = accelerator.unwrap_model(model).to(device)
        model.eval()

        noise = torch.randn(
            (config.eval_batch_size, 
             channel, 
             config.image_size, 
             config.image_size),
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
                    predicted_noise = model(noise, t_tensor)
                    noise = noise_scheduler.step(predicted_noise, t, noise).prev_sample
                    pbar.update(1)
            
            images = (noise.clamp(-1, 1) + 1) / 2
    image_grid = make_image_grid(images, rows=4, cols=4)
    image_grid.save(f"{test_dir}/{epoch:04d}.png")
    model.train()

def train_loop(config: TrainingConfig, model, noise_scheduler, optimizer, vae=None):
    train_dataloader = load_datasets(config= TrainingConfig)
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer, 
                                                   num_warmup_steps=config.lr_warmup_steps, 
                                                   num_training_steps=len(train_dataloader) * config.num_epochs)
    kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)]
    accelerator = Accelerator(mixed_precision=config.mixed_precision, 
                              gradient_accumulation_steps=config.gradient_accumulation_steps,
                              log_with= "tensorboard",
                              project_dir=os.path.join(config.output_dir, "logs"),
                              kwargs_handlers= kwargs_handlers if config.unet_model== 'dit' else None
                              )

    if accelerator.is_main_process:
        os.makedirs(config.output_dir, exist_ok=True)
        accelerator.init_trackers("train_face_data")
    
    if config.vae:
        model, optimizer, train_dataloader, lr_scheduler, vae = accelerator.prepare(model, 
                                                                                    optimizer, 
                                                                                    train_dataloader, 
                                                                                    lr_scheduler, 
                                                                                    vae)
    else:
        model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(model, 
                                                                               optimizer, 
                                                                               train_dataloader, 
                                                                               lr_scheduler)
    if accelerator.is_main_process:
        print("*"*20, f"\nModle Name: {config.unet_model} Batch Size: {config.batch_size} Data Nums: {len(train_dataloader)} VAE: {config.vae}\n", "*"*20)

    global_step = 0
    for epoch in range(config.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader), 
                            disable=not accelerator.is_local_main_process, desc=f"Epoch {epoch}")
        for batch in train_dataloader:
            # 获取数据
            if config.dataset_name== "CIFAR-100":
                clean_images, labels = batch
                labels.to(accelerator.device)
            else:
                clean_images = batch['images']
            clean_images= clean_images.to(accelerator.device)
            masks = batch['masks'].to(accelerator.device) if config.session == 'mask' else None

            # 添加噪声
            if config.vae is not None:
                with torch.no_grad():
                    try:
                        latents = vae.module.encode(clean_images).latent_dist.sample() * 0.18215
                    except Exception:
                        latents = vae.encode(clean_images).latent_dist.sample() * 0.18215
                    noise = torch.rand_like(latents)
                    timesteps = torch.randint(0, 
                                              noise_scheduler.config.num_train_timesteps, 
                                              (clean_images.shape[0],), 
                                              device=clean_images.device, 
                                              dtype=torch.int64)
                    noisy_images = noise_scheduler.add_noise(latents, noise, timesteps)
            else:
                noise = torch.randn(clean_images.shape, device=clean_images.device)
                timesteps = torch.randint(0, 
                                        noise_scheduler.config.num_train_timesteps, 
                                        (clean_images.shape[0],), 
                                        device=clean_images.device, 
                                        dtype=torch.int64)
                noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            if config.session == 'mask':
                noisy_images = masks * clean_images + (1 - masks) * noisy_images

            # 模型训练
            with accelerator.accumulate(model):
                if config.unet_model== 'hf':
                    if config.session== 'condiction':
                        noise_pred = model(noisy_images, 
                                           timesteps, 
                                           labels,
                                           return_dict=False)[0]
                    else:
                        noise_pred = model(noisy_images, 
                                           timesteps, 
                                           return_dict=False)[0]
                elif config.unet_model in ['sd', 'dit']:
                    if config.session== 'condiction':
                        #BUG: 对于sd标签可能需要编码
                        noise_pred = model(noisy_images, 
                                           timesteps,
                                           labels)
                    else:
                        noise_pred = model(noisy_images, timesteps)
                
                if config.session == 'mask':
                    loss = F.mse_loss(noise_pred * (1 - masks), 
                                      noise * (1 - masks))
                else:
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

        if accelerator.is_main_process:
            if config.unet_model== 'hf':
                pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), 
                                        scheduler=noise_scheduler)
                if (epoch+ 1)%10 ==0 or epoch== config.num_epochs-1:
                    evaluate(config, epoch, pipeline)
                if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                    pipeline.save_pretrained(config.output_dir)
                    print(f"Saved pipeline to {config.output_dir}")
            elif config.unet_model in ['sd', 'dit']:
                if (epoch+ 1)%10 ==0 or epoch== config.num_epochs-1:
                    evaluate(config, epoch, 
                             model= model,
                             noise_scheduler= noise_scheduler,
                             accelerator= accelerator)
                if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                    model_to_save = accelerator.unwrap_model(model)
                    torch.save({
                        'model_state_dict': model_to_save.state_dict(),
                        'epoch': epoch,
                        'config': vars(config)
                    }, f"{config.output_dir}/model_checkpoint_{epoch}.pth")
                    noise_scheduler.save_config(config.output_dir)
                    print(f"Saved state_dict to {config.output_dir}/model_checkpoint.pth")

    # 推理部分
    if accelerator.is_main_process:
        n_samples, image_size = 5, config.image_size
        try:
            scheduler = DDPMScheduler.from_pretrained(config.output_dir)
        except Exception:
            scheduler = DDPMScheduler.from_pretrained(f"{config.output_dir}/scheduler")
        scheduler.set_timesteps(config.num_train_timesteps)
        
        if config.session == 'mask':
            clean_images = load_clean_images('./face_clean.jpg')
            masks = torch.ones(n_samples, 1, image_size, image_size, device=config.device)
            mask_h = int(image_size * torch.rand(1).item() * 0.5)
            mask_w = int(image_size * torch.rand(1).item() * 0.5)
            y = int(torch.rand(1).item() * (image_size - mask_h))
            x = int(torch.rand(1).item() * (image_size - mask_w))
            masks[:, :, y:y+mask_h, x:x+mask_w] = 0
        elif config.session == 'normal':
            clean_images = torch.randn(n_samples, 
                                       3, 
                                       image_size, 
                                       image_size).to(config.device)
            masks = None
            labels= None
        elif config.session== 'condiction':
            clean_images = torch.randn(n_samples, 
                                       3, 
                                       image_size, 
                                       image_size).to(config.device)
            masks= None
            labels = torch.tensor([i for i in range(0, n_samples)],
                                  dtype= torch.int,
                                  device= config.device)
        
        noisy_images = torch.randn(n_samples, 
                                   3, 
                                   image_size, 
                                   image_size).to(clean_images.device)
        generate(model, 
                 noisy_images, 
                 scheduler, 
                 clean_images= clean_images, 
                 masks= masks,
                 labels= labels)
    
if __name__ == '__main__':
    # CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes=4 diffusion_model_train.py
    if config.vae:
        train_loop(config, model, 
                   noise_scheduler, 
                   optimizer, vae)
    else:
        train_loop(config, model, 
                   noise_scheduler, 
                   optimizer, None)
