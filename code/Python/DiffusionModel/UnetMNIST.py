import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from dataclasses import dataclass
from tqdm.auto import tqdm
import os
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import re

@dataclass
class TrainingConfig:
    image_size = 28  # FashionMNIST image size
    train_batch_size = 64
    eval_batch_size = 16
    num_epochs = 10
    learning_rate = 1e-4
    num_timesteps = 1000
    beta_start = 1e-4
    beta_end = 0.02
    output_dir = "ddpm-fashionmnist"
    seed = 0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Simple UNet model
class SimpleUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(SimpleUNet, self).__init__()
        self.down1 = nn.Conv2d(in_channels, 64, 3, padding=1)
        self.down2 = nn.Conv2d(64, 128, 3, padding=1, stride=2)
        self.bottleneck = nn.Conv2d(128, 256, 3, padding=1)
        self.up1 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1)
        self.up2 = nn.Conv2d(128 + 64, 64, 3, padding=1)
        self.final = nn.Conv2d(64, out_channels, 3, padding=1)
        self.time_emb = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )

    def forward(self, x, t):
        t_emb = self.time_emb(t.view(-1, 1).float() / 1000).view(-1, 64, 1, 1)
        x1 = F.relu(self.down1(x))
        x2 = F.relu(self.down2(x1))
        x3 = F.relu(self.bottleneck(x2))
        x4 = F.relu(self.up1(x3))
        x4 = F.interpolate(x4, size=x1.shape[2:], mode='bilinear', align_corners=False)
        x5 = F.relu(self.up2(torch.cat([x4, x1], dim=1)))
        return self.final(x5 + t_emb)

# Custom DDPM Scheduler
class DDPMScheduler:
    def __init__(self, num_timesteps, beta_start, beta_end):
        self.num_timesteps = num_timesteps
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0).to(config.device)
        self.timesteps = torch.arange(num_timesteps - 1, -1, -1)

    def add_noise(self, x, noise, t):
        sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod[t]).view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_cumprod = torch.sqrt(1 - self.alpha_cumprod[t]).view(-1, 1, 1, 1)
        return sqrt_alpha_cumprod * x + sqrt_one_minus_alpha_cumprod * noise

    def step(self, noise_pred, t, x):
        alpha = self.alphas[t].view(-1, 1, 1, 1)
        alpha_cumprod = self.alpha_cumprod[t].view(-1, 1, 1, 1)
        beta = self.betas[t].view(-1, 1, 1, 1)
        prev_x = (x - (1 - alpha) * noise_pred / torch.sqrt(1 - alpha_cumprod)) / torch.sqrt(alpha)
        return prev_x

# Data loading (FashionMNIST)
def get_data(config):
    transform = transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    return DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True)

# Training loop
def train_loop(config, model, scheduler, optimizer, dataloader):
    model.to(config.device)
    os.makedirs(config.output_dir, exist_ok=True)
    global_step = 0

    for epoch in range(config.num_epochs):
        progress_bar = tqdm(total=len(dataloader), desc=f"Epoch {epoch}")
        for batch, _ in dataloader:
            batch = batch.to(config.device)
            noise = torch.randn_like(batch)
            timesteps = torch.randint(0, scheduler.num_timesteps, (batch.shape[0],), device=config.device, dtype=torch.long)
            noisy_images = scheduler.add_noise(batch, noise, timesteps)

            optimizer.zero_grad()
            noise_pred = model(noisy_images, timesteps)
            loss = F.mse_loss(noise_pred, noise)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            progress_bar.update(1)
            progress_bar.set_postfix({"loss": loss.item()})
            global_step += 1

        # Save model
        if (epoch + 1) % 5 == 0 or epoch == config.num_epochs - 1:
            torch.save(model.state_dict(), f"{config.output_dir}/model_epoch_{epoch}.pt")

# Generation function
def generate(model, noise_image, scheduler, device='cuda', store_path='./FashionMNIST_generated/'):
    def save_samples(samples, title="Generated Samples"):
        fig, axes = plt.subplots(1, len(samples), figsize=(len(samples) * 2, 2))
        for i, img in enumerate(samples):
            axes[i].imshow(img.permute(1, 2, 0).numpy() * 0.5 + 0.5, cmap='gray')
            axes[i].axis('off')
        plt.title(title)
        plt.savefig(f"{store_path}/{title.replace(' ', '_')}.png")
        plt.close()

    def generate_gif(path_dir):
        images = []
        path_list = sorted(os.listdir(path_dir), key=lambda x: int(re.search(r'(\d+)', x).group(1)) if re.search(r'(\d+)', x) else x, reverse=True)
        for path in path_list:
            img_path = os.path.join(path_dir, path)
            if os.path.exists(img_path):
                images.append(imageio.imread(img_path))
        if images:
            name = os.path.basename(os.path.normpath(path_dir))
            imageio.mimsave(f'./{name}_Generate_image.gif', images, fps=2)

    os.makedirs(store_path, exist_ok=True)
    model.eval()
    with torch.no_grad():
        x = noise_image.to(device)
        for t in scheduler.timesteps:
            t_tensor = torch.full((x.shape[0],), t, dtype=torch.long, device=device)
            pred_noise = model(x, t_tensor)
            x = scheduler.step(pred_noise, t, x)
            tmp_x = (x.clamp(-1, 1) + 1) / 2
            save_samples(tmp_x.cpu(), f'{t.item()}')
        x = (x.clamp(-1, 1) + 1) / 2
        generate_gif(store_path)
        return x.cpu()

if __name__ == '__main__':
    config = TrainingConfig()
    model = SimpleUNet(in_channels=1, out_channels=1)
    scheduler = DDPMScheduler(config.num_timesteps, config.beta_start, config.beta_end)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    dataloader = get_data(config)

    # Train
    train_loop(config, model, scheduler, optimizer, dataloader)

    # Generate
    noise_image = torch.randn((config.eval_batch_size, 1, config.image_size, config.image_size))
    generated_images = generate(model, noise_image, scheduler, device=config.device, store_path=config.output_dir + '/generated/')