import os
import matplotlib.pyplot as plt
import pickle
import imageio
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.cuda import is_available as cuda_available
from collections import defaultdict

VAE_CONFIG = {
    'batch_size': 128,
    'learning_rate': 1e-3,
    'epochs': 100,
    'latent_dim': 20,
    'image_size': 28 * 28,
    'results_dir': 'MNIST_VAE_results',
    'device': 'cuda:0' if cuda_available() else 'cpu'
}

class VAE(nn.Module):
    def __init__(self, latent_dim=20, input_dim=784):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_var = nn.Linear(128, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_var(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        x = x.view(-1, VAE_CONFIG['image_size'])
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

def vae_loss(recon_x, x, mu, logvar):
    recon_x = torch.clamp(recon_x, 1e-8, 1-1e-8)
    x = x.view(-1, VAE_CONFIG['image_size'])
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return (BCE + KLD) / x.size(0)  # 平均损失

def train_vae():
    # 创建结果目录
    os.makedirs(VAE_CONFIG['results_dir'], exist_ok=True)
    os.makedirs(os.path.join(VAE_CONFIG['results_dir'], 'Reconstruction_results'), exist_ok=True)
    os.makedirs(os.path.join(VAE_CONFIG['results_dir'], 'Fixed_results'), exist_ok=True)
    os.makedirs(os.path.join(VAE_CONFIG['results_dir'], 'Random_results'), exist_ok=True)

    # 初始化模型和优化器
    model = VAE(VAE_CONFIG['latent_dim'], VAE_CONFIG['image_size']).to(VAE_CONFIG['device'])
    optimizer = optim.Adam(model.parameters(), lr=VAE_CONFIG['learning_rate'])

    # 数据加载
    transform = transforms.ToTensor()
    train_loader = DataLoader(
        datasets.MNIST('data', train=True, download=True, transform=transform),
        batch_size=VAE_CONFIG['batch_size'], shuffle=True, num_workers=2, pin_memory=True
    )
    test_loader = DataLoader(
        datasets.MNIST('data', train=False, transform=transform),
        batch_size=25, shuffle=True
    )

    history = {'loss': []}
    fix_z = torch.randn(25, VAE_CONFIG['latent_dim']).to(VAE_CONFIG['device'])
    for epoch in range(VAE_CONFIG['epochs']):
        model.train()
        train_loss = 0
        for data, _ in train_loader:
            data = data.to(VAE_CONFIG['device'])
            optimizer.zero_grad()
            recon_data, mu, logvar = model(data)
            loss = vae_loss(recon_data, data, mu, logvar)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        avg_loss = train_loss / len(train_loader)
        history['loss'].append(avg_loss)
        print(f'Epoch [{epoch+1}/{VAE_CONFIG["epochs"]}] Loss: {avg_loss:.4f}')

        # 保存重构和随机生成图像
        model.eval()
        with torch.no_grad():
            # 重构图像
            test_data, _ = next(iter(test_loader))
            test_data = test_data.to(VAE_CONFIG['device'])
            recon_data, _, _ = model(test_data)
            comparison = torch.cat([test_data.cpu(), recon_data.view(25, 1, 28, 28).cpu()])
            save_image(
                comparison,
                os.path.join(VAE_CONFIG['results_dir'], 'Reconstruction_results', f'epoch_{epoch+1}.png'),
                nrow=5
            )

            # 随机生成图像
            z = torch.randn(25, VAE_CONFIG['latent_dim']).to(VAE_CONFIG['device'])
            generated = model.decode(z).view(25, 1, 28, 28).cpu()
            save_image(
                generated,
                os.path.join(VAE_CONFIG['results_dir'], 'Random_results', f'epoch_{epoch+1}.png'),
                nrow=5
            )

            # 固定样本输出
            generated = model.decode(fix_z).view(25, 1, 28, 28).cpu()
            save_image(
                generated,
                os.path.join(VAE_CONFIG['results_dir'], 'Fixed_results', f'epoch_{epoch+1}.png'),
                nrow=5
            )

    # 保存模型和训练历史
    torch.save(model.state_dict(), os.path.join(VAE_CONFIG['results_dir'], 'vae_model.pth'))
    with open(os.path.join(VAE_CONFIG['results_dir'], 'train_history.pkl'), 'wb') as f:
        pickle.dump(history, f)

    # 绘制训练曲线
    plt.figure(figsize=(8, 4))
    plt.plot(history['loss'], label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(VAE_CONFIG['results_dir'], 'training_history.png'))
    plt.close()

    # 生成动画
    images = defaultdict(list)
    for name in ['Reconstruction_results', 'Random_results', 'Fixed_results']:
        for e in range(VAE_CONFIG['epochs']):
            img_path = os.path.join(VAE_CONFIG['results_dir'], name, f'epoch_{e+1}.png')
            if os.path.exists(img_path):
                images[name].append(imageio.v2.imread(img_path))
    if images:
        for name, image_path in images.items():
            imageio.mimsave(os.path.join(VAE_CONFIG['results_dir'], f'{name}.gif'), image_path, fps=5)

if __name__ == '__main__':
    train_vae()