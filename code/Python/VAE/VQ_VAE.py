import os
import matplotlib.pyplot as plt
import argparse
import pickle
import imageio
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader, TensorDataset

from PixcelCNN import *

VQVAE_CONFIG = {
    'warmup_epochs': 5,
    'batch_size': 128,
    'learning_rate': 1e-4,
    'epochs': 200,
    'pixelcnn_lr': 1e-4,
    'pixelcnn_epochs': 200,
    'save_frc': 2,
    'latent_dim': 16,
    'num_embeddings': 512,
    'image_size': 32 * 32 * 3,
    'results_dir': 'CIFAR10_VQVAE_results',
    'model_type': 'pixelcnn',
    'device': 'cuda:0' if torch.cuda.is_available() else 'cpu'
}

class LinearWarmupScheduler:
    def __init__(self, optimizer, warmup_steps, target_lr):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.target_lr = target_lr
        self.current_step = 0
    
    def step(self):
        self.current_step += 1
        if self.current_step <= self.warmup_steps:
            lr = self.target_lr * self.current_step / self.warmup_steps
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        return self.current_step <= self.warmup_steps

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25, return_indices=False):
        super(VectorQuantizer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.return_indices = return_indices
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        nn.init.uniform_(self.embeddings.weight, -1.0, 1.0)

    def forward(self, inputs):
        # 检查输入有效性
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        flat_input = inputs.view(-1, self.embedding_dim)

        # 计算距离
        distances = torch.cdist(flat_input, self.embeddings.weight, p=2) ** 2
        encoding_indices = torch.argmin(distances, dim=1)
        encodings = torch.zeros(encoding_indices.size(0), self.num_embeddings, 
                                device=inputs.device)
        encodings.scatter_(1, encoding_indices.unsqueeze(1), 1)

        # 量化
        quantized = torch.matmul(encodings, self.embeddings.weight).view(inputs.shape)
        
        e_latent_loss = torch.mean((quantized.detach() - inputs) ** 2)
        q_latent_loss = torch.mean((quantized - inputs.detach()) ** 2)
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # 直通估计器
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        if self.return_indices:
            encoding_indices = encoding_indices.view(inputs.shape[0], inputs.shape[1], inputs.shape[2])
            return quantized.permute(0, 3, 1, 2).contiguous(), loss, perplexity, encoding_indices
        return quantized.permute(0, 3, 1, 2).contiguous(), loss, perplexity

class VQVAE(nn.Module):
    def __init__(self, latent_dim=10, num_embeddings=256, return_indices=False):
        super(VQVAE, self).__init__()
        self.return_indices = return_indices
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, latent_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(latent_dim)
        )
        self.vq_layer = VectorQuantizer(num_embeddings, latent_dim,
                                        return_indices= self.return_indices)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        if self.return_indices:
            z_q, vq_loss, perplexity, encoding_indices = self.vq_layer(z)
            x_recon = self.decoder(z_q)
            return x_recon, vq_loss, perplexity, encoding_indices
        else:
            z_q, vq_loss, perplexity = self.vq_layer(z)
            x_recon = self.decoder(z_q)
            return x_recon, vq_loss, perplexity

def generate_samples(vqvae, pixelcnn, num_samples, input_shape, device, model_type='pixelcnn', num_embeddings=256):
    # torch.manual_seed(42)
    # torch.cuda.manual_seed(42)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    vqvae.eval()
    pixelcnn.eval()
    h, w = input_shape
    samples = torch.zeros(num_samples, 1, h, w, device=device)
    
    with torch.no_grad():
        for i in range(h):
            for j in range(w):
                logits = pixelcnn(samples)
                probs = F.softmax(logits[:, :, i, j], dim=1)
                samples[:, :, i, j] = torch.multinomial(probs, 1).float()
        
        # 将样本通过 VQ-VAE 解码器生成图像
        samples = samples.squeeze(1).long()  # [num_samples, h, w]
        z_q = vqvae.vq_layer.embeddings(samples)  # [num_samples, h, w, latent_dim]
        z_q = z_q.permute(0, 3, 1, 2).contiguous()  # [num_samples, latent_dim, h, w]
        generated_images = vqvae.decoder(z_q)  # [num_samples, 3, h', w']
    
    return generated_images

def generate_gif(path_dir):
    '''生成gif'''
    images = []
    for path in os.listdir(path_dir):
        img_path = os.path.join(path_dir, path)
        if os.path.exists(img_path):
            images.append(imageio.v2.imread(img_path))
    if images:
        name = os.path.basename(os.path.normpath(path_dir))
        imageio.mimsave(os.path.join(VQVAE_CONFIG['results_dir'], 
                                     f'{name}.gif'), images, fps=5)

def extract_latents(model, dataloader, save_path):
    model.eval()
    all_indices = []
    with torch.no_grad():
        for data, _ in dataloader:
            data = data.to(VQVAE_CONFIG['device'])
            _, _, _, indices = model(data)
            all_indices.append(indices.cpu().numpy())
    all_indices = np.concatenate(all_indices, axis=0)
    np.save(save_path, all_indices)
    return all_indices 

def vqvae_loss(recon_x, x, vq_loss):
    recon_x = torch.clamp(recon_x, 1e-6, 1-1e-6)
    x = torch.clamp(x, 0.0, 1.0)
    mse_loss = nn.functional.mse_loss(recon_x, x, reduction='mean')
    return mse_loss + vq_loss

def discretized_mix_logistic_loss(output, target, num_logistic_mix=10):
    '''计算 PixelCNN++ 的混合逻辑斯蒂分布损失'''
    pi, mu, s, _ = output  # coeff 未使用，忽略
    target = target.squeeze(1).float()  # [batch_size, h, w]
    target = target.unsqueeze(1).expand(-1, num_logistic_mix, -1, -1)  # [batch_size, num_logistic_mix, h, w]
    
    # Logistic distribution log-probability
    centered_x = target - mu
    inv_s = 1.0 / (s + 1e-5)
    cdf_plus = torch.sigmoid((centered_x + 0.5) * inv_s)
    cdf_minus = torch.sigmoid((centered_x - 0.5) * inv_s)
    log_prob = torch.log(cdf_plus - cdf_minus + 1e-7)
    
    # Mixture of logistics
    log_pi = F.log_softmax(pi, dim=1)
    loss = -torch.sum(log_pi + log_prob, dim=1).mean()
    return loss

def logistic_mixture_log_likelihood(output, target, num_logistic_mix=10):
    # output: [batch_size, num_logistic_mix * 4, height, width]
    # target: [batch_size, 1, height, width] or [batch_size, height, width]
    
    batch_size, _, height, width = output.shape
    target = target.squeeze(1)  # Ensure target is [batch_size, height, width]
    
    # Reshape output to separate mixture parameters
    output = output.view(batch_size, num_logistic_mix, 4, height, width)
    
    # Assume parameters: [weights, means, log_scales, extra_param]
    logit_probs = output[:, :, 0, :, :]  # Mixture weights (logits)
    means = output[:, :, 1, :, :]       # Means of logistics
    log_scales = output[:, :, 2, :, :]  # Log scales of logistics
    # extra_param = output[:, :, 3, :, :]  # (Optional, depending on your setup)
    
    # Apply softmax to mixture weights
    log_probs = F.log_softmax(logit_probs, dim=1)
    
    # Compute logistic log-likelihood for each mixture component
    target = target.unsqueeze(1)  # [batch_size, 1, height, width]
    inv_scales = torch.exp(-log_scales)  # [batch_size, num_logistic_mix, height, width]
    cdf_plus = torch.sigmoid((target + 1e-5) * inv_scales - means * inv_scales)
    cdf_minus = torch.sigmoid((target - 1e-5) * inv_scales - means * inv_scales)
    log_pdf = torch.log(cdf_plus - cdf_minus + 1e-7)
    
    # Sum log-likelihood across mixture components
    log_likelihood = log_probs + log_pdf
    log_likelihood = torch.logsumexp(log_likelihood, dim=1)  # [batch_size, height, width]
    
    # Compute negative log-likelihood loss
    loss = -log_likelihood.mean()
    return loss

def train_pixelcnn(model_type= 'pixelcnn'):
    assert model_type in ['pixelcnn', 'pixelcnn_plusplus', 'gatedpixelcnn']
    print(f'Model-{model_type} training.........')
    os.makedirs(os.path.join(VQVAE_CONFIG['results_dir'], model_type), exist_ok=True)
    
    transform = transforms.ToTensor()
    train_loader = DataLoader(
        datasets.CIFAR10('data', train=True, download=False, transform=transform),
        batch_size=VQVAE_CONFIG['batch_size'], shuffle=True, num_workers=2, pin_memory=True
    )

    if model_type == 'gated_pixelcnn':
        model = GatedPixelCNN(input_shape=(8, 8), num_embeddings=VQVAE_CONFIG['num_embeddings']).to(VQVAE_CONFIG['device'])
        criterion = nn.CrossEntropyLoss()
    elif model_type == 'pixelcnn_plusplus':
        model = PixelCNNPlusPlus(input_shape=(1, 8, 8), num_embeddings=VQVAE_CONFIG['num_embeddings']).to(VQVAE_CONFIG['device'])
        # criterion = discretized_mix_logistic_loss
        criterion = logistic_mixture_log_likelihood
        # criterion = nn.CrossEntropyLoss()
    else:
        model = PixelCNN(input_shape=(8, 8), num_embeddings=VQVAE_CONFIG['num_embeddings']).to(VQVAE_CONFIG['device'])
        criterion = nn.CrossEntropyLoss()

    vqvae = VQVAE(VQVAE_CONFIG['latent_dim'], VQVAE_CONFIG['num_embeddings'],
                  return_indices= True).to(VQVAE_CONFIG['device'])
    vqvae.load_state_dict(torch.load(
        os.path.join(VQVAE_CONFIG['results_dir'], 'vqvae_model_final.pth'),
        weights_only=True))

    indices = extract_latents(vqvae, train_loader, 
                              os.path.join(VQVAE_CONFIG['results_dir'], 'latent_indices.npy'))
    indices = torch.from_numpy(indices).long().to(VQVAE_CONFIG['device'])
    dataset = TensorDataset(indices.unsqueeze(1).float())
    dataloader = DataLoader(dataset, batch_size=VQVAE_CONFIG['batch_size'], shuffle=True)
    
    # Warm-up 设置
    optimizer = optim.Adam(model.parameters(), lr=VQVAE_CONFIG['pixelcnn_lr'])
    warmup_epochs = VQVAE_CONFIG['warmup_epochs']
    steps_per_epoch = len(dataloader)
    warmup_steps = warmup_epochs * steps_per_epoch
    lr_scheduler = LinearWarmupScheduler(optimizer, warmup_steps, 
                                         VQVAE_CONFIG['pixelcnn_lr'])
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                                  T_max=VQVAE_CONFIG['pixelcnn_epochs'] - warmup_epochs)
    if model_type == 'pixelcnn_plusplus':
        dropout_warmup_steps = 10 * steps_per_epoch
        dropout_target_p = 0.1
        dropout_current_step = 0

    history = {'loss': [], 'lr': []}
    for epoch in range(VQVAE_CONFIG['pixelcnn_epochs']):
        model.train()
        total_loss = 0

        for batch in dataloader:
            x = batch[0].to(VQVAE_CONFIG['device'])
            optimizer.zero_grad()
            if model_type == 'pixelcnn_plusplus' and dropout_current_step < dropout_warmup_steps:
                dropout_p = dropout_target_p * dropout_current_step / dropout_warmup_steps
                for module in model.modules():
                    if isinstance(module, nn.Dropout):
                        module.p = dropout_p
                dropout_current_step += 1

            if model_type == 'pixelcnn_plusplus':    
                output = model(x)
                loss = criterion(output, x)
            else:
                output = model(x)
                loss = criterion(output, x.squeeze(1).long())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if lr_scheduler.step():
                history['lr'].append(optimizer.param_groups[0]['lr'])
            elif epoch >= warmup_epochs:
                cosine_scheduler.step()
                history['lr'].append(optimizer.param_groups[0]['lr'])

        avg_loss = total_loss / len(dataloader)
        history['loss'].append(avg_loss)
        # 生成图像
        if epoch% VQVAE_CONFIG['save_frc'] ==0:
            with torch.no_grad():
                generated = generate_samples(vqvae, model, 25, 
                                            (8, 8), 
                                            VQVAE_CONFIG['device'],
                                            model_type)
                save_image(
                    generated.cpu(),
                    os.path.join(VQVAE_CONFIG['results_dir'], model_type, f'{model_type}_{epoch}.png'),
                    nrow=5
                )
            print(f'{model_type} Epoch [{epoch+1}/{VQVAE_CONFIG["pixelcnn_epochs"]}] Loss: {avg_loss:.4f} LR: {optimizer.param_groups[0]["lr"]:.6f}')
    # 保存模型
    torch.save(model.state_dict(), os.path.join(VQVAE_CONFIG['results_dir'], f'{model_type}_model.pth'))

    plt.figure(figsize=(10, 4))
    # plt.subplot(1, 2, 1)
    plt.plot(history['loss'])
    plt.title(f'{model_type} Training Loss')
    plt.xlabel('Epoch')
    plt.tight_layout()
    plt.savefig(os.path.join(VQVAE_CONFIG['results_dir'], f'training_{model_type}.png'))
    plt.close()

    generate_gif(path_dir= os.path.join(VQVAE_CONFIG['results_dir'], model_type))
        
def train_vqvae(training_model= 'both'):
    # 创建结果目录
    os.makedirs(VQVAE_CONFIG['results_dir'], exist_ok=True)
    os.makedirs(os.path.join(VQVAE_CONFIG['results_dir'], 'Reconstruction_results'), exist_ok=True)

    # 初始化模型和优化器
    model = VQVAE(VQVAE_CONFIG['latent_dim'], VQVAE_CONFIG['num_embeddings'],
                  return_indices= True).to(VQVAE_CONFIG['device'])
    optimizer = optim.Adam(model.parameters(), lr=VQVAE_CONFIG['learning_rate'])

    # 数据加载
    transform = transforms.ToTensor()
    train_loader = DataLoader(
        datasets.CIFAR10('data', train=True, download=False, transform=transform),
        batch_size=VQVAE_CONFIG['batch_size'], shuffle=True, num_workers=2, pin_memory=True
    )
    test_loader = DataLoader(
        datasets.CIFAR10('data', train=False, transform=transform),
        batch_size=25, shuffle=True
    )

    history = {'loss': [], 'perplexity': []}
    for epoch in range(VQVAE_CONFIG['epochs']):
        model.train()
        train_loss = 0
        train_perplexity = 0
        for data, _ in train_loader:
            data = data.to(VQVAE_CONFIG['device'])
            optimizer.zero_grad()
            recon_data, vq_loss, perplexity, _ = model(data)
            loss = vqvae_loss(recon_data, data, vq_loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
            train_perplexity += perplexity.item()

        avg_loss = train_loss / len(train_loader)
        avg_perplexity = train_perplexity / len(train_loader)
        history['loss'].append(avg_loss)
        history['perplexity'].append(avg_perplexity)
        print(f'Epoch [{epoch+1}/{VQVAE_CONFIG["epochs"]}] Loss: {avg_loss:.4f} Perplexity: {avg_perplexity:.4f}')

        if epoch% VQVAE_CONFIG["save_frc"] ==0:
            model.eval()
            with torch.no_grad():
                # 保存重构图像
                data, _ = next(iter(test_loader))
                data = data.to(VQVAE_CONFIG['device'])
                recon_data, _, _, _ = model(data)
                save_image(
                    torch.cat([data.cpu(), recon_data.cpu()], dim=0),
                    os.path.join(VQVAE_CONFIG['results_dir'], 'Reconstruction_results', f'epoch_{epoch+1}.png'),
                    nrow=5
                )

    # 保存模型和训练历史
    torch.save(model.state_dict(), os.path.join(VQVAE_CONFIG['results_dir'], 'vqvae_model_final.pth'))
    with open(os.path.join(VQVAE_CONFIG['results_dir'], 'train_history.pkl'), 'wb') as f:
        pickle.dump(history, f)

    # 绘制训练曲线
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'])
    plt.title('VQ-VAE Training Loss')
    plt.xlabel('Epoch')
    plt.subplot(1, 2, 2)
    plt.plot(history['perplexity'])
    plt.title('VQ-VAE Perplexity')
    plt.xlabel('Epoch')
    plt.tight_layout()
    plt.savefig(os.path.join(VQVAE_CONFIG['results_dir'], 'training_vavqe.png'))
    plt.close()
    generate_gif(os.path.join(VQVAE_CONFIG['results_dir'], 'Reconstruction_results'))
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='pixelcnn')
    args = parser.parse_args()

    if not os.path.exists('./CIFAR10_VQVAE_results/vqvae_model_final.pth'):
        print('Training VQ-VAE Model..........')
        train_vqvae()

    train_pixelcnn(args.model_type)