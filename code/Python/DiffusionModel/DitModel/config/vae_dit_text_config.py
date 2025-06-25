from dataclasses import dataclass
import torch

@dataclass
class DitConfig:
    training = 'dit'
    # 学习率调整
    learning_rate = 1e-4
    lr_warmup_steps = 1000 # 选择10-20% 50*100*0.1

    eval_batch_size= 16 
    epochs = 1
    sub_dataset = True if epochs==1 else False

    gradient_accumulation_steps = 1
    save_model_epochs = 20
    mixed_precision = "fp16"
    seed = 0
    
    # 配置线性方差调度
    num_train_timesteps = 10
    beta_start = 1e-4
    beta_end = 2e-2

    # data
    dataset_name = "Flicker30k"
    text_dir = "/data/huangjie/flickr30k/captions.txt"
    image_dir = "/data/huangjie/flickr30k/Images/"
    batch_size_dict = {"Flicker30k": 5, "CIFAR-100": 32, "CelebA-HQ": 128}
    image_size_dict = {"Flicker30k": 512, "CIFAR-100": 32, "CelebA-HQ": 128}
    image_size = image_size_dict[dataset_name]
    batch_size = batch_size_dict[dataset_name]
    channel = 3

    # model
    vae = "stabilityai/stable-diffusion-2-1"
    text_model = None
    params_dit = {
        'depth': 28,
        'hidden_size': 1152, 
        'patch_size': 8, 
        'num_heads': 16,
        'input_size': image_size,
        'in_channels': 3,
        'learn_sigma': False
    }

    # store
    output_dir = f"./DFModelTraining/DDPM-Dit-{dataset_name}"
    store_path = f"{output_dir}/Image/"
    cache_dir = '/data/huangjie'

    device= 'cuda' if torch.cuda.is_available() else 'cpu'