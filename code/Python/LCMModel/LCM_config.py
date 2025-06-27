from dataclasses import dataclass

@dataclass
class LCMTrainingConfig:
    project_name = 'LCMModel'
    # data config
    train_data_path = "pipe:curl -L -s https://huggingface.co/datasets/laion/conceptual-captions-12m-webdataset/resolve/main/data/{00000..01099}.tar?download=true"
    num_train_samples = 4000000
    resolution = 512
    train_batch_size = 44
    num_workers = 4

    # accelerator config
    gradient_accumulation_steps = 1
    mixed_precision = 'fp16'
    log_with = 'tensorboard'

    # model config
    teacher_model = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    num_ddim_timesteps = 50
    vae_encode_batch_size = 32 # 避免vae encode 导致OOM
    w_max = 15
    w_min = 5

    # training config
    xformer = False # 对显卡版本有要求
    use_8bit_adam = False
    gradient_checkpointing = False
    checkpointing_steps = 500
    validation_steps = 200
    checkpoints_total_limit= 10
    epochs = 100
    timestep_scaling_factor = 10
    loss_type = "huber"
    huber_c = 0.001
    ema_decay = 0.95
    
    # optim config
    lr_rate = 1e-6
    adam_beta1 = 0.9
    adam_beta2 = 0.999
    weight_decay = 1e-2
    epsilon = 1e-8
    max_grad_norm = 1
    lr_scheduler = "constant"
    lr_warmup_steps = 500
    # max_train_steps = 1000
    max_train_steps = 50000

    # store config
    out_dir = './LCMModelTraining'
    cache_dir = '/data/huangjie/'

    # other conifg
    seed = 20250625
    resume_from_checkpoint = None
