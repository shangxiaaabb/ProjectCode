import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super(LayerNorm, self).__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            self.gamma = nn.Parameter(torch.ones(1, *normalized_shape))
            self.beta = nn.Parameter(torch.zeros(1, *normalized_shape))
        else:
            self.register_parameter('gamma', None)
            self.register_parameter('beta', None)

    def forward(self, x: torch.Tensor):
        '''layer norm: 直接处理batch之外维度的均值/标准差'''
        norm_dims = list(range(x.dim() - len(self.normalized_shape), x.dim()))
        mean = x.mean(dim=norm_dims, keepdim=True)
        var = x.var(dim=norm_dims, unbiased=False, keepdim=True)

        x_normalized = (x - mean) / torch.sqrt(var + self.eps)
        if self.elementwise_affine:
            x_normalized = x_normalized * self.gamma + self.beta
        return x_normalized

class RMSNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super(RMSNorm, self).__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            self.gamma = nn.Parameter(torch.ones(1, *normalized_shape))
            self.beta = nn.Parameter(torch.zeros(1, *normalized_shape))
        else:
            self.register_parameter('gamma', None)
            self.register_parameter('beta', None)

    def forward(self, x: torch.Tensor):
        norm_dims = list(range(x.dim() - len(self.normalized_shape), x.dim()))
        rms = torch.sqrt(torch.mean(x**2, dim=norm_dims, keepdim=True) + self.eps)
        x_normalized = x / rms
        if self.elementwise_affine:
            x_normalized = x_normalized * self.gamma + self.beta

        return x_normalized

class BatchNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super(BatchNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        if self.affine:
            self.gamma = nn.Parameter(torch.ones(1, num_features, 1, 1))
            self.beta = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        else:
            self.register_parameter('gamma', None)
            self.register_parameter('beta', None)

        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(1, num_features, 1, 1))
            self.register_buffer('running_var', torch.ones(1, num_features, 1, 1))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_buffer('running_mean', None)
            self.register_buffer('running_var', None)
            self.register_buffer('num_batches_tracked', None)

    def forward(self, x: torch.Tensor):
        '''直接处理 channel 之外均值/标准差'''
        training = self.training

        if x.dim() == 3:  # (B, T, C) -> 转换为 (B, C, T, 1) 以统一处理
            x = x.transpose(1, 2).unsqueeze(-1)  # (B, C, T, 1)
            is_1d = True
        else:  # (B, C, H, W)
            is_1d = False

        if training:
            mean = x.mean(dim=(0, 2, 3), keepdim=True)  # 沿着 B, H, W（或 T）平均
            var = x.var(dim=(0, 2, 3), unbiased=False, keepdim=True)

            if self.track_running_stats:
                self.num_batches_tracked += 1
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            mean = self.running_mean if self.track_running_stats else x.mean(dim=(0, 2, 3), keepdim=True)
            var = self.running_var if self.track_running_stats else x.var(dim=(0, 2, 3), unbiased=False, keepdim=True)

        x_normalized = (x - mean) / torch.sqrt(var + self.eps)

        if self.affine:
            x_normalized = x_normalized * self.gamma + self.beta

        if is_1d:
            x_normalized = x_normalized.squeeze(-1).transpose(1, 2)  # (B, T, C)

        return x_normalized

class GlobalResponseNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super(GlobalResponseNorm, self).__init__()
        self.dim = dim
        self.eps = eps
        
        # 可学习的缩放和平移参数，初始化为 None，后续根据输入动态调整
        self.gamma = None
        self.beta = None

    def _init_parameters(self, shape, device):
        param_shape = [1, self.dim] + [1] * (len(shape) - 2)
        self.gamma = nn.Parameter(torch.ones(param_shape)).to(device)
        self.beta = nn.Parameter(torch.zeros(param_shape)).to(device)

    def forward(self, x):
        # 输入 x 的形状: (batch_size, channels, ...)，可能是 1D 或 2D
        if self.gamma is None or self.gamma.shape[1:] != x.shape[1:]:
            self._init_parameters(x.shape, x.device)
        
        spatial_dims = tuple(range(2, x.dim()))  # 对于 (N, C, L) 是 (2,)，对于 (N, C, H, W) 是 (2, 3)
        
        global_sum = torch.sum(x ** 2, dim=(0,) + spatial_dims, keepdim=True)
        
        norm = torch.sqrt(global_sum + self.eps)
        
        x_normalized = x / norm
        
        return self.gamma * x_normalized + self.beta

class InstanceNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super(InstanceNorm, self).__init__()
        self.dim = dim  # 通道数
        self.eps = eps  # 防止除零的小常数
        
        # 可学习的缩放和平移参数，初始化为 None，后续根据输入动态调整
        self.gamma = None
        self.beta = None

    def _init_parameters(self, shape, device):
        param_shape = [1, self.dim] + [1] * (len(shape) - 2)
        self.gamma = nn.Parameter(torch.ones(param_shape)).to(device)
        self.beta = nn.Parameter(torch.zeros(param_shape)).to(device)

    def forward(self, x):
        # 输入 x 的形状: (batch_size, channels, ...)，可能是 1D 或 2D
        if self.gamma is None or self.gamma.shape[1:] != x.shape[1:]:
            self._init_parameters(x.shape, x.device)
        
        spatial_dims = tuple(range(2, x.dim()))  # 对于 (N, C, L) 是 (2,)，对于 (N, C, H, W) 是 (2, 3)
        
        mean = x.mean(dim=spatial_dims, keepdim=True)
        var = x.var(dim=spatial_dims, keepdim=True)
        
        x_normalized = (x - mean) / torch.sqrt(var + self.eps)
        
        return self.gamma * x_normalized + self.beta
    
if __name__ == '__main__':
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    x0 = torch.randn(32, 100, 512).to(device)
    x1 = torch.randn(8, 3, 512, 512).to(device)

    layer_norm0 = LayerNorm((512)).to(device)
    layer_norm1 = LayerNorm((3, 512, 512)).to(device)

    out0 = layer_norm0(x0)
    out1 = layer_norm1(x1)
    print(f"Layer Norm:{out0.shape} {out1.shape}")

    batch_norm0 = BatchNorm(512).to(device)
    batch_norm1 = BatchNorm(3).to(device)
    out0 = batch_norm0(x0)
    out1 = batch_norm1(x1)
    print(f"Batch Norm: {out0.shape} {out1.shape}")

    rmse_norm0 = RMSNorm((512))
    rmse_norm1 = RMSNorm((3, 512, 512))
    rmse_norm0.to(device)
    rmse_norm1.to(device)

    out0 = rmse_norm0(x0)
    out1 = rmse_norm1(x1)
    print("RMSNorm:", out0.shape, out1.shape)

    grn_norm0 = GlobalResponseNorm(100).to(device)
    grn_norm1 = GlobalResponseNorm(3).to(device)
    out0 = grn_norm0(x0)
    out1 = grn_norm1(x1)
    print(f"GlobalResponseNorm: {out0.shape} {out1.shape}")

    ins_norm0 = InstanceNorm(100).to(device)
    ins_norm1 = InstanceNorm(3).to(device)
    out0 = ins_norm0(x0)
    out1 = ins_norm1(x1)
    print(f"InstanceNorm: {out0.shape} {out1.shape}")
