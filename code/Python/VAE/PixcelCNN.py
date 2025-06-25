import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskedConv2d(nn.Conv2d):
    def __init__(self, mask_type, *args, **kwargs):
        super(MaskedConv2d, self).__init__(*args, **kwargs)
        assert mask_type in ['A', 'B']
        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        if mask_type == 'A':
            self.mask[:, :, kH//2, kW//2:] = 0
            self.mask[:, :, kH//2+1:, :] = 0
        else:
            self.mask[:, :, kH//2, kW//2+1:] = 0
            self.mask[:, :, kH//2+1:, :] = 0
        self.weight.data *= self.mask
        self.weight.register_hook(lambda grad: grad * self.mask)
        # 使用 Xavier 初始化
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        return super(MaskedConv2d, self).forward(x)

class GatedBlock(nn.Module):
    def __init__(self, mask_type, in_channels, out_channels, kernel_size=3, padding=1):
        super(GatedBlock, self).__init__()
        self.mask_type = mask_type
        # Vertical stack: 仅依赖上方像素
        self.vertical_conv = MaskedConv2d(mask_type, in_channels, out_channels * 2, kernel_size=kernel_size, padding=padding)
        self.vertical_to_horizontal = nn.Conv2d(out_channels * 2, out_channels * 2, kernel_size=1)
        # Horizontal stack: 依赖左侧和上方像素
        self.horizontal_conv = MaskedConv2d(mask_type, in_channels, out_channels * 2, kernel_size=(1, kernel_size), padding=(0, padding))
        self.horizontal_residual = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.gate = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.1)
        # 初始化权重
        nn.init.xavier_uniform_(self.vertical_to_horizontal.weight)
        nn.init.xavier_uniform_(self.horizontal_residual.weight)

    def forward(self, v_input, h_input):
        # Vertical stack
        v_out = self.vertical_conv(v_input)
        v_out = self.dropout(v_out)
        v_out_tanh, v_out_gate = v_out.chunk(2, dim=1)
        v_out = torch.tanh(v_out_tanh) * self.gate(v_out_gate)
        
        # Horizontal stack
        h_out = self.horizontal_conv(h_input)
        v_to_h = self.vertical_to_horizontal(v_out)
        h_out = h_out + v_to_h
        h_out = self.dropout(h_out)
        h_out_tanh, h_out_gate = h_out.chunk(2, dim=1)
        h_out = torch.tanh(h_out_tanh) * self.gate(h_out_gate)
        
        # Residual connection
        h_residual = self.horizontal_residual(h_input)
        h_out = h_out + h_residual
        
        return v_out, h_out

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = MaskedConv2d('B', channels, channels//2, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(channels//2, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = MaskedConv2d('B', channels//2, channels//2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels//2, momentum=0.1)
        self.conv3 = MaskedConv2d('B', channels//2, channels, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(channels, momentum=0.1)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.dropout(out)
        out += residual  # 残差连接
        out = self.relu(out)
        return out

class PixelCNN(nn.Module):
    def __init__(self, input_shape, num_embeddings, num_layers= 15):
        super(PixelCNN, self).__init__()
        self.input_shape = input_shape
        self.num_embeddings = num_embeddings

        channels = 128
        layers = [
            MaskedConv2d('A', 1, channels, kernel_size=7, padding=3),
            nn.BatchNorm2d(channels, momentum=0.1),
            nn.ReLU(inplace=True)
        ]
        for _ in range(num_layers):
            layers.append(ResidualBlock(channels))
            layers.append(nn.Dropout(p=0.1))
        layers.extend([
            nn.Conv2d(channels, channels*2, kernel_size=1),
            nn.BatchNorm2d(channels*2, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels*2, num_embeddings, kernel_size=1)
        ])

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class GatedPixelCNN(nn.Module):
    def __init__(self, input_shape, num_embeddings, num_layers=15):
        super(GatedPixelCNN, self).__init__()
        self.input_shape = input_shape
        self.num_embeddings = num_embeddings
        channels = 128  # 增加通道数

        # 初始卷积
        self.input_conv = MaskedConv2d('A', 1, channels, kernel_size=7, padding=3)
        
        # Gated blocks
        self.gated_blocks = nn.ModuleList()
        for i in range(num_layers):
            mask_type = 'A' if i == 0 else 'B'
            self.gated_blocks.append(GatedBlock(mask_type, channels, channels))
        
        # 输出层
        self.output_layers = nn.Sequential(
            nn.BatchNorm2d(channels, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels * 2, kernel_size=1),
            nn.BatchNorm2d(channels * 2, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels * 2, num_embeddings, kernel_size=1)
        )
        # 初始化权重
        nn.init.xavier_uniform_(self.input_conv.weight)
        for layer in self.output_layers:
            if isinstance(layer, nn.Conv2d):
                nn.init.xavier_uniform_(layer.weight)

    def forward(self, x):
        v_input = self.input_conv(x)
        h_input = v_input  # 初始水平输入与垂直输入相同
        for gated_block in self.gated_blocks:
            v_input, h_input = gated_block(v_input, h_input)
        out = self.output_layers(h_input)
        return out

class PixelCNNPlusPlus(nn.Module):
    def __init__(self, input_shape, num_embeddings, num_logistic_mix=10, channels=128):
        super(PixelCNNPlusPlus, self).__init__()
        self.input_shape = input_shape
        self.num_embeddings = num_embeddings
        self.num_logistic_mix = num_logistic_mix
        
        # Initial convolution
        self.initial_conv = MaskedConv2d('A', 1, channels, kernel_size=7, padding=3)
        
        # Downsampling path
        self.downsample = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=False)
            ),
            nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(num_features=channels),
                nn.ReLU(inplace=False)
            )
        ])
        
        # Residual blocks
        self.res_blocks = nn.ModuleList([
            ResidualBlock(channels) for _ in range(5)
        ])
        
        # Upsampling path
        self.upsample = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(channels, channels, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=False)
            ),
            nn.Sequential(
                nn.ConvTranspose2d(channels, channels, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=False)
            )
        ])
        
        # Final layers for mixture of logistics
        self.final_layers = nn.Sequential(
            nn.Conv2d(channels, channels * 2, kernel_size=1),
            nn.BatchNorm2d(channels * 2, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels * 2, num_embeddings, kernel_size=1)
        )
        for layer in self.final_layers:
            if isinstance(layer, nn.Conv2d):
                nn.init.xavier_uniform_(layer.weight)
        self.final_conv = nn.Conv2d(channels, num_logistic_mix * 4, kernel_size=1)
    
    def forward(self, x):
        # Initial convolution
        out = self.initial_conv(x)
        
        # Downsampling
        skip_connections = []
        for down in self.downsample:
            out = down(out)
            skip_connections.append(out)
        
        # Residual blocks
        for res_block in self.res_blocks:
            out = res_block(out)
        
        # Upsampling with skip connections
        for i, up in enumerate(self.upsample):
            out = up(out)
            # Crop skip connection to match output size if necessary
            skip = skip_connections[-(i+1)]
            if out.shape[2:] != skip.shape[2:]:
                skip = F.interpolate(skip, size=out.shape[2:], mode='bilinear', align_corners=False)
            out = out + skip  # Changed from out += skip to avoid inplace operation
        
        # Output mixture parameters
        out = self.final_conv(out)
        return out

