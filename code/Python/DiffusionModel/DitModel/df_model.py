import torch
import torch.nn as nn

class DiffusionModel(nn.Module):
    def __init__(self, 
                 df_model_name: str, 
                 params: dict, 
                 text_model_name: tuple = ('clip', 'ViT-B/32'),
                 cache_dir: str = '/data/huangjie/'):
        super().__init__()
        if df_model_name == 'dit':
            from Dit import DiffusionTranformer
            self.model = DiffusionTranformer(**params)
        else:
            raise ValueError("ERROR! Model Name must Dit")
        
        if text_model_name is not None:
            if text_model_name[0] == 'clip':
                import clip
                self.text_encoder, _ = clip.load(text_model_name[1], download_root=cache_dir)
        else:
            self.text_encoder = None
        
    def text_encode(self, text: tuple):
        # TODO: 实现文本编码逻辑
        pass
    
    def forward(self, x: torch.Tensor, t: torch.Tensor, text: tuple = None) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        if self.text_encoder is None:
            x = self.model(x, t)
            return x
        else:
            pass