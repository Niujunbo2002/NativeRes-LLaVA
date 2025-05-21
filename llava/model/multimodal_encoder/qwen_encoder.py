import os
from transformers.models.qwen2_vl.modeling_qwen2_vl import (
    Qwen2VisionTransformerPretrainedModel,
)
from transformers.models.qwen2_vl.configuration_qwen2_vl import Qwen2VLConfig, Qwen2VLVisionConfig
from transformers import AutoConfig, AutoProcessor
from safetensors.torch import load_file
from torch import nn
import torch
import warnings
# ANSI escape sequences for colored output
RED = '\033[91m'
GREEN = '\033[92m'
BLUE = '\033[94m'
RESET = '\033[0m'

class VisionRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(seq, self.inv_freq)
        return freqs

class Qwen2VisionTransformerPretrainedModelForLLaVA(nn.Module):
    def __init__(self, model_path,args):
        super().__init__()

        self.is_loaded = False
        self.model_path = model_path
        self.vision_tower_name = model_path
        self.select_layer = -1
        self.select_feature = 'patch'
        self.min_token=getattr(args,"mm_min_image_token",4)
        self.max_token=getattr(args,"mm_max_image_token",2048)
        self.resize_image_size=getattr(args,"resize_image_size",None)
        self.load_model(self.model_path)

    def load_model(self,model_path):
        config = Qwen2VLVisionConfig.from_pretrained(model_path)
        self.vision_tower = Qwen2VisionTransformerPretrainedModel._from_config(
            config=config,
            use_flash_attention_2=True,
        ).half()

        self.vision_tower.requires_grad_(False)
        self.is_loaded = True
        
    def reset_image_processor(self, min_tokens, max_tokens):
        min_pixels=min_tokens * 28 * 28
        max_pixels=max_tokens * 28 * 28
        self.image_processor = AutoProcessor.from_pretrained(self.model_path, 
                                                            min_pixels = min_pixels,
                                                            max_pixels = max_pixels
                                                            )
        self.image_processor.resize_image_size = self.resize_image_size
        # Simplified output format
        print(f"{GREEN}MIN_PIXELS: {min_tokens} * 28 * 28 \nMAX_PIXELS: {max_tokens} * 28 * 28{RESET}")
                
    def forward(self, pixel_values, grid_thw):
        """
        pixel_values:[all_seq_len,patch_size*patch_size*3*2]
        image_grid_thw:[num_img,3],每个长度为3的向量为[1,h,w],1表示时间,如果为video,则会大于1.h,w为图像的高和宽(以patch为单位)
        """
        image_features = self.vision_tower(pixel_values, grid_thw=grid_thw)#[all_seq_len//4,hidden_size(1536)]
        print(f"{GREEN}Number of image tokens: {image_features.shape[0]}{RESET}")
        return  image_features
    
    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size


if __name__=="__main__":
    config=AutoConfig.from_pretrained("/data/niujunbo/model/Qwen/Qwen2-VL-2B-Instruct")
