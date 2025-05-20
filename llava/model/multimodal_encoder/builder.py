import os
from .clip_encoder import CLIPVisionTower
from .siglip_encoder import SigLipVisionTower
from .clip_encoder import CLIPVisionTower, CLIPVisionTowerS2
from .qwen_encoder import Qwen2VisionTransformerPretrainedModelForLLaVA
from llava.utils import rank0_print

def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, "mm_vision_tower", getattr(vision_tower_cfg, "vision_tower", None))
    use_s2 = getattr(vision_tower_cfg, "s2", False)
    rank0_print(f"Loading vision tower: {vision_tower}")
    if "clip" in vision_tower or vision_tower.startswith("openai") or vision_tower.startswith("laion") or "ShareGPT4V" in vision_tower:
        if use_s2:
            return CLIPVisionTowerS2(vision_tower, args=vision_tower_cfg, **kwargs)
        else:
            return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    elif "siglip" in vision_tower:
        return SigLipVisionTower(vision_tower, vision_tower_cfg=vision_tower_cfg, **kwargs)
    elif "qwen" in vision_tower.split('/')[-1].lower():
        return Qwen2VisionTransformerPretrainedModelForLLaVA(vision_tower,args=vision_tower_cfg)

    raise ValueError(f"Unknown vision tower: {vision_tower}")
