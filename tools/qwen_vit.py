import torch
from transformers import AutoConfig
from transformers import AutoProcessor
from transformers.models.qwen2_vl.modeling_qwen2_vl import (
    Qwen2VisionTransformerPretrainedModel,
)
from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor
from safetensors.torch import load_file
import time
import os
from transformers.models.qwen2_vl.configuration_qwen2_vl import Qwen2VLConfig, Qwen2VLVisionConfig

visual_model = None
processor = None
device = "cuda"



def load_model():
    # import ipdb;ipdb.set_trace()
    global visual_model, processor
    model_path = "/home/mineru/Document/niujunbo/github/NativeRes-LLaVA/qwen2-vl-vit"
    # config = AutoConfig.from_pretrained(model_path)
    # print(type(config.vision_config))
    config = Qwen2VLVisionConfig.from_pretrained(model_path)
    print(config)

    visual_model = Qwen2VisionTransformerPretrainedModel._from_config(
        config=config,
        use_flash_attention_2=True
    ).to(dtype=torch.bfloat16)
    print(visual_model)
    return
    checkpoint_path = os.path.join(model_path, "model-00001-of-00002.safetensors")
    print("开始加载权重")
    checkpoint = load_file(checkpoint_path)
    print("权重文件加载完成")
    visual_weights = {
        key.replace("visual.", ""): value
        for key, value in checkpoint.items()
        if key.startswith("visual.")
    }
    visual_model.load_state_dict(visual_weights, strict=True)
    print("权重替换完成")
    visual_model.to(device)
    visual_model.eval()
    processor = AutoProcessor.from_pretrained(model_path)
    
def load_and_save_model():
    # import ipdb;ipdb.set_trace()
    global visual_model, processor
    model_path = "/mnt/hwfile/mllm/niujunbo/model-image/Qwen/Qwen2-VL-2B-Instruct"
    config = AutoConfig.from_pretrained(model_path)
    visual_model = Qwen2VisionTransformerPretrainedModel._from_config(
        config=config.vision_config,
        use_flash_attention_2=True
    ).to(dtype=torch.bfloat16)
    checkpoint_path = os.path.join(model_path, "model-00001-of-00002.safetensors")
    print("开始加载权重")
    checkpoint = load_file(checkpoint_path)
    print("权重文件加载完成")
    visual_weights = {
        key.replace("visual.", ""): value
        for key, value in checkpoint.items()
        if key.startswith("visual.")
    }
    visual_model.load_state_dict(visual_weights, strict=True)
    print("权重替换完成")
    visual_model.to(device)
    visual_model.eval()
    processor = AutoProcessor.from_pretrained(model_path)
    
    save_path = os.path.join("/mnt/petrelfs/niujunbo/NativeRes/model", "qwen2vl-665m-patch14-native")
    os.makedirs(save_path, exist_ok=True)
    visual_model.save_pretrained(save_path)
    processor.save_pretrained(save_path)
    print(visual_model)
    print(f"模型和处理器已保存到: {save_path}")

from PIL import Image
if __name__ == "__main__":
    load_and_save_model()

