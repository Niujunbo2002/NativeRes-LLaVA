import sys
sys.path.append("/home/mineru/Document/niujunbo/github/NativeRes-LLaVA")
import argparse
import torch
import warnings
warnings.filterwarnings("ignore")
from transformers import logging
logging.set_verbosity_warning()
from transformers import logging
logging.set_verbosity_error()

from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)
from PIL import Image
import requests
from PIL import Image
from io import BytesIO
import re

# Define constants
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"
IMAGE_PLACEHOLDER = "<image-placeholder>"
# ANSI escape sequences for colored output
RED = '\033[91m'
GREEN = '\033[92m'
BLUE = '\033[94m'
RESET = '\033[0m'


def image_parser(args):
    out = args.image_file.split(args.sep)
    return out


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_files,packing):
    if packing:
        return image_files
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out


def eval_model(args):
    packing=False
    if 'qwen' in args.model_path.lower():
        packing=True
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)

    print(f"{GREEN}Loading model from {args.model_path} ...{RESET}")
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path,
        None, 
        model_name,
        min_image_tokens=args.min_image_tokens,
        max_image_tokens=args.max_image_tokens,
    )
    print(f"{GREEN}Model loaded successfully!\n{RESET}")
    
    qs = args.query
    qs=qs.replace("\\n", "\n")
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        if model.config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if model.config.mm_use_im_start_end:
            qs = image_token_se + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs # <image>\nWhat are the things I should be cautious about when I visit here?
    
    conv_mode=args.conv_mode
    print(f"{GREEN}conv_mode: {conv_mode}\n{RESET}")

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    image_files = image_parser(args)
    images = load_images(image_files,packing)
    image_sizes = [x.size for x in images] if not packing else None
    images_tensor, grid_thw = process_images(
        images,
        image_processor,
        model.config,
        packing,
    )
    images_tensor=images_tensor.to(model.device, dtype=torch.float16)
    grid_thw=grid_thw.to(model.device) if grid_thw is not None else None
    
    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .cuda()
    )

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images_tensor,
            image_sizes=image_sizes,
            # do_sample=True if args.temperature > 0 else False,#true
            do_sample=False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
            packing=packing,
            grid_thw=grid_thw,
        )
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    # 打印带颜色的输出
    print(f"{GREEN}The output is :{RESET}")
    print(f"{BLUE}{outputs}{RESET}")


if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "1" 

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/data/niujunbo/model/ckpts/llava-next-Qwen2-7B-Instruct-qwenvit-4_4096_visual_token-ft-790k-8192_2")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-file", type=str, default="/home/mineru/Document/niujunbo/github/NativeRes-LLaVA/demo/demo2.jpg")
    parser.add_argument("--query", type=str, default="Describe the image in detail." )
    parser.add_argument("--conv-mode", type=str, default="qwen_1_5")
    parser.add_argument("--sep", type=str, default=",")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--min_image_tokens", type=int, default=2)
    parser.add_argument("--max_image_tokens", type=int, default=4096)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    args = parser.parse_args()

    eval_model(args)
