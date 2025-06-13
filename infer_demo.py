import sys
import argparse
import warnings
import re
from io import BytesIO

import torch
from PIL import Image
import requests
from transformers import logging

# Append local path
sys.path.append("/mnt/petrelfs/niujunbo/niujunbo_dev/github/NativeRes-LLaVA")

# Local imports
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)
from llava.constants import (
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IMAGE_PATCH_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)

# Suppress warnings
warnings.filterwarnings("ignore")
logging.set_verbosity_error()

# ANSI terminal colors
RED = '\033[91m'
GREEN = '\033[92m'
BLUE = '\033[94m'
RESET = '\033[0m'


def image_parser(args):
    return args.image_file.split(args.sep)


def load_image(image_file):
    """Load image from local path or URL"""
    if image_file.startswith("http"):
        response = requests.get(image_file)
        return Image.open(BytesIO(response.content)).convert("RGB")
    return Image.open(image_file).convert("RGB")


def load_images(image_files, packing):
    """Conditionally load and process images"""
    if packing:
        return image_files
    return [load_image(f) for f in image_files]


def eval_model(args):
    packing = 'qwen' in args.model_path.lower()

    disable_torch_init()
    model_name = get_model_name_from_path(args.model_path)

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path,
        None,
        model_name,
        min_image_tokens=args.min_image_tokens,
        max_image_tokens=args.max_image_tokens,
    )

    # Load image(s)
    image_files = image_parser(args)
    images = load_images(image_files, packing)
    image_sizes = [img.size for img in images] if not packing else None

    # Preprocess images
    images_tensor, grid_thw = process_images(
        images, image_processor, model.config, packing
    )
    images_tensor = images_tensor.to(model.device, dtype=torch.float16)
    grid_thw = grid_thw.to(model.device) if grid_thw is not None else None

    # Prepare prompt
    qs = args.query.replace("\\n", "\n")
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN

    if IMAGE_PLACEHOLDER in qs:
        qs = re.sub(
            IMAGE_PLACEHOLDER,
            image_token_se if model.config.mm_use_im_start_end else DEFAULT_IMAGE_TOKEN,
            qs
        )
    else:
        prefix = image_token_se if model.config.mm_use_im_start_end else DEFAULT_IMAGE_TOKEN
        qs = (prefix + "\n") * len(image_files) + qs

    # Build conversation
    print(f"{GREEN}conv_mode: {args.conv_mode}\n{RESET}")
    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    # Tokenize input
    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .cuda()
    )

    # Run model inference
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images_tensor,
            image_sizes=image_sizes,
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
    print(f"{GREEN}The output is:{RESET}")
    print(f"{BLUE}{outputs}{RESET}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/mnt/petrelfs/niujunbo/niujunbo_dev/ocr_ckpts/nativeres-llava-Qwen2-1.5B-Instruct-qwenvit_2_5-ft-v2")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-file", type=str, default="demo/paper2.jpg")
    parser.add_argument("--query", type=str, default="Describe the image in detail.")
    parser.add_argument("--conv-mode", type=str, default="qwen_1_5")
    parser.add_argument("--sep", type=str, default=",")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--min_image_tokens", type=int, default=4)
    parser.add_argument("--max_image_tokens", type=int, default=4096)
    parser.add_argument("--max_new_tokens", type=int, default=512)

    args = parser.parse_args()
    eval_model(args)
