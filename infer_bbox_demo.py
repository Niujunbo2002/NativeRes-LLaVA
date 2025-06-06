import sys
sys.path.append("/mnt/petrelfs/niujunbo/niujunbo_dev/NativeRes-LLaVA")
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

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_PLACEHOLDER
# # Define constants
# IGNORE_INDEX = -100
# IMAGE_TOKEN_INDEX = -200
# DEFAULT_IMAGE_TOKEN = "<image>"
# DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
# DEFAULT_IM_START_TOKEN = "<im_start>"
# DEFAULT_IM_END_TOKEN = "<im_end>"
# IMAGE_PLACEHOLDER = "<image-placeholder>"
# ANSI escape sequences for colored output
RED = '\033[91m'
GREEN = '\033[92m'
BLUE = '\033[94m'
RESET = '\033[0m'


import re

def parse_bbox_prompt(prompt_string):
    """
    Parses a structured string to extract bounding box coordinates, labels, and content.

    Args:
        prompt_string (str): The string containing one or more bbox definitions.

    Returns:
        list: A list of dictionaries, where each dictionary contains:
              'box_coords': (Y1, X1, Y2, X2) tuple of integers (top, left, bottom, right)
              'label': str (e.g., 'text', 'table', 'title')
              'content': str (the markdown content associated with the box)
    """
    bboxes_data = []
    current_pos = 0

    # Define the tags
    start_box_tag = "<|box_start|>"
    end_box_tag = "<|box_end|>"
    start_ref_tag = "<|ref_start|>"
    end_ref_tag = "<|ref_end|>"
    start_md_tag = "<|md_start|>"
    end_md_tag = "<|md_end|>"

    while True:
        # Find the start of the next box entry
        start_coords_idx = prompt_string.find(start_box_tag, current_pos)
        if start_coords_idx == -1:
            break  # No more box entries found

        # Extract coordinates string
        start_coords_val_idx = start_coords_idx + len(start_box_tag)
        end_coords_idx = prompt_string.find(end_box_tag, start_coords_val_idx)
        if end_coords_idx == -1:
            # Malformed string, missing end_box_tag
            break
        coords_str = prompt_string[start_coords_val_idx:end_coords_idx].strip()
        
        try:
            # Coordinates are expected as Y1 X1 Y2 X2
            x1, y1, x2, y2 = map(int, coords_str.split())
            box_coords = (x1, y1, x2, y2) # (top, left, bottom, right)
        except ValueError:
            # Malformed coordinates string
            current_pos = end_coords_idx + len(end_box_tag) # Move past this malformed entry
            print(f"Warning: Skipping malformed coordinate string: '{coords_str}'")
            continue

        # Extract label
        start_ref_idx = prompt_string.find(start_ref_tag, end_coords_idx + len(end_box_tag))
        if start_ref_idx == -1:
            break # Malformed string
        start_ref_val_idx = start_ref_idx + len(start_ref_tag)
        end_ref_idx = prompt_string.find(end_ref_tag, start_ref_val_idx)
        if end_ref_idx == -1:
            break # Malformed string
        label = prompt_string[start_ref_val_idx:end_ref_idx].strip()

        # Extract markdown content
        start_md_idx = prompt_string.find(start_md_tag, end_ref_idx + len(end_ref_tag))
        if start_md_idx == -1:
            break # Malformed string
        start_md_val_idx = start_md_idx + len(start_md_tag)
        end_md_idx = prompt_string.find(end_md_tag, start_md_val_idx)
        if end_md_idx == -1:
            break # Malformed string
        content = prompt_string[start_md_val_idx:end_md_idx].strip()
        
        bboxes_data.append({
            "box_coords": box_coords,
            "label": label,
            "content": content
        })

        # Move current_pos to the end of the current parsed block
        current_pos = end_md_idx + len(end_md_tag)


    return bboxes_data

def draw_bbox(img_path, prompt):
    """
    Parses the prompt to extract bounding box information.
    This function extracts the data and can optionally include drawing logic
    if an image processing library and image are available.

    Args:
        img_path (str): Path to the image file. (Required for actual drawing)
        prompt (str): The structured string containing bounding box data from the model output.
        input_image_size: (height, width)
    Returns:
        list: A list of dictionaries, where each dictionary contains:
              'box_coords': (Y1, X1, Y2, X2) tuple of integers (top, left, bottom, right)
              'label': string (e.g., 'text', 'table', 'title')
              'content': string (the markdown content associated with the box)
    """
    parsed_data = parse_bbox_prompt(prompt)
    
    # --- Placeholder for actual drawing logic ---
    # To draw on an image, you would typically use a library like Pillow (PIL).
    # Example (requires Pillow: pip install Pillow):
    #
    from PIL import Image, ImageDraw, UnidentifiedImageError
    
    if not parsed_data:
        print("No bounding box data was parsed.")
        return parsed_data
    
    try:
        image = Image.open(img_path)
        draw = ImageDraw.Draw(image)
        for item in parsed_data:
            
            x1, y1, x2, y2 = item['box_coords'] # These are (top, left, bottom, right)
            # print(f"{RED} {x1} {y1} {x2} {y2} {RESET}")
            # # Pillow's rectangle uses (left, top, right, bottom)
            
            # print(f"{RED} {image.width} {image.height} {RESET}")
            
            # input_image_size
            x1 = x1 / 1000 * image.width
            y1 = y1 / 1000 * image.height
            x2 = x2 / 1000 * image.width
            y2 = y2 / 1000 * image.height
            
            pil_box = (int(x1), int(y1), int(x2), int(y2)) 


            label = item['label']
    
            outline_color = "yellow" # Default color
            if label == 'text':
                outline_color = "red"
            elif label == 'table':
                outline_color = "blue"
            elif label == 'title':
                outline_color = "green"
    
            draw.rectangle(pil_box, outline=outline_color, width=4)
            # Optionally draw the label text near the box
            # Adjust text_position as needed for visibility
            text_position = (x1, y1 - 12 if y1 > 10 else y1 + 2) 
            try: # Basic font if default is not available
                draw.text(text_position, label, fill=outline_color)
            except ImportError: # Fallback if no default font found on some systems
               # For a truly robust solution, specify a font file:
               # from PIL import ImageFont
               # font = ImageFont.truetype("arial.ttf", 10)
               # draw.text(text_position, label, fill=outline_color, font=font)
               pass # Or handle error appropriately
               
        image.save("./playground/output_with_bboxes.png")
        print(f"Bounding boxes drawn on {img_path} (if it was a valid image).")
    
    except FileNotFoundError:
        print(f"Error: Image file not found at '{img_path}'. Cannot draw bounding boxes.")
        print("Returning parsed data only.")
    except UnidentifiedImageError:
        print(f"Error: Cannot identify image file at '{img_path}'. It might be corrupted or not an image.")
        print("Returning parsed data only.")
    except Exception as e:
        print(f"An error occurred during image processing: {e}")
        print("Returning parsed data only.")
    
    # --- End of placeholder drawing logic ---

    return parsed_data

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

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path,
        None, 
        model_name,
        min_image_tokens=args.min_image_tokens,
        max_image_tokens=args.max_image_tokens,
    )

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
            # qs = DEFAULT_IMAGE_TOKEN + "\n" + qs # <image>\nWhat are the things I should be cautious about when I visit here?
            num_images = len(image_files)
            qs = (DEFAULT_IMAGE_TOKEN + "\n") * num_images + qs

    conv_mode=args.conv_mode
    print(f"{GREEN}conv_mode: {conv_mode}\n{RESET}")
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()


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

    print(f"{GREEN}The output is :{RESET}")
    print(f"{BLUE}{outputs}{RESET}")

    draw_bbox(args.image_file, outputs)


if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "1" 

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/mnt/petrelfs/niujunbo/NativeRes-LLaVA/playground/NativeRes-LLaVA-qwen2-0.5b-qwen2vit-Exp-4-7290")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-file", type=str, default="/mnt/petrelfs/niujunbo/NativeRes-LLaVA/demo/ocr2.png")
    parser.add_argument("--query", type=str, default="Document Parsing: " )
    parser.add_argument("--conv-mode", type=str, default="qwen_1_5")
    parser.add_argument("--sep", type=str, default=",")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--min_image_tokens", type=int, default=4)
    parser.add_argument("--max_image_tokens", type=int, default=3096)
    parser.add_argument("--max_new_tokens", type=int, default=14000)
    args = parser.parse_args()

    eval_model(args)
