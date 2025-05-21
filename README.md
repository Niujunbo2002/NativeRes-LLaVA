# NativeRes-LLaVA
Official code repo for our work [**Native Visual Understanding: Resolving Resolution Dilemmas in Vision-Language Models**](https://github.com/Niujunbo2002/NativeRes-LLaVA#)!

<p align="center">
  <a href="https://arxiv.org/abs/2501.05510" style="margin-right: 10px;"> 
    <img src="https://img.shields.io/badge/arXiv-2501.05510-b31b1b.svg?logo=arXiv">
  </a>
  <a href="https://huggingface.co/collections/Niujunbo2002/nativeres-llava-682d6f2f94ed89a9b2b71cb1" style="margin-right: 10px;"> 
    <img src="https://img.shields.io/badge/🤗%20Hugging%20Face-Model-ffd21e">
  </a>
</p>

## 📰 News
- [2025/1/6] 🔥🔥🔥 We released the paper on [arXiv](https://github.com/Niujunbo2002/NativeRes-LLaVA#)!

## 📌 ToDo Lists
- [x] Release Inference Code
- [x] Release NativeRes-LLaVA 1B && 2B && 7B Checkpoints
- [x] Release NativeRes-ViT (qwen2-vl-665m-patch14-nativeres)
- [ ] Release Training Code (The code is being organized.)
- [ ] Support SigLIP 2 with native resolution
- [ ] Support RL for post-train
- [ ] Support Multimodal Reasoning like O3

## Install

This is a repo enabling you train a LLaVA using images with native resolution.

1. Clone this repository and navigate to LLaVA folder

```bash
git clone https://github.com/Niujunbo2002/NativeRes-LLaVA.git
cd NativeRes-LLaVA
```

2. Install Package

```Shell
conda create -n nativeres python=3.10 -y
conda activate nativeres
pip install --upgrade pip  # enable PEP 660 support
pip install -r requirements.txt
pip install transformers==4.50.3
```

Install the required environment in `requirements.txt`. The Transforms version should be able to support at least `Qwen2-VL model`.

## Quick Start
First download the checkpoints at the folder. 
https://huggingface.co/collections/Niujunbo2002/nativeres-llava-682d6f2f94ed89a9b2b71cb1

### Inference

For Inference, we have a simple example, just run:

```
python ./infer_demo.py
```

## Train
Please note that the following is merely our reference to the official LLaVA training strategy. You are free to choose any training strategy you believe to be correct and efficient based on our codebase.

### Stage1: Pretrain

If you want to run using siglip ViT, which not support NativeRes, you can run:

```
bash scripts/train/pretrain_siglip.sh
```

Otherwise you can run in NativeRes mode which utilize Qwen2-VL ViT to support any resolution:

```
bash scripts/train/pretrain_qwenvit.sh
```

### Stage2: Finetune

For finetuning using siglip, just run

```
bash scripts/train/direct_finetune_siglip_a4_v1.5.sh
```

Otherwise you can run in NativeRes mode by:(using the LLaVA1.5 Fintuning Dataset now, you can change it anyway.)

```
bash scripts/train/direct_finetune_qwen_a4_v1.5_4_2048.sh
```

### Notes

1. Still not support zero3 in NativeRes mode now.
2. Update `sys.path.append("/mnt/petrelfs/niujunbo/zhengyuanhong/NativeResLLaVA")` to your personal path.
3. Still not support `video` now.



## Contact
Junbo Niu: 21376334@buaa.edu.cn


## Acknowledgements
This codebase is built upon [LLaVA](https://github.com/haotian-liu/LLaVA) and leverages several open-source libraries. We extend our gratitude to the contributors and maintainers of these projects.


## Citation
If you find our work helpful for your research, please consider giving a star ⭐ and citation 📝.
```bibtex

```