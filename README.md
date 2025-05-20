# NativeRes-LLaVA
>[**Native Visual Understanding: Resolving Resolution Dilemmas in Vision-Language Models**]


## 📰 News
- [2025/1/6] 🔥🔥🔥 We released the paper on [arXiv](http://arxiv.org/abs/2501.03218)!

## 📌 ToDo Lists
- [x] Release Inference Code
- [x] Release Checkpoints
- [ ] Release Training Code (The code is being organized.)
- [ ] Support SigLIP 2 with native resolution
- [ ] Release NativeRes ViT
- [ ] Support RL for post-train
- [ ] Support Multimodal Reasoning like O3

## Install

This is a repo enabling you train a LLaVA using images with any resolution.

1. Clone this repository and navigate to LLaVA folder

```bash
git clone https://github.com/Yuanhong-Zheng/NativeResLLaVA
cd NativeResLLaVA
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