# DiffMAE Implementation

This repository contains an implementation of the DiffMAE model as described in the paper:

**"DiffMAE: Diffusion Models for Masked Autoencoders"**
(arXiv: [2304.03283](https://arxiv.org/pdf/2304.03283))

## Overview

The implementation is based on the CrossMAE architecture, with modifications to incorporate diffusion processes.

## Instructions
Please install the dependencies in `requirements.txt`:
```sh
# Optionally create a conda environment
conda create -n diffmae python=3.10 -y
conda activate diffmae
# Install dependencies
pip install -r requirements.txt
```

### Pre-training DiffMAE
To pre-train ViT-Base, run the following on 4 GPUs:
```sh
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port 1234 main_pretrain.py --batch_size 1024 --model mae_vit_base_patch16 --norm_pix_loss --blr 1.5e-4 --weight_decay 0.05 --data_path ${IMAGENET_DIR} --num_workers 20 --enable_flash_attention2 --multi_epochs_dataloader --output_dir output/imagenet-diffmae-vitb-pretrain-wfm-mr0.75-dd12-ep1600 --cross_mae --weight_fm --decoder_depth 12 --mask_ratio 0.75 --epochs 1600 --warmup_epochs 40
```

To train ViT-Small or ViT-Large, set `--model mae_vit_small_patch16` or `--model mae_vit_large_patch16`. You can use `--accum_iter` to perform gradient accumulation if your hardware could not fit the batch size. [FlashAttention 2](https://github.com/Dao-AILab/flash-attention) should be installed with `pip install flash-attn --no-build-isolation`.

### Fine-tuning DiffMAE
To pre-train ViT-Base, run the following on 4 GPUs:
```sh
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port 1234 main_finetune.py --batch_size 256 --model vit_base_patch16 --finetune output/imagenet-diffmae-vitb-pretrain-wfm-mr0.75-dd12-ep1600/checkpoint-1600.pth --epoch 100 --blr 5e-4 --layer_decay 0.65 --weight_decay 0.05 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 --dist_eval --data_path ${IMAGENET_DIR} --output_dir output/imagenet-diffmae-vitb-finetune-wfm-mr0.75-dd12-ep1600 --enable_flash_attention2 --multi_epochs_dataloader
```

## Evaluation
Evaluate ViT-Base in a single GPU (`${IMAGENET_DIR}` is a directory containing `{train, val}` sets of ImageNet). `${FINETUNED_CHECKPOINT_PATH}` is the path to the fine-tuned checkpoint:
```sh
python main_finetune.py --eval --resume ${FINETUNED_CHECKPOINT_PATH} --model vit_base_patch16 --batch_size 16 --data_path ${IMAGENET_DIR}
```

### Pre-training MaskDiT 
To pre-train ViT-Base, run the following on 4 GPUs:
```sh
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 main_pretrain.py \
--batch_size 512 \
--model mae_vit_base_patch16 \
--blr 1.5e-4 \
--weight_decay 0.05 \
--data_path "/data/datasets/ImageNet" \
--num_workers 20 \
--multi_epochs_dataloader \
--output_dir output/imagenet-dit-vitb-pretrain-wfm-mr0.75-dd12-ep400 \
--mask_ratio 0.5 \
--epochs 400 \
--norm_pix_loss \
--warmup_epochs 20 \
```

### Fine-tuning MaskDiT
To pre-train ViT-Base, run the following on 4 GPUs:
(use time_step 12)
```sh
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port 65432 --nproc_per_node=4 main_finetune.py \
--batch_size 1024 \
--model vit_base_patch16 \
--num_workers 20 \
--finetune {checkpoint} \
--epoch 101 \
--blr 5e-4 \
--layer_decay 0.65 \
--weight_decay 0.05 \
--drop_path 0.1 \
--reprob 0.25 \
--mixup 0.8 \
--cutmix 1.0 \
--dist_eval \
--data_path /data/datasets/ImageNet \
--output_dir output/ditmae_finetune_400 \
--multi_epochs_dataloader
```

## Acknowledgements

We would like to acknowledge the authors of the CrossMAE paper for their foundational work, which has greatly informed this implementation.
