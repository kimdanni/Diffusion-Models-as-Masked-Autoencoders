#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 main_pretrain.py \
--batch_size 512 \
--model mae_vit_base_patch16 \
--blr 1.5e-4 \
--weight_decay 0.05 \
--data_path "/data/datasets/ImageNet" \
--num_workers 20 \
--multi_epochs_dataloader \
--output_dir output/imagenet-dit-vitb-pretrain-wfm-mr0.75-dd12-ep800 \
--mask_ratio 0.5 \
--epochs 400 \
--norm_pix_loss \
--warmup_epochs 20 \
--resume /data/daeun/DiffMAE/output/imagenet-dit-vitb-pretrain-wfm-mr0.75-dd12-ep800/checkpoint-200.pth \