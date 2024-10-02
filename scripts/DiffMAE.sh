#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 main_pretrain.py \
--batch_size 1024 \
--model mae_vit_base_patch16 \
--blr 1.5e-4 \
--weight_decay 0.05 \
--data_path "/data/datasets/ImageNet" \
--num_workers 20 \
--enable_flash_attention2 \
--multi_epochs_dataloader \
--output_dir output/imagenet-diffmae-vitb-pretrain-wfm-mr0.75-dd12-ep1600 \
--cross_mae \
--weight_fm \
--decoder_depth 12 \
--mask_ratio 0.75 \
--epochs 1600 \
--warmup_epochs 40 \