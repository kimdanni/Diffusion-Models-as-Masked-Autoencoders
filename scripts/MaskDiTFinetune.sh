#!/bin/bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port 65432 --nproc_per_node=4 main_finetune.py \
--batch_size 1024 \
--model vit_base_patch16 \
--num_workers 20 \
--finetune /data/daeun/DiffMAE/output/imagenet-dit-vitb-pretrain-wfm-mr0.75-dd12-ep800/checkpoint-200.pth \
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
--output_dir output/ditmae_finetune_380 \
--multi_epochs_dataloader