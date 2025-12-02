#!/bin/bash

export CUDA_DEVICE_ORDER=PCI_BUS_ID


CUDA_VISIBLE_DEVICES=$1 python sft.py \
  --base_model "HuggingFaceTB/SmolVLM-256M-Instruct" \
  --processor "HuggingFaceTB/SmolVLM-256M-Instruct" \
  --dataset_path "/tmp/cropvlm_dataset_sft" \
  --output_dir "models/sft_512" \
  --data_type "roi" \
  --lora_r 128 \
  --lora_alpha 256 \
  --lora_dropout 0.05 \
  --use_lora True \
  --longest_edge 512 \
  --remove_unused_columns false \
  --lr_scheduler_type "cosine" \
  --warmup_ratio 0.03 \
  --per_device_train_batch_size 2 \
  --learning_rate 5e-5 \
  --num_train_epochs 1 \
  --gradient_accumulation_steps 8 \
  --dataloader_num_workers 16 \
  --logging_steps 1 \
  --save_strategy "steps" \
  --save_steps 100 \
  --gpu 0 \

