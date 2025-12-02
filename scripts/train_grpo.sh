#!/bin/bash

export CUDA_DEVICE_ORDER=PCI_BUS_ID

CUDA_VISIBLE_DEVICES=$1 python grpo.py \
  --base_model "models/sft_512/checkpoint-100/roi" \
  --processor "HuggingFaceTB/SmolVLM-256M-Instruct" \
  --longest_edge 512 \
  --dataset_path "/tmp/cropvlm_dataset_grpo" \
  --output_dir "models/grpo_512" \
  --remove_unused_columns false \
  --lr_scheduler_type "cosine" \
  --warmup_ratio 0.03 \
  --per_device_train_batch_size 1 \
  --learning_rate 5e-6 \
  --max_grad_norm 0.1 \
  --num_train_epochs 1 \
  --gradient_accumulation_steps 16 \
  --logging_steps 1 \
  --save_strategy "steps" \
  --save_steps 150 \
  --gpu 0 \
  --num_generations 6 \
  --temperature 0.8 \
  --beta 0.01 \
  --max_prompt_length 32000 \
  --max_completion_length 32 \