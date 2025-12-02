from utils import *
import torch
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoProcessor,
    AutoModelForVision2Seq,
    Trainer,
)
from datasets import load_from_disk, load_dataset
from functools import partial


model_args, data_args, lora_args, training_args = parse_args()

all_args = {
    "model_args": vars(model_args),
    "data_args": vars(data_args),
    "lora_args": vars(lora_args),
    "training_args": vars(training_args),
}


processor = AutoProcessor.from_pretrained(
    model_args.processor, size={"longest_edge": model_args.longest_edge}
)

lora_config = LoraConfig(
    r=lora_args.lora_r,
    lora_alpha=lora_args.lora_alpha,
    lora_dropout=lora_args.lora_dropout,
    target_modules=[
        "down_proj",
        "o_proj",
        "k_proj",
        "q_proj",
        "gate_proj",
        "up_proj",
        "v_proj",
        "fc1",
        "fc2",
        "proj",
    ],
    # modules_to_save=["lm_head"],
)

model = AutoModelForVision2Seq.from_pretrained(
    model_args.base_model,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
).to(f"cuda:{training_args.gpu}")

if model_args.use_lora:
    model = get_peft_model(model, lora_config, adapter_name=data_args.data_type)
    model.print_trainable_parameters()

try:
    train_ds = load_dataset(data_args.dataset_path, split="train", cache_dir="/tmp")
except:
    train_ds = load_from_disk(data_args.dataset_path)

image_token_id = processor.tokenizer.additional_special_tokens_ids[
    processor.tokenizer.additional_special_tokens.index("<image>")
]

collate_fn = partial(
    collate_fn,
    processor=processor,
    image_token_id=image_token_id,
    data_type=data_args.data_type,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collate_fn,
    train_dataset=train_ds,
)

trainer.train()
