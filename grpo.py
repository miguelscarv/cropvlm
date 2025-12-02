from utils import *
import torch
from transformers import (
    AutoProcessor,
    AutoModelForVision2Seq,
)
from datasets import load_from_disk, load_dataset
from trainer import VLMGRPOTrainer
import os


model_args, data_args, lora_args, training_args = parse_args(is_sft=False)

all_args = {
    "model_args": vars(model_args),
    "data_args": vars(data_args),
    "lora_args": vars(lora_args),
    "training_args": vars(training_args),
}


processor = AutoProcessor.from_pretrained(
    model_args.processor, size={"longest_edge": model_args.longest_edge}
)

response_processor = AutoProcessor.from_pretrained(
    model_args.processor, size={"longest_edge": 512}
)


model = AutoModelForVision2Seq.from_pretrained(
    model_args.base_model,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
).to(f"cuda:{training_args.gpu}")

vqa_model = AutoModelForVision2Seq.from_pretrained(
    model_args.vqa_model,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
).to(f"cuda:{training_args.gpu}")

for n, p in model.named_parameters():
    if "lora" in n:
        p.requires_grad = True

print(
    "trainable parameters: ",
    sum(p.numel() for p in model.parameters() if p.requires_grad),
)

try:
    train_ds = load_dataset(data_args.dataset_path, split="train", cache_dir="/tmp")
except:
    train_ds = load_from_disk(data_args.dataset_path)

train_ds = train_ds.map(
    lambda x: {
        "first_prompt": [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {
                        "type": "text",
                        "text": f"{x['question'].capitalize()}\nOutline the region in the image that would help answer this question.",
                    },
                ],
            }
        ],
        "second_prompt": [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "image"},
                    {
                        "type": "text",
                        "text": f"{x['question'].capitalize()}\nGive a very brief answer.",
                    },
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": get_single_answer(x["answers"]).capitalize() + ".",
                    }
                ],
            },
        ],
    }
)

image_token_id = processor.tokenizer.additional_special_tokens_ids[
    processor.tokenizer.additional_special_tokens.index("<image>")
]


trainer = VLMGRPOTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    reward_funcs=[log_likelihood_rewards, valid_first_completion_rewards],
    processing_class=processor,
    vqa_model=vqa_model,
    response_processor=response_processor,
)

trainer.train()
