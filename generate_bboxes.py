from datasets import load_from_disk, load_dataset
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
import json
from PIL import Image
import argparse
from tqdm import tqdm
import os


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Specify GPU, model path, and output file."
    )
    parser.add_argument("--gpu", type=int, help="CUDA GPU number to use", default=0)
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to the model",
        default="HuggingFaceTB/SmolVLM-256M-Instruct",
    )
    parser.add_argument(
        "--longest_edge",
        type=int,
        help="Longest image side size",
        default=512,
    )
    parser.add_argument(
        "--base_model_path",
        type=str,
        help="Path to the base model",
        default="HuggingFaceTB/SmolVLM-256M-Instruct",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to the final response",
        default="predictions/",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="If you want to show cropped out images.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        help="Maximum number of samples to process (None for all)",
        default=None,
    )
    return parser.parse_args()


# add more to generate for other datasets
DATASETS = {
    "textvqa": "lmms-lab/textvqa",
}
args = parse_arguments()

processor = AutoProcessor.from_pretrained(
    args.base_model_path, size={"longest_edge": args.longest_edge}
)
model = AutoModelForVision2Seq.from_pretrained(
    args.model_path,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
).to(f"cuda:{args.gpu}")
model.eval()

for k in DATASETS:
    ds = load_dataset(DATASETS[k], split="validation", cache_dir="datasets")

    predictions = []
    i = 0
    for e in tqdm(ds):
        if args.max_samples and i >= args.max_samples:
            break
        q = e["question"]
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {
                        "type": "text",
                        "text": f"{q.capitalize()}\nOutline the region in the image that would help answer this question.",
                    },
                ],
            },
        ]

        qid = e["question_id"]

        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(
            text=prompt,
            images=[
                (
                    e["image"].convert("RGB")
                    if hasattr(e["image"], "mode") and e["image"].mode != "RGB"
                    else e["image"]
                )
            ],
            return_tensors="pt",
        )
        inputs = inputs.to(f"cuda:{args.gpu}")

        # Generate outputs
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=500)

        generated_texts = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )
        response = generated_texts[0].split("Assistant: ")[-1]

        predictions.append({"question_id": qid, "answer": response})
        if args.verbose:
            print(f"{i} - {e['question']} - {e['answers']}")
            print(f"Response: {response}")
            e["image"].save(f"full_{i}.png")
            height, width = e["image"].height, e["image"].width

            coordinates = (
                response.replace("[", "").replace("].", "").replace("]", "").split(",")
            )

            x1, y1, x2, y2 = (
                float(coordinates[0]) * width / 100,
                float(coordinates[1]) * height / 100,
                float(coordinates[2]) * width / 100,
                float(coordinates[3]) * height / 100,
            )

            if [x1, y1, x2, y2] == [0, 0, 0, 0]:
                continue

            e["image"].crop([x1, y1, x2, y2]).save(f"crop_{i}.png")
        i += 1

    with open(os.path.join(args.output, f"{k}_bbox.json"), "w") as f:
        json.dump(predictions, f)
