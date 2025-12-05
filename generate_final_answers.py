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
        "--vqa_model_path",
        type=str,
        help="Path to the model",
        default="HuggingFaceTB/SmolVLM-256M-Instruct",
    )
    parser.add_argument(
        "--bbox",
        type=str,
        help="Path to the bounding boxes",
        default="predictions/textvqa_bbox.json",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to the final response",
        default="predictions/",
    )
    parser.add_argument(
        "--longest_edge",
        type=int,
        help="Longest image side size",
        default=512,
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="If you want to show cropped out images.",
    )
    return parser.parse_args()


# add more to generate for other datasets
DATASETS = {
    "textvqa": "lmms-lab/textvqa",
}
args = parse_arguments()

processor = AutoProcessor.from_pretrained(
    args.vqa_model_path, size={"longest_edge": args.longest_edge}
)

vqa_model = AutoModelForVision2Seq.from_pretrained(
    args.vqa_model_path,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
).to(f"cuda:{args.gpu}")
vqa_model.eval()

with open(args.bbox, "r") as f:
    bbox = json.load(f)

for k in DATASETS:
    ds = load_dataset(DATASETS[k], split="validation", cache_dir="/tmp")

    predictions = []
    i = 0
    for e, b in tqdm(zip(ds, bbox)):
        img = (
            e["image"].convert("RGB")
            if hasattr(e["image"], "mode") and e["image"].mode != "RGB"
            else e["image"]
        )
        qid = e["question_id"]
        height, width = img.height, img.width

        coordinates = (
            b["answer"].replace("[", "").replace("].", "").replace("]", "").split(",")
        )
        try:
            x1, y1, x2, y2 = (
                float(coordinates[0]) * width / 100,
                float(coordinates[1]) * height / 100,
                float(coordinates[2]) * width / 100,
                float(coordinates[3]) * height / 100,
            )
        except:
            x1, y1, x2, y2 = 0, 0, 99, 99
            print("Error in bbox")

        cropped_image = img.crop((x1, y1, x2, y2))

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "image"},
                    {
                        "type": "text",
                        "text": f"{e['question'].capitalize()}\nGive a very brief answer.",
                    },
                ],
            },
        ]

        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(
            text=prompt,
            images=[img, cropped_image],
            return_tensors="pt",
        )
        inputs = inputs.to(f"cuda:{args.gpu}")

        # Generate outputs
        with torch.no_grad():
            generated_ids = vqa_model.generate(**inputs, max_new_tokens=500)

        generated_texts = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )
        final_response = generated_texts[0].split("Assistant: ")[-1]
        if final_response[-1] == ".":
            final_response = final_response[:-1]

        predictions.append({"question_id": qid, "answer": final_response})

        if args.verbose:
            img.save(f"full_{i}.png")
            cropped_image.save(f"cropped_{i}.png")
            print(i, e["question"], e["answers"], final_response)

        i += 1

    with open(
        os.path.join(
            args.output, args.bbox.split("/")[-1].replace("_bbox.json", "_answers.json")
        ),
        "w",
    ) as f:
        json.dump(predictions, f)
