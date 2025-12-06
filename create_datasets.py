from datasets import load_dataset
import torch
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
)
from qwen_vl_utils import process_vision_info
import json
import argparse
from tqdm import tqdm


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Generate bounding boxes using Qwen and expand them based on percentiles."
    )
    parser.add_argument("--gpu", type=int, help="CUDA GPU number to use", default=0)
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to the model",
        default="Qwen/Qwen2.5-VL-7B-Instruct",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        help="Dataset path (HuggingFace dataset name or local path)",
        default="lmms-lab/textvqa",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="Output dataset path",
        default="datasets/cropvlm_dataset",
    )
    parser.add_argument(
        "--split",
        type=str,
        help="Dataset split to use",
        default="train",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        help="Maximum number of samples to process (None for all)",
        default=None,
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="If you want to print the crops being generated.",
    )
    return parser.parse_args()


def expand_bbox(bbox, width, height, area_multiplier):
    """
    Expands a bounding box by a factor while maintaining its center and ensuring it stays within image bounds.

    Args:
        bbox: List [x1, y1, x2, y2] representing the bounding box
        width: Image width
        height: Image height
        area_multiplier: Factor by which to increase the area

    Returns:
        List [x1, y1, x2, y2] representing the expanded bounding box
    """
    # Current dimensions
    curr_width = bbox[2] - bbox[0]
    curr_height = bbox[3] - bbox[1]
    curr_area = curr_width * curr_height

    # Calculate new dimensions maintaining aspect ratio
    scale = (area_multiplier) ** 0.5
    new_width = curr_width * scale
    new_height = curr_height * scale

    # Calculate center point
    center_x = (bbox[0] + bbox[2]) / 2
    center_y = (bbox[1] + bbox[3]) / 2

    # Calculate new coordinates maintaining center
    x1 = center_x - new_width / 2
    y1 = center_y - new_height / 2
    x2 = center_x + new_width / 2
    y2 = center_y + new_height / 2

    # Adjust if box goes outside image bounds
    if x1 < 0:
        x2 -= x1  # Move right by the amount we're out of bounds
        x1 = 0
    if y1 < 0:
        y2 -= y1  # Move down by the amount we're out of bounds
        y1 = 0
    if x2 > width:
        x1 -= x2 - width  # Move left by the amount we're out of bounds
        x2 = width
    if y2 > height:
        y1 -= y2 - height  # Move up by the amount we're out of bounds
        y2 = height

    # Final bounds check (in case the box is larger than the image)
    x1 = max(0, min(x1, width))
    y1 = max(0, min(y1, height))
    x2 = max(0, min(x2, width))
    y2 = max(0, min(y2, height))

    return [
        int(x1 / width * 100),
        int(y1 / height * 100),
        int(x2 / width * 100),
        int(y2 / height * 100),
    ]


def get_qwenvl_response(model, processor, image, instruction, args):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": instruction},
            ],
        },
    ]
    # Prepare inputs using Qwen's specific processing
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(f"cuda:{args.gpu}")

    # Generate outputs
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

    response = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    return response


def parse_response(s):
    start = 0
    end = 0
    for i, c in enumerate(s):
        if c == "[":
            start = i
        elif c == "]":
            end = i
            break

    return s[start : end + 1]


def extract_bbox_from_response(response_text):
    response_json = parse_response(response_text)
    return json.loads(response_json)


def main():
    args = parse_arguments()

    print(f"Loading dataset from {args.dataset_path}...")
    ds = load_dataset(args.dataset_path, split=args.split)

    if args.max_samples:
        ds = ds.select(range(min(args.max_samples, len(ds))))

    print(f"Dataset loaded: {len(ds)} samples")

    # Split dataset into two equal parts
    total_samples = len(ds)
    split_point = total_samples // 2
    ds_sft = ds.select(range(split_point))
    ds_grpo = ds.select(range(split_point, total_samples))

    print(f"\nSplit dataset into two equal parts:")
    print(f"  SFT dataset: {len(ds_sft)} samples")
    print(f"  GRPO dataset: {len(ds_grpo)} samples")

    print(f"\nLoading model from {args.model_path}...")
    processor = AutoProcessor.from_pretrained(args.model_path)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype="auto",
        device_map=f"cuda:{args.gpu}",
        attn_implementation="flash_attention_2",
    )
    model.eval()
    print("Model loaded successfully")

    # Process SFT dataset with Qwen bbox generation (exactly as before)
    print("\n" + "=" * 60)
    print("Processing SFT dataset with Qwen bbox generation...")
    print("=" * 60)

    print("\nStep 1: Generating bounding boxes using Qwen...")
    generated_bboxes = []
    for i, e in tqdm(enumerate(ds_sft), total=len(ds_sft), desc="Generating bboxes"):
        try:
            question = e["question"]
            instruction = f"Outline the region in the image that would help answer the following question: {question}\nOutput the coordinates in JSON format with a 'bbox_2d' field containing [x1, y1, x2, y2]."

            with torch.no_grad():
                response = get_qwenvl_response(
                    model,
                    processor,
                    e["image"],
                    instruction,
                    args,
                )

            bbox = extract_bbox_from_response(response)
            if args.verbose:
                print(
                    f"{i} - Bbox: {bbox}, Question: {question}, GT Answer: {e['answers']}"
                )
                e["image"].crop([bbox[0], bbox[1], bbox[2], bbox[3]]).save(
                    f"crop_{i}.png"
                )
            generated_bboxes.append(bbox)
        except Exception as exc:
            print(f"Error processing sample {i}: {exc}")
            generated_bboxes.append([0, 0, e["image"].width, e["image"].height])

    print("\nStep 2: Calculating percentiles of relative bounding box areas...")
    relative = []
    for i, e in tqdm(
        enumerate(ds_sft), total=len(ds_sft), desc="Calculating percentiles"
    ):
        bbox = generated_bboxes[i]
        if bbox[0] == -1:
            continue
        image = e["image"]
        width, height = image.width, image.height
        rel_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) / (width * height)
        relative.append(rel_area)

    if len(relative) == 0:
        print("ERROR: No valid bounding boxes found. Cannot calculate percentiles.")
        return

    relative.sort()
    idx_20 = len(relative) // 5
    idx_40 = (len(relative) * 2) // 5
    idx_60 = (len(relative) * 3) // 5
    idx_80 = (len(relative) * 4) // 5

    value_20 = relative[idx_20]
    value_40 = relative[idx_40]
    value_60 = relative[idx_60]
    value_80 = relative[idx_80]

    print(f"Value at 20th percentile: {value_20:.4f}")
    print(f"Value at 40th percentile: {value_40:.4f}")
    print(f"Value at 60th percentile: {value_60:.4f}")
    print(f"Value at 80th percentile: {value_80:.4f}")

    print("\nStep 3: Expanding bounding boxes based on percentiles...")
    larger_bbox = []
    percentiles = []
    invalid_expanded_count = 0

    for i, e in tqdm(enumerate(ds_sft), total=len(ds_sft), desc="Expanding bboxes"):
        image = e["image"]
        width, height = image.width, image.height
        bbox = generated_bboxes[i]

        if bbox[0] != -1:
            rel = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) / (width * height)

            if rel < value_20:
                larger_bbox.append(expand_bbox(bbox, width, height, 45))
                s = "20"
            elif rel < value_40:
                larger_bbox.append(expand_bbox(bbox, width, height, 10))
                s = "40"
            elif rel < value_60:
                larger_bbox.append(expand_bbox(bbox, width, height, 4))
                s = "60"
            elif rel < value_80:
                larger_bbox.append(expand_bbox(bbox, width, height, 2))
                s = "80"
            else:
                larger_bbox.append(expand_bbox(bbox, width, height, 1))
                s = "100"

            percentiles.append(s)
        else:
            larger_bbox.append(expand_bbox(bbox, width, height, 1))
            percentiles.append("Error")
            invalid_expanded_count += 1

    print("\nStep 4: Creating SFT dataset...")
    ds_sft = ds_sft.add_column("bbox", larger_bbox)

    # GRPO dataset: no bboxes, just keep as-is
    print("\n" + "=" * 60)
    print("GRPO dataset: keeping original dataset without bboxes...")
    print("=" * 60)

    # Save both datasets
    output_path_sft = args.output_path + "_sft"
    output_path_grpo = args.output_path + "_grpo"

    print(f"\nSaving SFT dataset to {output_path_sft}...")
    ds_sft.save_to_disk(output_path_sft)

    print(f"\nSaving GRPO dataset to {output_path_grpo}...")
    ds_grpo.save_to_disk(output_path_grpo)

    print(f"\n" + "=" * 60)
    print(f"Datasets saved successfully!")
    print(f"  SFT dataset: {len(ds_sft)} samples -> {output_path_sft}")
    print(f"  GRPO dataset: {len(ds_grpo)} samples -> {output_path_grpo}")
    print(f"=" * 60)


if __name__ == "__main__":
    main()
