from dataclasses import dataclass, field
from typing import Optional
from transformers import HfArgumentParser, TrainingArguments
from trl import GRPOConfig
from anls import anls_score
import torch
import difflib
import itertools
import random
from collections import defaultdict
import time
from typing import Optional
import subprocess
import re


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune
    """

    base_model: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co"
        },
        default="HuggingFaceTB/SmolVLM-256M-Instruct",
    )

    vqa_model: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co"
        },
        default="HuggingFaceTB/SmolVLM-256M-Instruct",
    )

    processor: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co"
        },
        default="HuggingFaceTB/SmolVLM-256M-Instruct",
    )

    use_lora: bool = field(default=True, metadata={"help": "Whether to use LoRA"})

    longest_edge: int = field(
        default=2048, metadata={"help": "Longest image side size"}
    )


@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training
    """

    dataset_path: str = field(
        metadata={"help": "Path to the training dataset"}, default="textvqa_train"
    )

    data_type: str = field(
        metadata={"help": "Type of data to use (vqa, roi, etc...)"}, default="vqa"
    )


@dataclass
class LoraArguments:
    """
    Arguments pertaining to LoRA configuration
    """

    lora_r: int = field(
        default=8, metadata={"help": "Rank of the LoRA update matrices"}
    )
    lora_alpha: int = field(
        default=16, metadata={"help": "LoRA alpha parameter - scaling factor"}
    )
    lora_dropout: float = field(
        default=0.05, metadata={"help": "Dropout probability for LoRA layers"}
    )


@dataclass
class TrainingArguments(TrainingArguments):
    """
    Arguments pertaining to training configuration
    """

    bf16: bool = field(
        default=True,
        metadata={"help": "Whether to use bf16 16-bit (mixed) precision training"},
    )
    remove_unused_columns: bool = field(
        default=False,
        metadata={"help": "Remove unused columns from the dataset"},
    )
    lr_scheduler_type: str = field(
        default="cosine",
        metadata={
            "help": "The scheduler type to use (linear, cosine, cosine_with_restarts, etc)"
        },
    )
    warmup_ratio: float = field(
        default=0.03,
        metadata={"help": "Linear warmup over warmup_ratio fraction of total steps"},
    )
    per_device_train_batch_size: int = field(
        default=2, metadata={"help": "Batch size per GPU/TPU core/CPU for training"}
    )
    learning_rate: float = field(
        default=1e-4, metadata={"help": "The initial learning rate for AdamW"}
    )
    num_train_epochs: int = field(
        default=1, metadata={"help": "Total number of training epochs to perform"}
    )
    gradient_accumulation_steps: int = field(
        default=8,
        metadata={
            "help": "Number of updates steps to accumulate before performing a backward/update pass"
        },
    )
    logging_steps: int = field(
        default=1, metadata={"help": "Log every X updates steps"}
    )
    save_strategy: str = field(default="epoch", metadata={"help": "Save strategy"})
    save_steps: int = field(
        default=100, metadata={"help": "Save checkpoint every X updates steps"}
    )

    seed: int = field(default=42, metadata={"help": "Random seed for initialization"})
    gpu: int = field(default=0, metadata={"help": "GPU to use"})
    is_sft: bool = field(default=True, metadata={"help": "Is this SFT?"})
    is_dr_grpo: bool = field(
        default=True,
        metadata={
            "help": "Whether or not to remove the reward scaling and length scaling."
        },
    )
    report_to: str = field(
        default="none",
        metadata={"help": "The list of integrations to report the results and logs to."},
    )


@dataclass
class GRPOArguments(TrainingArguments, GRPOConfig):
    """
    Arguments pertaining to GRPO configuration
    """

    lambda_ll: float = field(
        default=1.0,
        metadata={"help": "Lambda for the LL loss"},
    )


def parse_args(is_sft=True):
    parser = HfArgumentParser(
        (
            ModelArguments,
            DataArguments,
            LoraArguments,
            TrainingArguments if is_sft else GRPOArguments,
        )
    )
    model_args, data_args, lora_args, training_args = (
        parser.parse_args_into_dataclasses()
    )
    return model_args, data_args, lora_args, training_args


def string_similar(s1, s2):
    return difflib.SequenceMatcher(None, s1, s2).quick_ratio()


def processPunctuation(inText):
    outText = inText
    for p in punct:
        if (p + " " in inText or " " + p in inText) or (
            re.search(commaStrip, inText) != None
        ):
            outText = outText.replace(p, "")
        else:
            outText = outText.replace(p, " ")
    outText = periodStrip.sub("", outText, re.UNICODE)
    return outText


def processDigitArticle(inText):
    outText = []
    tempText = inText.lower().split()
    for word in tempText:
        word = manualMap.setdefault(word, word)
        if word not in articles:
            outText.append(word)
        else:
            pass
    for wordId, word in enumerate(outText):
        if word in contractions:
            outText[wordId] = contractions[word]
    outText = " ".join(outText)
    return outText


def process_text(text):
    text = processPunctuation(text)
    text = processDigitArticle(text)
    return text


def vqa_accuracy(prediction, true_answers, model="smolvlm"):
    if model == "smolvlm":
        temp = []
        for comb in itertools.combinations(true_answers, 9):
            temp.append(min(1, comb.count(prediction.lower()) / 3))
        return sum(temp) / len(temp)
    else:
        pred = process_text(prediction)
        gts = [process_text(gt) for gt in true_answers]
        same_num = sum([1 if pred == gt else 0 for gt in gts])
        return min(0.3 * same_num, 1)


def accuracy_rewards(prompts, completions, true_answers, **kwargs):
    responses = [completion["second"] for completion in completions]
    for i in range(len(responses)):
        responses[i] = responses[i].strip()
        if responses[i]:
            if responses[i][-1] == ".":
                responses[i] = responses[i][:-1]

    scores = []
    for r, t in zip(responses, true_answers):
        if len(true_answers[0]) == 10:  # textvqa
            scores.append(vqa_accuracy(r, t))
        else:
            scores.append(anls_score(prediction=r, gold_labels=t, threshold=0.5))

    return scores

def log_likelihood_rewards(prompts, completions, **kwargs):
    return [completion["second"] for completion in completions]


def valid_first_completion_rewards(prompts, completions, **kwargs):
    responses = [completion["first"] for completion in completions]
    scores = []
    for bbox in responses:
        try:
            coordinates = bbox.strip().replace("[", "").replace("].", "").split(",")
            if len(coordinates) != 4:
                raise ValueError("Invalid number of coordinates")

            x1_pct = float(coordinates[0])
            y1_pct = float(coordinates[1])
            x2_pct = float(coordinates[2])
            y2_pct = float(coordinates[3])

            # Validate coordinates are between 0-100 and x2>x1, y2>y1
            if not (
                0 <= x1_pct <= 100
                and 0 <= y1_pct <= 100
                and 0 <= x2_pct <= 100
                and 0 <= y2_pct <= 100
                and x2_pct > x1_pct
                and y2_pct > y1_pct
            ):
                raise ValueError("Invalid coordinate values")

            score = 1
        except:
            score = 0

        if bbox.strip() == "Not needed.":
            score = 0.25

        scores.append(score)

    return scores


def collate_fn(examples, processor=None, image_token_id=None, data_type=None):
    texts = []
    images = []
    for example in examples:
        image = example["image"]
        if image.mode != "RGB":
            image = image.convert("RGB")
        question = example["question"]

        if data_type == "vqa":
            # Get majority vote from answers
            answers = example["answers"]
            answer_counts = {}
            for ans in answers:
                ans = ans.lower().strip()  # Normalize answers
                answer_counts[ans] = answer_counts.get(ans, 0) + 1
            answer = max(answer_counts.items(), key=lambda x: x[1])[0]
        elif data_type == "roi":
            bbox = example["bbox"]
            width, height = image.width, image.height
            bbox[0] = int(bbox[0] / width * 100)
            bbox[1] = int(bbox[1] / height * 100)
            bbox[2] = int(bbox[2] / width * 100) + 1
            bbox[3] = int(bbox[3] / height * 100) + 1
            answer = f"[{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]"
        else:
            raise ValueError(f"Have not implemented this data_type yet: {data_type}")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {
                        "type": "text",
                        "text": (
                            f"{question.capitalize()}\nOutline the region in the image that would help answer this question."
                        ),
                    },
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": answer.capitalize() + "."}],
            },
        ]

        
        text = processor.apply_chat_template(messages, add_generation_prompt=False)
        texts.append(text.strip())
        images.append([image])
        assert len(images[0]) == 1, "Used wrong data type"

    batch = processor(text=texts, images=images, return_tensors="pt", padding=True)
    labels = batch["input_ids"].clone()

    # Set all labels to -100 initially
    labels[:, :] = -100

    # Assistant token sequence
    assistant_tokens = torch.tensor([198, 9519, 9531, 42])
    assistant_start_token = 198

    for i in range(labels.shape[0]):
        pos = 0
        while pos < len(labels[i]) - len(assistant_tokens):
            # Find the assistant sequence
            if torch.all(
                batch["input_ids"][i, pos : pos + len(assistant_tokens)]
                == assistant_tokens
            ):
                # Mark the assistant sequence itself as -100
                labels[i, pos : pos + len(assistant_tokens)] = -100

                # Start position after the sequence
                start_pos = pos + len(assistant_tokens)

                # Find the next assistant start token or end of sequence
                next_pos = start_pos
                while next_pos < len(labels[i]):
                    if batch["input_ids"][i, next_pos] == assistant_start_token:
                        next_pos += 1
                        break
                    next_pos += 1

                # Keep the completion (everything between the sequence and next start token)
                labels[i, start_pos:next_pos] = batch["input_ids"][
                    i, start_pos:next_pos
                ]

                # Move position to continue search
                pos = next_pos
            else:
                pos += 1

    # Set padding tokens to -100
    padding_mask = batch["attention_mask"] == 0
    labels[padding_mask] = -100

    # print(texts, batch["input_ids"], labels)
    # raise
    batch["labels"] = labels

    return batch


def crop_image(image, bbox):
    height, width = image.height, image.width

    try:
        coordinates = bbox.strip().replace("[", "").replace("].", "").split(",")
        if len(coordinates) != 4:
            raise ValueError("Invalid number of coordinates")

        x1_pct = float(coordinates[0])
        y1_pct = float(coordinates[1])
        x2_pct = float(coordinates[2])
        y2_pct = float(coordinates[3])

        # Validate coordinates are between 0-100 and x2>x1, y2>y1
        if not (
            0 <= x1_pct <= 100
            and 0 <= y1_pct <= 100
            and 0 <= x2_pct <= 100
            and 0 <= y2_pct <= 100
            and x2_pct > x1_pct
            and y2_pct > y1_pct
        ):
            raise ValueError("Invalid coordinate values")

        x1 = x1_pct * width / 100
        y1 = y1_pct * height / 100
        x2 = x2_pct * width / 100
        y2 = y2_pct * height / 100

        squaredness = min((x2 - x1) / (y2 - y1), (y2 - y1) / (x2 - x1))

    except:

        x1 = 0
        y1 = 0
        x2 = 2
        y2 = 2

        squaredness = None

    cropped_image = image.crop((x1, y1, x2, y2))

    return cropped_image, squaredness


def calculate_area_iou(bboxes):

    scores = []
    for bbox in bboxes:
        try:
            coordinates = bbox.strip().replace("[", "").replace("].", "").split(",")
            if len(coordinates) != 4:
                raise ValueError("Invalid number of coordinates")

            x1_pct = float(coordinates[0])
            y1_pct = float(coordinates[1])
            x2_pct = float(coordinates[2])
            y2_pct = float(coordinates[3])

            # Validate coordinates are between 0-100 and x2>x1, y2>y1
            if not (
                0 <= x1_pct <= 100
                and 0 <= y1_pct <= 100
                and 0 <= x2_pct <= 100
                and 0 <= y2_pct <= 100
                and x2_pct > x1_pct
                and y2_pct > y1_pct
            ):
                raise ValueError("Invalid coordinate values")

            valid = True
        except:
            valid = False

        scores.append(valid)

    areas = []
    valid_bboxes = []
    for i, s in enumerate(scores):
        if s:
            coordinates = (
                bboxes[i].strip().replace("[", "").replace("].", "").split(",")
            )
            valid_bboxes.append([float(x) for x in coordinates])
            x1_pct = float(coordinates[0])
            y1_pct = float(coordinates[1])
            x2_pct = float(coordinates[2])
            y2_pct = float(coordinates[3])
            areas.append((x2_pct - x1_pct) * (y2_pct - y1_pct) / 10000)

    if len(valid_bboxes) > 1:
        mean_area = sum(areas) / len(areas)

        # Calculate IoU between each bbox and all others
        ious = []
        for i, bbox1 in enumerate(valid_bboxes):
            for j, bbox2 in enumerate(valid_bboxes):
                if i != j:
                    # Calculate intersection coordinates
                    x1 = max(bbox1[0], bbox2[0])
                    y1 = max(bbox1[1], bbox2[1])
                    x2 = min(bbox1[2], bbox2[2])
                    y2 = min(bbox1[3], bbox2[3])

                    if x2 > x1 and y2 > y1:
                        intersection = (x2 - x1) * (y2 - y1)
                        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
                        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
                        union = area1 + area2 - intersection
                        ious.append(intersection / union)
                    else:
                        ious.append(0.0)

        iou = sum(ious) / len(ious) if ious else 0
    else:
        mean_area, iou = 0, 0

    return mean_area, iou


def get_ll(model, prompt_inputs, return_grads=False):
    logits = model(
        **prompt_inputs,
    ).logits  # (B, L, V)

    logits = logits[
        :, :-1, :
    ]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
    input_ids = prompt_inputs["input_ids"][:, 1:]

    # Find position of token 42 for each sequence in the batch
    start_positions = []
    for seq in input_ids:
        pos = (seq == 42).nonzero(as_tuple=True)[0]
        if len(pos) > 0:
            start_positions.append(pos[-1].item() + 1)
        else:
            # If token 42 not found, start from beginning
            raise ValueError(f"Token 42 not found in sequence: {seq}")

    # print(start_positions)

    # Compute the log probabilities for the input tokens after token 42
    per_token_logps = []
    for i, (logits_row, input_ids_row) in enumerate(zip(logits, input_ids)):
        log_probs = logits_row.log_softmax(dim=-1)
        start_pos = start_positions[i]
        # Only gather log probs starting from token 42
        token_log_prob = torch.gather(
            log_probs[start_pos:], dim=1, index=input_ids_row[start_pos:].unsqueeze(1)
        ).squeeze(1)
        per_token_logps.append(token_log_prob[:-1])
    lls = torch.stack(per_token_logps).sum(-1)
    if return_grads:
        return lls
    else:
        return lls.detach().cpu().to(torch.float32).numpy().tolist()


def get_single_answer(answers):
    answer_counts = {}
    for ans in answers:
        ans = ans.lower().strip()  # Normalize answers
        answer_counts[ans] = answer_counts.get(ans, 0) + 1
    answer = max(answer_counts.items(), key=lambda x: x[1])[0]

    return answer