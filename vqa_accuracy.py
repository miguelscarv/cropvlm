import argparse
from datasets import load_dataset
import json
from utils import vqa_accuracy

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Specify dataset path and output file."
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        help="Dataset path",
        default="lmms-lab/textvqa",
    )
    parser.add_argument(
        "--predictions_file",
        type=str,
        required=True,
        help="Output file name",
        default="predictions.json",
    )
    return parser.parse_args()


args = parse_arguments()

with open(args.predictions_file, "r") as f:
    predictions = json.load(f)


dataset = load_dataset(args.dataset_path, split="validation")

all_accuracies = []
for prediction, row in zip(predictions, dataset):
    prediction_answer = prediction["answer"]
    true_answers = row["answers"]
    all_accuracies.append(vqa_accuracy(prediction_answer, true_answers) * 100)


print(sum(all_accuracies) / len(all_accuracies))
