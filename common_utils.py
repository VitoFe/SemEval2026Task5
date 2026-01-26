import json
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from scipy.stats import spearmanr


@dataclass
class StoryExample:
    """Data structure for a single story example"""
    id: str
    homonym: str
    judged_meaning: str
    precontext: str
    sentence: str
    ending: str
    example_sentence: str
    average: float = None
    stdev: float = None
    choices: List[int] = None


def load_data(file_path: str) -> List[StoryExample]:
    """
    Load data from JSON file and convert to StoryExample objects
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    examples = []
    for id_key, item in data.items():
        example = StoryExample(
            id=id_key,
            homonym=item["homonym"],
            judged_meaning=item["judged_meaning"],
            precontext=item["precontext"],
            sentence=item["sentence"],
            ending=item.get("ending", ""),
            example_sentence=item["example_sentence"],
            average=item.get("average"),
            stdev=item.get("stdev"),
            choices=item.get("choices"),
        )
        examples.append(example)

    return examples

def accuracy_within_std(
    predictions: np.ndarray, labels: np.ndarray, stdevs: np.ndarray
) -> float:
    """
    Calculate Accuracy Within Standard Deviation metric
    """
    abs_diff = np.abs(predictions - labels)
    thresholds = np.maximum(stdevs, 1.0)
    return np.mean(abs_diff <= thresholds)


def evaluate_predictions(
    predictions: np.ndarray, labels: np.ndarray, stdevs: np.ndarray
) -> Dict[str, float]:
    """
    Evaluate predictions using both metrics
    """
    spearman_corr, _ = spearmanr(predictions, labels)
    acc_within_std = accuracy_within_std(predictions, labels, stdevs)
    mae = np.mean(np.abs(predictions - labels))

    return {
        "spearman": spearman_corr,
        "accuracy_within_std": acc_within_std,
        "mae": mae,
    }


def save_predictions(predictions: List[Tuple[str, float]], output_path: str):
    with open(output_path, "w", encoding="utf-8") as f:
        for id_val, pred in predictions:
            f.write(json.dumps({"id": int(id_val), "prediction": float(pred)}) + "\n")


def clip_predictions(
    predictions: np.ndarray, min_val: float = 1.0, max_val: float = 5.0
) -> np.ndarray:
    """
    Clip predictions to valid range [1, 5]
    """
    return np.clip(predictions, min_val, max_val)


if __name__ == "__main__":
    print("Testing utilities...")
    examples = load_data("sample_data.json")
    print(f"Loaded {len(examples)} examples")
    example = examples[0]
    print(f"\nExample ID: {example.id}")
    print(f"Homonym: {example.homonym}")
    print(f"Average score: {example.average}")
    dummy_preds = np.array([3.5, 2.0, 1.0, 4.8, 2.4, 4.6])
    dummy_labels = np.array([3.6, 2.0, 1.0, 4.8, 2.4, 4.6])
    dummy_stdevs = np.array([1.95, 1.73, 0.0, 0.45, 1.52, 0.55])
    metrics = evaluate_predictions(dummy_preds, dummy_labels, dummy_stdevs)
    print(f"\nTest metrics: {metrics}")
