"""
Approach to SemEval 2026 Task 5 using DeBERTa-v3-Large Regression
Memory-optimized for consumer GPUs
Using Optuna for hyperparameter optimization
Optimizing for Accuracy within SD + Spearman combination.
"""

import argparse
import json
import os
import shutil
import statistics

import numpy as np
import optuna
import requests
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    DebertaV2Tokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from common_utils import load_data


def ensure_data_exists(train_path: str, test_path: str) -> None:
    """
    Ensure that the train and test data files exist.
    If they don't exist, download them from the GitHub repository.
    """
    GITHUB_RAW_BASE = "https://raw.githubusercontent.com/Janosch-Gehring/ambistory/main"

    files_to_check = [
        (train_path, f"{GITHUB_RAW_BASE}/train.json"),
        (test_path, f"{GITHUB_RAW_BASE}/dev.json"),
    ]

    for file_path, github_url in files_to_check:
        if os.path.exists(file_path):
            print(f"  [Data] Found: {file_path}")
            continue
        print(f"  [Data] {file_path} not found. Downloading from GitHub...")
        data_dir = os.path.dirname(file_path)
        if data_dir and not os.path.exists(data_dir):
            os.makedirs(data_dir)
            print(f"  [Data] Created directory: {data_dir}")
        try:
            response = requests.get(github_url, timeout=30)
            response.raise_for_status()

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(response.text)

            print(f"  [Data] Downloaded: {file_path}")
        except requests.exceptions.RequestException as e:
            print(f"  [Error] Failed to download {file_path}: {e}")
            print(
                "  [Info] Manually: git clone https://github.com/Janosch-Gehring/ambistory.git data"
            )
            raise SystemExit(1)


def is_within_standard_deviation(prediction, labels):
    avg = sum(labels) / len(labels)
    stdev = statistics.stdev(labels) if len(labels) > 1 else 0.0
    if (avg - stdev) < prediction < (avg + stdev):
        return True
    if abs(avg - prediction) < 1:
        return True
    return False


def tag_sentence(sentence, homonym):
    """
    Highlight the homonym in the sentence with uppercase for emphasis.
    Better than asterisks as it preserves tokenization integrity.
    """
    if not homonym or not sentence:
        return sentence

    import re

    try:
        # Convert homonym to uppercase while preserving case-insensitive matching
        tagged = re.sub(
            f"\\b({re.escape(homonym)})\\b",
            lambda m: m.group(1).upper(),
            sentence,
            flags=re.IGNORECASE,
        )
        return tagged
    except Exception:
        return sentence


def format_input_parts(example):
    """
    Format input as a single concatenated text string.
    Format: {homonym}: {meaning} Example: {example} Story: {precontext} {sentence}[SEP]{ending}
    """
    homonym = example.homonym if hasattr(example, "homonym") else ""
    meaning = example.judged_meaning if hasattr(example, "judged_meaning") else ""
    example_sentence = (
        example.example_sentence if hasattr(example, "example_sentence") else ""
    )
    precontext = example.precontext if hasattr(example, "precontext") else ""
    sentence = example.sentence if hasattr(example, "sentence") else ""
    ending = example.ending if hasattr(example, "ending") else ""

    # Tag the homonym in the sentence
    tagged_sentence = tag_sentence(sentence, homonym)

    # Handle missing ending
    if len(ending) > 0:
        refined_ending = ending
    else:
        refined_ending = "No ending provided"

    # Combine into single text (tokenizer will add [SEP] tokens automatically)
    text = f"{homonym}: {meaning} Example: {example_sentence} Story: {precontext} {tagged_sentence} [SEP] {refined_ending}"

    return text


def semantic_group_split(examples, test_ratio=0.15, seed=42):
    """
    Split examples by semantic groups (story/homonym combinations).

    prevents data leakage by ensuring all meanings of the same story
    stay together in either train or val.
    """
    import random

    # groups: {group_signature: [list of examples]}
    groups = {}
    for ex in examples:
        sig = (ex.precontext, ex.sentence, ex.homonym)
        if sig not in groups:
            groups[sig] = []
        groups[sig].append(ex)

    # shuffle keys
    group_keys = list(groups.keys())
    rng = random.Random(seed)
    rng.shuffle(group_keys)

    total_examples = len(examples)
    target_val_size = int(total_examples * test_ratio)

    val_examples = []
    train_examples = []

    for key in group_keys:
        group_examples = groups[key]
        if len(val_examples) < target_val_size:
            val_examples.extend(group_examples)
        else:
            train_examples.extend(group_examples)

    actual_ratio = len(val_examples) / total_examples if total_examples > 0 else 0
    print(
        f"  [Semantic Split] {len(groups)} story groups -> "
        f"Train: {len(train_examples)}, Val: {len(val_examples)} "
        f"(actual ratio: {actual_ratio:.1%})"
    )

    return train_examples, val_examples


class PlausibilityDataset(Dataset):
    """Dataset for plausibility regression task"""

    def __init__(self, examples, tokenizer, max_length=512, truncation_side="right"):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.truncation_side = truncation_side

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]

        text = format_input_parts(example)

        original_truncation_side = self.tokenizer.truncation_side
        self.tokenizer.truncation_side = self.truncation_side

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )

        self.tokenizer.truncation_side = original_truncation_side

        item = {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
        }

        if example.average is not None:
            item["labels"] = torch.tensor(example.average, dtype=torch.float)

        if example.choices is not None:
            item["choices"] = example.choices

        # stdev for uncertainty-aware training and accuracy-aware loss
        if example.stdev is not None:
            item["stdev"] = torch.tensor(example.stdev, dtype=torch.float)

        return item


class CustomDataCollator:
    """Dynamic padding collator for efficiency"""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        labels = None
        choices_list = None
        stdevs = None

        if "labels" in features[0]:
            labels = torch.tensor([f.pop("labels") for f in features])
        if "choices" in features[0]:
            choices_list = [f.pop("choices") for f in features]
        if "stdev" in features[0]:
            stdevs = torch.tensor([f.pop("stdev") for f in features])

        batch = self.tokenizer.pad(
            features,
            padding=True,
            return_tensors="pt",
        )

        if labels is not None:
            batch["labels"] = labels
        if choices_list is not None:
            batch["choices"] = choices_list
        if stdevs is not None:
            batch["stdevs"] = stdevs

        return batch


class MetricsCalculator:
    def __init__(self, examples):
        self.current_choices = [ex.choices for ex in examples]

    def __call__(self, eval_pred):
        """Compute metrics (called by Trainer during eval)"""
        predictions, labels = eval_pred
        predictions = predictions.squeeze()

        # REVERSE SIGMOID NORMALIZATION
        predictions = 1.0 / (1.0 + np.exp(-predictions))
        predictions = predictions * LABEL_RANGE + LABEL_MIN

        # clip predictions to [LABEL_MIN, LABEL_MAX]
        predictions = np.clip(predictions, LABEL_MIN, LABEL_MAX)

        from scipy.stats import spearmanr

        spearman_corr, _ = spearmanr(predictions, labels)

        # Accuracy within SD
        if self.current_choices and len(self.current_choices) == len(predictions):
            correct = sum(
                1
                for pred, choices in zip(predictions, self.current_choices)
                if is_within_standard_deviation(pred, choices)
            )
            acc_within_sd = correct / len(predictions)

            # Continuous score for smoother hp optimization
            # average soft error where values within threshold are near 0
            soft_acc_score = 0
            for pred, choices in zip(predictions, self.current_choices):
                avg = sum(choices) / len(choices)
                stdev = statistics.stdev(choices) if len(choices) > 1 else 1.0
                threshold = max(stdev, 1.0)
                error = abs(pred - avg)
                # Sigmoid-like reward: 1 if within threshold, decays outside
                reward = 1.0 / (1.0 + np.exp(5.0 * (error - threshold)))
                soft_acc_score += reward
            eval_soft_acc = soft_acc_score / len(predictions)
        else:
            acc_within_sd = np.mean(np.abs(predictions - labels) < 1.0)
            eval_soft_acc = acc_within_sd

        return {
            "spearman": spearman_corr,
            "acc_within_sd": acc_within_sd,
            "soft_acc": eval_soft_acc,
        }

    def update_choices(self, new_examples):
        """Update choices for a different dataset"""
        self.current_choices = [ex.choices for ex in new_examples]


class RegressionTrainer(Trainer):
    """Custom trainer with uncertainty weighting and accuracy-aware loss.

    - Uncertainty-aware weighting reduces impact of high-disagreement examples
    - Accuracy-aware loss: differentiable approximation of the accuracy metric
    """

    def __init__(
        self,
        use_uncertainty_weighting=False,
        uncertainty_scale=1.0,
        accuracy_loss_weight=0.0,
        accuracy_loss_temperature=10.0,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.use_uncertainty_weighting = use_uncertainty_weighting
        self.uncertainty_scale = uncertainty_scale
        self.accuracy_loss_weight = accuracy_loss_weight
        self.accuracy_loss_temperature = accuracy_loss_temperature

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs["labels"]
        stdevs = inputs.get("stdevs", None)

        model_inputs = {
            k: v for k, v in inputs.items() if k not in ["labels", "stdevs", "choices"]
        }

        outputs = model(**model_inputs)

        logits = outputs.logits.view(-1)

        # Bounded Regression (BCE With Logits)
        normalized_labels = (labels - 1.0) / 4.0
        normalized_labels = torch.clamp(normalized_labels, 0.0, 1.0)

        # Uncertainty-aware weighting using exponential decay
        # High stdev (disagreement) -> lower weight, low stdev (consensus) -> higher weight
        # weight = exp(-scale * stdev), clipped to minimum of 0.1
        if self.use_uncertainty_weighting and stdevs is not None:
            weights = torch.exp(-self.uncertainty_scale * stdevs)
            weights = torch.clamp(weights, min=0.1)
            weights = weights / weights.mean()

            # weighted BCE loss
            bce_unreduced = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, normalized_labels.view(-1), reduction="none"
            )
            bce_loss = (bce_unreduced * weights).mean()
        else:
            bce_loss = torch.nn.BCEWithLogitsLoss()(logits, normalized_labels.view(-1))

        loss = bce_loss

        # Accuracy-Aware Loss (differentiable approximation of accuracy metric)
        # This loss encourages predictions to be within the threshold, not exact matches
        if self.accuracy_loss_weight > 0 and stdevs is not None:
            preds = torch.sigmoid(logits) * 4.0 + 1.0  # [1, 5] range
            threshold = torch.clamp(stdevs, min=1.0)
            error = torch.abs(preds - labels)
            # Soft accuracy: sigmoid approximation of step function
            # High value when error < threshold, low value when error > threshold
            # Temperature controls sharpness (higher = sharper, more like step function)
            k = self.accuracy_loss_temperature
            soft_correct = torch.sigmoid(k * (threshold - error))
            accuracy_loss = 1.0 - soft_correct.mean()
            loss = loss + (self.accuracy_loss_weight * accuracy_loss)

        return (loss, outputs) if return_outputs else loss


MODEL_NAME = "microsoft/deberta-v3-large"
# 99th percentile: 139, 95th percentile: 128
MAX_LENGTH = 140

# label normalization constants (rating scale 1-5)
LABEL_MIN = 1.0
LABEL_MAX = 5.0
LABEL_RANGE = LABEL_MAX - LABEL_MIN  # 4.0
PRED_CLIP_MIN = 1.0
PRED_CLIP_MAX = 5.0

TRAIN_EXAMPLES = []
VAL_EXAMPLES = []
EPOCHS = 10
BATCH_SIZE = 6
USE_UNCERTAINTY_WEIGHTING = True  # --use_uncertainty_weighting
USE_ACCURACY_LOSS = False  # --use_accuracy_loss
TEST_EXAMPLES = []
HAS_OFFICIAL_TEST = False  # when official test.json is present (no labels)


def create_model_and_tokenizer():
    tokenizer = DebertaV2Tokenizer.from_pretrained(MODEL_NAME, use_fast=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=1,
        ignore_mismatched_sizes=True,
        problem_type="regression",
    )
    # for low vram
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    return model, tokenizer


def generate_predictions(
    model,
    tokenizer,
    examples: list,
    truncation_side: str = "left",
    batch_size: int = 8,
    show_progress: bool = True,
    clip_min: float = None,
    clip_max: float = None,
    temperature: float = 1.0,
) -> list:
    """Generate predictions for a list of examples.

    Args:
        temperature: Temperature scaling for logits.
                     < 1.0 = more extreme predictions (wider range)
                     > 1.0 = more conservative predictions (narrower range)
                     = 1.0 = no scaling (default)
    """
    model.eval()
    device = next(model.parameters()).device

    effective_clip_min = clip_min if clip_min is not None else PRED_CLIP_MIN
    effective_clip_max = clip_max if clip_max is not None else PRED_CLIP_MAX

    predictions = []

    for i in range(0, len(examples), batch_size):
        batch_examples = examples[i : i + batch_size]

        texts = [format_input_parts(ex) for ex in batch_examples]

        tokenizer.truncation_side = truncation_side
        encodings = tokenizer(
            texts,
            max_length=MAX_LENGTH,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )

        input_ids = encodings["input_ids"].to(device)
        attention_mask = encodings["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.squeeze(-1)

            # Apply temperature scaling
            logits = logits / temperature

            # First apply sigmoid to get [0, 1], then scale to [1, 5]
            preds = torch.sigmoid(logits) * LABEL_RANGE + LABEL_MIN
            # apply custom clipping bounds
            preds = torch.clamp(preds, effective_clip_min, effective_clip_max)

        for j, ex in enumerate(batch_examples):
            pred_value = preds[j].item()
            predictions.append((ex.id, pred_value))

        if show_progress and (
            (i + batch_size) % 100 == 0 or (i + batch_size) >= len(examples)
        ):
            print(
                f"  Processed {min(i + batch_size, len(examples))}/{len(examples)} examples"
            )

    return predictions


def save_predictions_jsonl(predictions: list, output_path: str):
    import json

    with open(output_path, "w", encoding="utf-8") as f:
        for id_val, pred in predictions:
            pred_rounded = round(pred, 2)
            f.write(json.dumps({"id": int(id_val), "prediction": pred_rounded}) + "\n")
    print(f"  [Saved] Predictions saved to: {output_path}")


def create_datasets(train_examples, val_examples, tokenizer, truncation_side="right"):
    train_dataset = PlausibilityDataset(
        train_examples, tokenizer, MAX_LENGTH, truncation_side
    )
    val_dataset = PlausibilityDataset(
        val_examples, tokenizer, MAX_LENGTH, truncation_side
    )
    data_collator = CustomDataCollator(tokenizer)
    return train_dataset, val_dataset, data_collator


def create_trainer(
    model,
    tokenizer,
    train_dataset,
    val_dataset,
    val_examples,
    data_collator,
    output_dir,
    epochs,
    learning_rate,
    weight_decay,
    warmup_ratio,
    early_stopping_patience=4,
    batch_size=8,
    max_grad_norm=None,
    use_uncertainty_weighting=False,
    uncertainty_scale=1.0,
    max_steps=None,
    accuracy_loss_weight=0.0,
    accuracy_loss_temperature=10.0,
):
    """Create the trainer with the given hyperparameters.

    Args:
        use_uncertainty_weighting: weight the loss by annotation agreement
                                   high consensus -> higher weight
        uncertainty_scale: How aggressively to downweight uncertain examples (default 1.0).
                          higher values -> more aggressive
        max_steps: If set, train for this many steps instead of using epochs.
                   for full training mode where we know the optimal duration.
        accuracy_loss_weight: Weight for the accuracy-aware loss (default 0.0 = disabled).
        accuracy_loss_temperature: Sharpness of the soft accuracy sigmoid (default 10.0).
    """

    def get_optimizer_grouped_parameters(model, lr, weight_decay):
        """
        Grouped parameters for weight decay.
        """
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        return optimizer_grouped_parameters

    # evaluation frequency based on dataset size
    steps_per_epoch = max(1, len(train_dataset) // batch_size)
    eval_steps = max(steps_per_epoch // 2, 10)  # at least every 10 steps
    if steps_per_epoch < 20:
        eval_steps = steps_per_epoch

    print(f"  [Eval] Steps per epoch: {steps_per_epoch}, eval every {eval_steps} steps")
    if use_uncertainty_weighting:
        print(f"  [Loss] Uncertainty weighting: enabled, scale={uncertainty_scale}")
    if accuracy_loss_weight > 0:
        print(
            f"  [Loss] Accuracy-aware loss: enabled, weight={accuracy_loss_weight}, temp={accuracy_loss_temperature}"
        )

    # Determine training duration
    if max_steps is not None:
        effective_epochs = (max_steps // steps_per_epoch) + 2
        print(f"  [Duration] max_steps={max_steps} (overrides epochs)")
    else:
        effective_epochs = epochs
        max_steps = -1  # TrainingArguments uses -1 to mean "use epochs"

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=effective_epochs,
        max_steps=max_steps if max_steps > 0 else -1,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        max_grad_norm=max_grad_norm,
        logging_steps=min(eval_steps, 50),  # log at least as often as we eval
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=eval_steps,
        load_best_model_at_end=True,  # load best model for evaluation
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        # dataloader_num_workers=2,
        # dataloader_pin_memory=True,
        report_to="none",
        seed=42,
        metric_for_best_model="acc_within_sd",
        greater_is_better=True,
        optim="adafactor",
    )

    # Grouped parameters for weight decay
    grouped_params = get_optimizer_grouped_parameters(
        model, learning_rate, weight_decay
    )

    from transformers.optimization import Adafactor

    optimizer = Adafactor(
        grouped_params,
        lr=learning_rate,
        weight_decay=weight_decay,
        scale_parameter=False,
        relative_step=False,
        warmup_init=False,
        clip_threshold=1.0,
    )

    metrics_calc = MetricsCalculator(val_examples)

    # Only add early stopping if patience is specified
    callbacks = []
    if early_stopping_patience is not None:
        callbacks.append(
            EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)
        )
    else:
        print(f"  [Training] Early stopping: DISABLED (full training mode)")

    trainer = RegressionTrainer(
        use_uncertainty_weighting=use_uncertainty_weighting,
        uncertainty_scale=uncertainty_scale,
        accuracy_loss_weight=accuracy_loss_weight,
        accuracy_loss_temperature=accuracy_loss_temperature,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
        compute_metrics=metrics_calc,
        optimizers=(optimizer, None),  # custom optimizer for weight decay grouping
        callbacks=callbacks if callbacks else None,
    )

    trainer.metrics_calc = metrics_calc

    return trainer


def run_training(trainer):
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    trainer.train()

    if torch.cuda.is_available():
        peak_mem = torch.cuda.max_memory_allocated() / 1024**3
        print(f"  [Memory] Peak GPU memory: {peak_mem:.2f} GB")

    training_info = {
        "best_step": trainer.state.best_global_step
        if hasattr(trainer.state, "best_global_step")
        else None,
        "best_metric": trainer.state.best_metric
        if hasattr(trainer.state, "best_metric")
        else None,
        "total_steps": trainer.state.global_step,
        "epochs_completed": trainer.state.epoch,
    }

    if training_info["best_step"] is not None and trainer.state.max_steps > 0:
        steps_per_epoch = trainer.state.max_steps / trainer.args.num_train_epochs
        training_info["best_epoch"] = (
            training_info["best_step"] / steps_per_epoch if steps_per_epoch > 0 else 0
        )
    else:
        training_info["best_epoch"] = None

    eval_results = trainer.evaluate()
    return eval_results, training_info


def objective(trial):
    learning_rate = trial.suggest_float("learning_rate", 2e-6, 2e-4, log=True)
    weight_decay = trial.suggest_float("weight_decay", 0.05, 0.15, step=0.0001)
    warmup_ratio = trial.suggest_float("warmup_ratio", 0.01, 0.2, step=0.0001)
    truncation_side = "left"  # analysis showed LEFT truncation preserves critical info
    max_grad_norm = None  # rely on Adafactor's update clipping

    if USE_UNCERTAINTY_WEIGHTING:
        uncertainty_scale = trial.suggest_float(
            "uncertainty_scale",
            1.5,
            3.0,
            step=0.001,
        )
    else:
        uncertainty_scale = 1.0

    if USE_ACCURACY_LOSS:
        accuracy_loss_weight = trial.suggest_float(
            "accuracy_loss_weight",
            0.6,
            1.0,
            step=0.001,
        )
        accuracy_loss_temperature = trial.suggest_float(
            "accuracy_loss_temperature",
            2.0,
            12.0,
            step=0.5,
        )
        pred_clip_min = trial.suggest_float("pred_clip_min", 1.0, 2.2, step=0.001)
        pred_clip_max = trial.suggest_float("pred_clip_max", 3.8, 5.0, step=0.001)
    else:
        accuracy_loss_weight = 0.0
        accuracy_loss_temperature = 10.0
        pred_clip_min = PRED_CLIP_MIN
        pred_clip_max = PRED_CLIP_MAX

    trial_output_dir = f"./optuna_trials_large/trial_{trial.number}"

    model, tokenizer = create_model_and_tokenizer()
    train_dataset, val_dataset, data_collator = create_datasets(
        TRAIN_EXAMPLES, VAL_EXAMPLES, tokenizer, truncation_side
    )

    trainer = create_trainer(
        model,
        tokenizer,
        train_dataset,
        val_dataset,
        VAL_EXAMPLES,
        data_collator,
        output_dir=trial_output_dir,
        epochs=EPOCHS,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        early_stopping_patience=4,
        batch_size=BATCH_SIZE,
        max_grad_norm=max_grad_norm,
        use_uncertainty_weighting=USE_UNCERTAINTY_WEIGHTING,
        uncertainty_scale=uncertainty_scale,
        accuracy_loss_weight=accuracy_loss_weight,
        accuracy_loss_temperature=accuracy_loss_temperature,
    )

    eval_results, training_info = run_training(trainer)

    spearman = eval_results["eval_spearman"]
    accuracy = eval_results["eval_acc_within_sd"]
    soft_accuracy = eval_results["eval_soft_acc"]

    if HAS_OFFICIAL_TEST and len(TEST_EXAMPLES) > 0:
        predictions_dir = "./trial_predictions"
        os.makedirs(predictions_dir, exist_ok=True)

        # predictions with multiple temperature values
        temperatures = [1.0, 1.1, 0.98, 0.95]

        for temp in temperatures:
            temp_suffix = f"_temp{temp}" if temp != 1.0 else ""

            # Test set predictions
            predictions_file = os.path.join(
                predictions_dir, f"predictions-{trial.number}{temp_suffix}.jsonl"
            )
            predictions = generate_predictions(
                trainer.model,
                tokenizer,
                TEST_EXAMPLES,
                truncation_side=truncation_side,
                batch_size=BATCH_SIZE,
                show_progress=False,
                clip_min=pred_clip_min,
                clip_max=pred_clip_max,
                temperature=temp,
            )
            save_predictions_jsonl(predictions, predictions_file)

            # Dev set predictions
            dev_predictions_file = os.path.join(
                predictions_dir, f"predictions-{trial.number}_dev{temp_suffix}.jsonl"
            )
            dev_predictions = generate_predictions(
                trainer.model,
                tokenizer,
                VAL_EXAMPLES,
                truncation_side=truncation_side,
                batch_size=BATCH_SIZE,
                show_progress=False,
                clip_min=pred_clip_min,
                clip_max=pred_clip_max,
                temperature=temp,
            )
            save_predictions_jsonl(dev_predictions, dev_predictions_file)

        print(f"  [Predictions] Generated for temperatures: {temperatures}")

    # cleanup
    del model, tokenizer, trainer
    torch.cuda.empty_cache()
    if os.path.exists(trial_output_dir):
        shutil.rmtree(trial_output_dir)

    # combined score for optimization (weight accuracy higher, main leaderboard metric)
    # use soft_accuracy for smoother optimization landscape
    combined_score = 0.2 * spearman + 0.8 * soft_accuracy
    trial.set_user_attr("spearman", spearman)
    trial.set_user_attr("accuracy", accuracy)
    trial.set_user_attr("soft_accuracy", soft_accuracy)
    trial.set_user_attr("use_uncertainty_weighting", USE_UNCERTAINTY_WEIGHTING)
    trial.set_user_attr("use_accuracy_loss", USE_ACCURACY_LOSS)
    trial.set_user_attr("pred_clip_min", pred_clip_min)
    trial.set_user_attr("pred_clip_max", pred_clip_max)

    if training_info["best_step"] is not None:
        trial.set_user_attr("best_step", training_info["best_step"])
    if training_info["best_epoch"] is not None:
        trial.set_user_attr("best_epoch", round(training_info["best_epoch"], 2))
    trial.set_user_attr("epochs_completed", round(training_info["epochs_completed"], 2))
    trial.set_user_attr("total_steps", training_info["total_steps"])

    return combined_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Optimization script for DeBERTa-v3-Large"
    )
    parser.add_argument("--train_data", type=str, default="data/train.json")
    parser.add_argument("--test_data", type=str, default="data/dev.json")
    parser.add_argument("--n_trials", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument(
        "--storage",
        type=str,
        default="sqlite:///optuna_study_large.db",
        help="Database URL for Optuna persistence",
    )
    parser.add_argument(
        "--study_name",
        type=str,
        default="deberta_large_optimization",
        help="Unique name for the study",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./deberta_large_best",
        help="Output directory for trained model",
    )
    parser.add_argument(
        "--use_uncertainty_weighting",
        action="store_true",
        help="Enable uncertainty-aware training: weight loss by annotation agreement. ",
    )
    parser.add_argument(
        "--use_accuracy_loss",
        action="store_true",
        help="Enable accuracy-aware loss: directly optimizes the accuracy metric. "
        "Also tunes prediction clipping bounds to exploit stdev >= 1.0 rule.",
    )
    args = parser.parse_args()

    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    USE_UNCERTAINTY_WEIGHTING = args.use_uncertainty_weighting
    USE_ACCURACY_LOSS = args.use_accuracy_loss

    print("=" * 60)
    print("DeBERTa-v3-Large Optimization")
    print("=" * 60)
    print(f"  - Batch size: {BATCH_SIZE}")
    print(f"  - Max length: {MAX_LENGTH} tokens")
    if USE_UNCERTAINTY_WEIGHTING:
        print("  - Uncertainty weighting: ENABLED")
    if USE_ACCURACY_LOSS:
        print("  - Accuracy-aware loss: ENABLED (will tune clipping bounds)")
    print("=" * 60)

    print("\nChecking data files...")
    ensure_data_exists(args.train_data, args.test_data)

    print("\nLoading data...")

    test_json_path = os.path.join(os.path.dirname(args.train_data), "test.json")

    if os.path.exists(test_json_path):
        print("  [!] Official test.json found")
        print("  [Optuna] Will generate predictions on test.json")

        TRAIN_EXAMPLES = load_data(args.train_data)
        VAL_EXAMPLES = load_data(args.test_data)
        TEST_EXAMPLES = load_data(test_json_path)
        HAS_OFFICIAL_TEST = True

        print(
            f"  Loaded {len(TRAIN_EXAMPLES)} train, {len(VAL_EXAMPLES)} val, {len(TEST_EXAMPLES)} test examples."
        )
    else:
        # train.json for training, dev.json for validation
        TRAIN_EXAMPLES = load_data(args.train_data)
        VAL_EXAMPLES = load_data(args.test_data)
        print(
            f"  Loaded {len(TRAIN_EXAMPLES)} train, {len(VAL_EXAMPLES)} val examples."
        )

    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU: {torch.cuda.get_device_name(0)} ({gpu_mem:.1f} GB)")

    print(f"\nStarting Optuna Study '{args.study_name}'...")
    print(f"Storage: {args.storage}")

    study = optuna.create_study(
        direction="maximize",
        storage=args.storage,
        study_name=args.study_name,
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=args.n_trials)

    print("\n" + "=" * 50)
    print("Optimization Finished")
    print("=" * 50)

    best_trial = study.best_trial
    print(f"Best Trial Score: {best_trial.value:.4f}")
    print("Best Hyperparameters:")
    for key, value in best_trial.params.items():
        print(f"  {key}: {value}")

    print("\nBest Trial Metrics:")
    print(f"  Spearman: {best_trial.user_attrs['spearman']:.4f}")
    print(f"  Accuracy within SD: {best_trial.user_attrs['accuracy']:.4f}")

    with open("optuna_best_params_large.json", "w") as f:
        json.dump(best_trial.params, f, indent=4)
    print("\nSaved best parameters to optuna_best_params_large.json")
