"""
PCL Binary Classification — Prediction Script
Loads the best checkpoint, tunes the classification threshold on the dev set,
and writes dev.txt and test.txt predictions.
"""

import sys
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import f1_score, precision_score, recall_score

# ── Configuration ────────────────────────────────────────────
MAX_LENGTH = 256
BATCH_SIZE = 32

# Paths (relative to project root)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "dontpatronizeme", "semeval-2022")
SPLITS_DIR = os.path.join(DATA_DIR, "practice splits")
TEST_PATH = os.path.join(DATA_DIR, "TEST", "task4_test.tsv")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "model")

# Add the data module to path
sys.path.insert(0, DATA_DIR)
from dont_patronize_me import DontPatronizeMe


class InferenceDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=MAX_LENGTH):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )

    def __len__(self):
        return self.encodings["input_ids"].shape[0]

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.encodings.items()}


def get_probabilities(model, dataloader, device):
    """Run inference and return positive-class probabilities."""
    model.eval()
    all_probs = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=-1)[:, 1].cpu().numpy()
            all_probs.extend(probs)
    return np.array(all_probs)


def find_best_threshold(probs, labels, low=0.30, high=0.70, step=0.01):
    """Sweep thresholds on dev set to maximise F1."""
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(low, high + step, step):
        preds = (probs >= t).astype(int)
        f1 = f1_score(labels, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
    return best_t, best_f1


def write_predictions(preds, path):
    """Write one integer per line."""
    with open(path, "w") as f:
        for p in preds:
            f.write(str(int(p)) + "\n")
    print(f"  Written {len(preds)} predictions to {path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load data ──
    print("Loading data...")
    dpm = DontPatronizeMe(DATA_DIR, TEST_PATH)
    dpm.load_task1()
    dpm.load_test()

    df = dpm.train_task1_df.copy()
    df["par_id"] = df["par_id"].astype(int)

    dev_ids = pd.read_csv(os.path.join(SPLITS_DIR, "dev_semeval_parids-labels.csv"))
    dev_ids["par_id"] = dev_ids["par_id"].astype(int)
    df_dev = df[df["par_id"].isin(dev_ids["par_id"])].reset_index(drop=True)
    df_test = dpm.test_set_df

    print(f"  Dev: {len(df_dev)} | Test: {len(df_test)}")

    # ── Load model ──
    print(f"Loading model from {MODEL_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.to(device)

    # ── Dev inference ──
    print("Running dev inference...")
    dev_dataset = InferenceDataset(df_dev["text"].tolist(), tokenizer)
    dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=False)
    dev_probs = get_probabilities(model, dev_loader, device)
    dev_labels = df_dev["label"].values

    # ── Threshold tuning ──
    print("Tuning threshold on dev set...")
    best_threshold, best_f1 = find_best_threshold(dev_probs, dev_labels)
    print(f"  Best threshold: {best_threshold:.2f} | Dev F1: {best_f1:.4f}")

    # Also report default threshold for comparison
    default_preds = (dev_probs >= 0.5).astype(int)
    default_f1 = f1_score(dev_labels, default_preds)
    print(f"  Default (0.50) threshold F1: {default_f1:.4f}")

    # ── Dev predictions ──
    dev_preds = (dev_probs >= best_threshold).astype(int)
    dev_f1 = f1_score(dev_labels, dev_preds)
    dev_prec = precision_score(dev_labels, dev_preds, zero_division=0)
    dev_rec = recall_score(dev_labels, dev_preds, zero_division=0)
    print(f"\nDev results (threshold={best_threshold:.2f}):")
    print(f"  F1: {dev_f1:.4f} | P: {dev_prec:.4f} | R: {dev_rec:.4f}")

    write_predictions(dev_preds, os.path.join(SCRIPT_DIR, "dev.txt"))

    # ── Test inference ──
    print("\nRunning test inference...")
    test_dataset = InferenceDataset(df_test["text"].tolist(), tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_probs = get_probabilities(model, test_loader, device)
    test_preds = (test_probs >= best_threshold).astype(int)

    write_predictions(test_preds, os.path.join(SCRIPT_DIR, "test.txt"))

    # ── Summary ──
    print(f"\nDone. Threshold: {best_threshold:.2f}")
    print(f"Dev: {len(dev_preds)} predictions | Test: {len(test_preds)} predictions")


if __name__ == "__main__":
    main()
