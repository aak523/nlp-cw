"""
PCL Binary Classification — Layer Ablation Study
Evaluates the trained model with progressively fewer transformer layers
to measure the contribution of model depth to performance.
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

# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "dontpatronizeme", "semeval-2022")
SPLITS_DIR = os.path.join(DATA_DIR, "practice splits")
TEST_PATH = os.path.join(DATA_DIR, "TEST", "task4_test.tsv")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "model")

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
    """Sweep thresholds to maximise F1."""
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(low, high + step, step):
        preds = (probs >= t).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
    return best_t, best_f1


def evaluate_with_n_layers(model, n_layers, dev_loader, dev_labels, device):
    """Temporarily truncate the model to n_layers and evaluate."""
    encoder = model.roberta.encoder
    original_layers = encoder.layer
    original_num_hidden = model.config.num_hidden_layers

    # Truncate to first n_layers
    encoder.layer = original_layers[:n_layers]
    model.config.num_hidden_layers = n_layers

    try:
        probs = get_probabilities(model, dev_loader, device)
        best_thresh, best_f1 = find_best_threshold(probs, dev_labels)
        preds = (probs >= best_thresh).astype(int)
        prec = precision_score(dev_labels, preds, zero_division=0)
        rec = recall_score(dev_labels, preds, zero_division=0)
        return {
            "layers": n_layers,
            "f1": best_f1,
            "precision": prec,
            "recall": rec,
            "threshold": best_thresh,
        }
    finally:
        # Restore original model
        encoder.layer = original_layers
        model.config.num_hidden_layers = original_num_hidden


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

    print(f"  Dev: {len(df_dev)} examples")

    # ── Load model ──
    print(f"Loading model from {MODEL_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.to(device)

    # ── Prepare dev data ──
    dev_dataset = InferenceDataset(df_dev["text"].tolist(), tokenizer)
    dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=False)
    dev_labels = df_dev["label"].values

    # ── Run ablation ──
    layer_counts = [12, 10, 8, 6, 4, 2]
    results = []

    print("\n" + "=" * 65)
    print("LAYER ABLATION STUDY")
    print("Evaluating model with progressively fewer transformer layers")
    print("=" * 65)

    for n in layer_counts:
        print(f"\nEvaluating with {n}/12 layers...")
        result = evaluate_with_n_layers(model, n, dev_loader, dev_labels, device)
        results.append(result)
        print(
            f"  F1: {result['f1']:.4f} | P: {result['precision']:.4f} | "
            f"R: {result['recall']:.4f} | Thresh: {result['threshold']:.2f}"
        )

    # ── Summary table ──
    print("\n" + "=" * 65)
    print("SUMMARY")
    print("=" * 65)
    print(f"{'Layers':>8} {'F1':>8} {'Prec':>8} {'Recall':>8} {'Thresh':>8} {'D F1':>8}")
    print("-" * 56)

    baseline_f1 = results[0]["f1"]  # 12-layer result
    for r in results:
        delta = r["f1"] - baseline_f1
        sign = "+" if delta >= 0 else ""
        print(
            f"{r['layers']:>8} {r['f1']:>8.4f} {r['precision']:>8.4f} "
            f"{r['recall']:>8.4f} {r['threshold']:>8.2f} {sign}{delta:>7.4f}"
        )

    print("-" * 56)
    print(f"Baseline (12 layers) F1: {baseline_f1:.4f}")


if __name__ == "__main__":
    main()
