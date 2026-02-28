"""
PCL Binary Classification — Training Script
Fine-tunes DeBERTa-v3-base with weighted cross-entropy loss.
"""

import sys
import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import f1_score, precision_score, recall_score

# ── Configuration ────────────────────────────────────────────
MODEL_NAME = "microsoft/deberta-v3-base"
MAX_LENGTH = 256
BATCH_SIZE = 16
LEARNING_RATE = 3e-5
HEAD_LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
EPOCHS = 10
PATIENCE = 3
SEED = 42

# Paths (relative to project root)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "dontpatronizeme", "semeval-2022")
SPLITS_DIR = os.path.join(DATA_DIR, "practice splits")
TEST_PATH = os.path.join(DATA_DIR, "TEST", "task4_test.tsv")
SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")

# Add the data module to path
sys.path.insert(0, DATA_DIR)
from dont_patronize_me import DontPatronizeMe


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class PCLDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=MAX_LENGTH):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


class PCLTestDataset(Dataset):
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


def load_data():
    """Load dataset and apply official train/dev split."""
    print("Loading data...")
    dpm = DontPatronizeMe(DATA_DIR, TEST_PATH)
    dpm.load_task1()
    dpm.load_test()

    df = dpm.train_task1_df.copy()
    df["par_id"] = df["par_id"].astype(int)

    # Official splits
    train_ids = pd.read_csv(os.path.join(SPLITS_DIR, "train_semeval_parids-labels.csv"))
    dev_ids = pd.read_csv(os.path.join(SPLITS_DIR, "dev_semeval_parids-labels.csv"))
    train_ids["par_id"] = train_ids["par_id"].astype(int)
    dev_ids["par_id"] = dev_ids["par_id"].astype(int)

    df_train = df[df["par_id"].isin(train_ids["par_id"])].reset_index(drop=True)
    df_dev = df[df["par_id"].isin(dev_ids["par_id"])].reset_index(drop=True)

    print(f"  Train: {len(df_train)} | Dev: {len(df_dev)}")
    print(f"  Train positive rate: {df_train['label'].mean():.4f}")
    print(f"  Dev positive rate:   {df_dev['label'].mean():.4f}")

    return df_train, df_dev, dpm.test_set_df


def compute_class_weights(labels):
    """Compute balanced class weights: N_total / (2 * N_class)."""
    n_total = len(labels)
    n_neg = (labels == 0).sum()
    n_pos = (labels == 1).sum()
    w_neg = n_total / (2.0 * n_neg)
    w_pos = n_total / (2.0 * n_pos)
    print(f"  Class weights: neg={w_neg:.4f}, pos={w_pos:.4f}")
    return torch.tensor([w_neg, w_pos], dtype=torch.float32)


def evaluate(model, dataloader, device):
    """Evaluate model on a dataloader. Returns metrics and raw probabilities."""
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=-1)[:, 1].cpu().numpy()

            all_probs.extend(probs)
            all_labels.extend(labels.numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    preds = (all_probs >= 0.5).astype(int)

    f1 = f1_score(all_labels, preds)
    prec = precision_score(all_labels, preds, zero_division=0)
    rec = recall_score(all_labels, preds, zero_division=0)

    return f1, prec, rec, all_probs, all_labels


def train():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load data ──
    df_train, df_dev, _ = load_data()

    # ── Tokeniser ──
    print(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # ── Datasets ──
    print("Tokenizing...")
    train_dataset = PCLDataset(
        df_train["text"].tolist(), df_train["label"].tolist(), tokenizer
    )
    dev_dataset = PCLDataset(
        df_dev["text"].tolist(), df_dev["label"].tolist(), tokenizer
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE * 2, shuffle=False)

    # ── Model ──
    print(f"Loading model: {MODEL_NAME}")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2
    )
    model.float()  # ensure full float32 precision
    model.to(device)

    # ── Loss with class weights ──
    class_weights = compute_class_weights(df_train["label"].values)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))

    # ── Optimiser with separate LR for head vs backbone ──
    head_params = list(model.classifier.parameters()) + list(model.pooler.parameters())
    head_param_ids = set(id(p) for p in head_params)
    backbone_params = [p for p in model.parameters() if id(p) not in head_param_ids]

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": LEARNING_RATE},
        {"params": head_params, "lr": HEAD_LEARNING_RATE},
    ], weight_decay=WEIGHT_DECAY)
    total_steps = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # ── Training loop ──
    best_f1 = 0.0
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.float()  # ensure float32 for loss
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches

        # ── Evaluate on dev ──
        f1, prec, rec, _, _ = evaluate(model, dev_loader, device)

        print(
            f"Epoch {epoch}/{EPOCHS} | "
            f"Loss: {avg_loss:.4f} | "
            f"Dev F1: {f1:.4f} | P: {prec:.4f} | R: {rec:.4f}"
        )

        # ── Early stopping & checkpointing ──
        if f1 > best_f1:
            best_f1 = f1
            patience_counter = 0
            print(f"  >> New best F1: {best_f1:.4f}. Saving model...")
            model.save_pretrained(SAVE_DIR)
            tokenizer.save_pretrained(SAVE_DIR)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  Early stopping at epoch {epoch} (patience={PATIENCE})")
                break

    print(f"\nTraining complete. Best dev F1: {best_f1:.4f}")
    print(f"Model saved to: {SAVE_DIR}")


if __name__ == "__main__":
    train()
