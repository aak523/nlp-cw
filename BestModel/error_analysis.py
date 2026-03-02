"""
Error Analysis — joins dev.txt predictions with source paragraphs.
Outputs error_analysis.csv containing only misclassified examples.
"""

import sys
import os
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "dontpatronizeme", "semeval-2022")
SPLITS_DIR = os.path.join(DATA_DIR, "practice splits")
TEST_PATH = os.path.join(DATA_DIR, "TEST", "task4_test.tsv")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, DATA_DIR)
from dont_patronize_me import DontPatronizeMe


def main():
    # ── Load data (mirrors predict.py exactly) ──
    dpm = DontPatronizeMe(DATA_DIR, TEST_PATH)
    dpm.load_task1()

    df = dpm.train_task1_df.copy()
    df["par_id"] = df["par_id"].astype(int)

    dev_ids = pd.read_csv(os.path.join(SPLITS_DIR, "dev_semeval_parids-labels.csv"))
    dev_ids["par_id"] = dev_ids["par_id"].astype(int)
    df_dev = df[df["par_id"].isin(dev_ids["par_id"])].reset_index(drop=True)

    # ── Load predictions ──
    preds_path = os.path.join(SCRIPT_DIR, "dev.txt")
    preds = pd.read_csv(preds_path, header=None, names=["pred"])

    assert len(preds) == len(df_dev), (
        f"Prediction count ({len(preds)}) does not match dev set size ({len(df_dev)})"
    )

    # ── Join ──
    df_dev["pred"] = preds["pred"].values
    df_dev["error_type"] = None
    df_dev.loc[(df_dev["label"] == 1) & (df_dev["pred"] == 0), "error_type"] = "FN"
    df_dev.loc[(df_dev["label"] == 0) & (df_dev["pred"] == 1), "error_type"] = "FP"

    df_errors = df_dev[df_dev["error_type"].notna()].copy()

    # ── Summary ──
    n_fn = (df_errors["error_type"] == "FN").sum()
    n_fp = (df_errors["error_type"] == "FP").sum()
    print(f"Dev set: {len(df_dev)} examples")
    print(f"Errors:  {len(df_errors)} total | FN={n_fn} | FP={n_fp}")

    # ── Write CSV ──
    out_cols = ["par_id", "keyword", "country", "orig_label", "pred", "error_type", "text"]
    out_path = os.path.join(SCRIPT_DIR, "error_analysis.csv")
    df_errors[out_cols].to_csv(out_path, index=False)
    print(f"Written: {out_path}")


if __name__ == "__main__":
    main()
