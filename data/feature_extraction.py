import os
import pandas as pd

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(DATA_DIR, "raw_predictions.csv")
OUTPUT_PATH = os.path.join(DATA_DIR, "features.csv")

# drop any col that directly encodes the label, straight up leakage otherwise
EXCLUDE_COLS = {"model_id", "label", "acc_gap", "reasoning_score", "knowledge_score", "total_score", "gpqa_acc"}


def get_feature_columns(df):
    return [
        c for c in df.columns
        if c not in EXCLUDE_COLS and pd.api.types.is_numeric_dtype(df[c])
    ]

#by ug
def main():
    if not os.path.exists(INPUT_PATH):
        print(f"Input not found: {INPUT_PATH}\nRun load_dataset.py first.")
        return

    df = pd.read_csv(INPUT_PATH).dropna()
    print(f"Loaded {len(df)} rows")

    feature_cols = get_feature_columns(df)
    print(f"Features ({len(feature_cols)}): {feature_cols}")

    # clean features + label only, keeping train.py honest
    df[["model_id"] + feature_cols + ["label"]].to_csv(OUTPUT_PATH, index=False)
    print(f"\nLabel distribution:\n{df['label'].value_counts().to_string()}")
    print(f"Saved {len(df)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
