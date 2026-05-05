import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from models.logistic_regression import LogisticRegression
from models.baseline import ThresholdBaseline

DATA_PATH = "data/features.csv"
LOG_PATH = "experiments.log"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def normalize_splits(X_train, X_val, X_test):
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1.0
    return (X_train - mean) / std, (X_val - mean) / std, (X_test - mean) / std


def load_data(path):
    df = pd.read_csv(path)
    exclude = {"question_id", "label", "arc_split", "model_id", "task"}
    feature_cols = [c for c in df.columns if c not in exclude
                    and pd.api.types.is_numeric_dtype(df[c])]
    X = df[feature_cols].values.astype(np.float64)
    y = df["label"].values.astype(np.float64)
    return X, y, feature_cols


def log_experiment(params, metrics, model_name):
    entry = {
        "timestamp": datetime.now().isoformat(),
        "model": model_name,
        "params": params,
        "metrics": {k: round(float(v), 4) if isinstance(v, (int, float)) else v for k, v in metrics.items()},
    }
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")
    print(f"  Logged: {model_name} | acc={metrics['accuracy']:.4f} | f1={metrics['f1']:.4f} | auc={metrics.get('roc_auc', 0):.4f}")


def evaluate(y_true, y_pred, y_prob):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0,
    }


def run_grid_search(X_train, y_train, X_val, y_val, X_test, y_test):
    # brute force grid search over lr x l2, not glamorous but gets the job done
    learning_rates = [0.001, 0.01, 0.1]
    l2_lambdas = [0.0, 0.01, 0.1]
    best_val_f1, best_model, best_params = 0, None, None

    print("\n=== Logistic Regression Grid Search ===")
    for lr in learning_rates:
        for l2 in l2_lambdas:
            model = LogisticRegression(
                learning_rate=lr,
                num_epochs=500,
                batch_size=32,
                l2_lambda=l2,
                verbose=False,
            )
            model.fit(X_train, y_train, X_val, y_val)
            val_pred = model.predict(X_val)
            val_prob = model.predict_proba(X_val)
            val_metrics = evaluate(y_val, val_pred, val_prob)

            params = model.get_params()
            log_experiment(params, {**val_metrics, "split": "val"}, "LogisticRegression")

            if val_metrics["f1"] > best_val_f1:
                best_val_f1 = val_metrics["f1"]
                best_model = model
                best_params = params

    print(f"\nBest params: {best_params}")
    test_pred = best_model.predict(X_test)
    test_prob = best_model.predict_proba(X_test)
    test_metrics = evaluate(y_test, test_pred, test_prob)
    log_experiment(best_params, {**test_metrics, "split": "test"}, "LogisticRegression_best")
    print(f"Test results: {test_metrics}")

    return best_model, test_metrics


def run_baseline(X_train, y_train, X_val, y_val, X_test, y_test, feature_cols):
    print("\n=== Threshold Baseline ===")
    baseline = ThresholdBaseline(feature_names=feature_cols)
    baseline.fit(X_train, y_train)

    test_pred = baseline.predict(X_test)
    test_prob = baseline.predict_proba(X_test)
    metrics = evaluate(y_test, test_pred, test_prob)
    log_experiment({"margin_threshold": baseline.margin_threshold}, metrics, "ThresholdBaseline")
    return baseline, metrics


def save_predictions(model, X_test, y_test, name):
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)
    df = pd.DataFrame({"true": y_test, "pred": preds, "prob": probs})
    path = os.path.join(RESULTS_DIR, f"{name}_predictions.csv")
    df.to_csv(path, index=False)
    print(f"Saved predictions to {path}")
    return df

#by ug
def main():
    print(f"Loading data from {DATA_PATH}...")
    X, y, feature_cols = load_data(DATA_PATH)
    print(f"  {X.shape[0]} samples, {X.shape[1]} features")
    print(f"  Label balance: {int(y.sum())} positive / {len(y)} total ({y.mean():.2%})")

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.15, random_state=42, stratify=y_trainval
    )

    X_train, X_val, X_test = normalize_splits(X_train, X_val, X_test)
    print(f"  Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    best_model, lr_test_metrics = run_grid_search(X_train, y_train, X_val, y_val, X_test, y_test)
    baseline, base_test_metrics = run_baseline(X_train, y_train, X_val, y_val, X_test, y_test, feature_cols)

    save_predictions(best_model, X_test, y_test, "logistic_regression")
    save_predictions(baseline, X_test, y_test, "baseline")

    weights_df = pd.DataFrame({
        "feature": feature_cols,
        "weight": best_model.weights,
    }).sort_values("weight", ascending=False)
    weights_df.to_csv(os.path.join(RESULTS_DIR, "feature_weights.csv"), index=False)
    print(f"\nFeature weights:\n{weights_df.to_string(index=False)}")

    print("\n=== Final Comparison ===")
    print(f"{'Model':<25} {'Accuracy':>10} {'F1':>10} {'ROC-AUC':>10}")
    print("-" * 57)
    for name, m in [("LogisticRegression", lr_test_metrics), ("ThresholdBaseline", base_test_metrics)]:
        print(f"{name:<25} {m['accuracy']:>10.4f} {m['f1']:>10.4f} {m.get('roc_auc', 0):>10.4f}")

    np.save(os.path.join(RESULTS_DIR, "train_losses.npy"), best_model.train_losses)
    if best_model.val_losses:
        np.save(os.path.join(RESULTS_DIR, "val_losses.npy"), best_model.val_losses)


if __name__ == "__main__":
    main()
