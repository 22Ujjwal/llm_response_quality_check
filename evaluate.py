import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc
)

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def plot_confusion_matrix(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Memorization", "Reasoning"],
                yticklabels=["Memorization", "Reasoning"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, filename)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_roc_curve(models_data, filename):
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, y_true, y_prob in models_data:
        if len(np.unique(y_true)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})", linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, filename)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_training_loss(train_losses, val_losses=None, filename="training_loss.png"):
    fig, ax = plt.subplots(figsize=(7, 4))
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, label="Train Loss", linewidth=2)
    if val_losses is not None and len(val_losses) > 0:
        ax.plot(epochs, val_losses, label="Val Loss", linewidth=2, linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Binary Cross-Entropy Loss")
    ax.set_title("Training Curve")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, filename)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_feature_importance(weights_path, filename="feature_importance.png"):
    # green = pushes toward high GPQA, red = opposite, color-coded for quick reads
    df = pd.read_csv(weights_path).sort_values("weight")
    fig, ax = plt.subplots(figsize=(7, max(4, len(df) * 0.4)))
    colors = ["#d73027" if w < 0 else "#1a9850" for w in df["weight"]]
    ax.barh(df["feature"], df["weight"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Weight")
    ax.set_title("Logistic Regression Feature Weights\n(positive → reasoning, negative → memorization)")
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, filename)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def print_metrics_table(models_preds):
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    print("\n" + "=" * 65)
    print(f"{'Model':<25} {'Accuracy':>10} {'F1':>10} {'Precision':>10} {'Recall':>10}")
    print("-" * 65)
    for name, y_true, y_pred in models_preds:
        print(f"{name:<25} "
              f"{accuracy_score(y_true, y_pred):>10.4f} "
              f"{f1_score(y_true, y_pred, zero_division=0):>10.4f} "
              f"{precision_score(y_true, y_pred, zero_division=0):>10.4f} "
              f"{recall_score(y_true, y_pred, zero_division=0):>10.4f}")
    print("=" * 65)

#by ug
def main():
    lr_path = os.path.join(RESULTS_DIR, "logistic_regression_predictions.csv")
    base_path = os.path.join(RESULTS_DIR, "baseline_predictions.csv")

    if not os.path.exists(lr_path):
        print("No predictions found. Run train.py first.")
        return

    lr_df = pd.read_csv(lr_path)
    y_true = lr_df["true"].values
    lr_pred = lr_df["pred"].values
    lr_prob = lr_df["prob"].values

    models_preds = [("LogisticRegression", y_true, lr_pred)]
    models_roc = [("LogisticRegression", y_true, lr_prob)]

    if os.path.exists(base_path):
        base_df = pd.read_csv(base_path)
        base_pred = base_df["pred"].values
        base_prob = base_df["prob"].values
        models_preds.append(("ThresholdBaseline", y_true, base_pred))
        models_roc.append(("ThresholdBaseline", y_true, base_prob))

    print("\nLogistic Regression Classification Report:")
    print(classification_report(y_true, lr_pred,
                                target_names=["Memorization", "Reasoning"],
                                zero_division=0))

    print_metrics_table(models_preds)

    plot_confusion_matrix(y_true, lr_pred,
                          "Confusion Matrix: Logistic Regression",
                          "confusion_matrix_lr.png")

    if os.path.exists(base_path):
        plot_confusion_matrix(y_true, base_pred,
                              "Confusion Matrix: Baseline",
                              "confusion_matrix_baseline.png")

    plot_roc_curve(models_roc, "roc_curve.png")

    train_losses_path = os.path.join(RESULTS_DIR, "train_losses.npy")
    val_losses_path = os.path.join(RESULTS_DIR, "val_losses.npy")
    if os.path.exists(train_losses_path):
        train_losses = np.load(train_losses_path)
        val_losses = np.load(val_losses_path) if os.path.exists(val_losses_path) else None
        plot_training_loss(train_losses, val_losses)

    weights_path = os.path.join(RESULTS_DIR, "feature_weights.csv")
    if os.path.exists(weights_path):
        plot_feature_importance(weights_path)

    print("\nAll plots saved to results/")


if __name__ == "__main__":
    main()
