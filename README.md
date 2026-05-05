# Reasoning vs Memorization Classifier
## by Ujjwal Gupta 🫡

## Report

Full project report (PDF): [Ujjwal_Gupta_ML_report.pdf](Ujjwal_Gupta_ML_report.pdf)

---

LLMs get benchmarked constantly, but high scores don't always mean the model is actually reasoning.
This project asks a real question: can a model's reasoning benchmark performance predict how well
it does on GPQA (a graduate-level knowledge benchmark)?

We build a binary classifier to answer exactly that. No data leakage, no shortcuts.

**Dataset:** ~995 models from Open LLM Leaderboard v2
**Label:** `gpqa_acc > median(gpqa_acc)` - held out from features entirely
**Features:** BBH subtask scores, MATH-Hard, MUSR, MMLU-Pro, score variance (15 total)

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Run (in order)

### Step 1 - Download leaderboard results

Downloads individual model JSON files from HuggingFace. No full repo snapshot, disk-safe.

```bash
python data/load_dataset.py
```

### Step 2 - Extract features

Drops leakage-prone columns, saves clean feature matrix. Normalization happens later in train.py
(fit on train only), so this step outputs raw values.

```bash
python data/feature_extraction.py
```

### Step 3 - Train and run experiments

Grid search over learning rate and L2 lambda. Best params picked on val F1, evaluated on
held-out test set. All runs logged to `experiments.log`.

```bash
python train.py
```

### Step 4 - Generate evaluation plots

Confusion matrix, ROC curve, training loss curve, feature importance chart.

```bash
python evaluate.py
```

---

## Approach

Two models built and compared:

### Logistic Regression (from scratch)
- Pure numpy, no sklearn used for training
- Mini-batch SGD with L2 regularization
- He initialization for weights
- Numerically stable sigmoid (piecewise at z=0)
- Grid search: lr in {0.001, 0.01, 0.1} x l2 in {0.0, 0.01, 0.1} = 9 combos
- Normalization: mean/std fit on train split only, applied to val/test

### Threshold Baseline
- Rule-based, zero ML
- Scans 50 threshold values on train set to find best logit margin cutoff
- Intentionally dumb - exists to set the performance floor

---

## Outputs

| File | What it is |
|------|-----------|
| `data/raw_predictions.csv` | Raw benchmark scores per model |
| `data/features.csv` | Clean feature matrix with labels |
| `experiments.log` | JSON log, one entry per training run |
| `experiment_log.md` | Formatted experiment report for submission |
| `results/logistic_regression_predictions.csv` | Test set predictions |
| `results/baseline_predictions.csv` | Baseline test set predictions |
| `results/feature_weights.csv` | Learned LR weights per feature |
| `results/confusion_matrix_lr.png` | Confusion matrix (LR) |
| `results/confusion_matrix_baseline.png` | Confusion matrix (baseline) |
| `results/roc_curve.png` | ROC curve comparing both models |
| `results/training_loss.png` | Train/val loss curve |
| `results/feature_importance.png` | Feature weight bar chart |

---

## Results (final run)

| Model | Accuracy | F1 | ROC-AUC |
|-------|----------|----|----|
| Logistic Regression | 91.33% | 0.9091 | 0.9712 |
| Threshold Baseline | 50.00% | 0.0000 | 0.5000 |

Best params: `lr=0.1, l2=0.0, epochs=500, batch_size=32`

Key finding: MATH-Hard score is the strongest positive predictor of GPQA performance.
MMLU-Pro is the strongest negative predictor - models that lean heavy on broad knowledge
without deep reasoning tend to score lower on GPQA.

---

## Dataset

Open LLM Leaderboard v2: https://huggingface.co/datasets/open-llm-leaderboard/results
