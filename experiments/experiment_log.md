# Experiment Log - Reasoning vs Memorization Classifier

**Task:** Binary classification - predict whether an LLM exhibits high GPQA (knowledge) performance,
using reasoning benchmark scores as features.  
**Dataset:** Open LLM Leaderboard v2 (huggingface.co/datasets/open-llm-leaderboard/results)  
**Algorithm:** Logistic Regression (from scratch, numpy) + Threshold Baseline  
**Error Function:** Binary Cross-Entropy + L2 regularization  
**Evaluation Metrics:** Accuracy, F1, ROC-AUC  

---

## Notes on Session History

Sessions 1–4 used an earlier label definition (`acc_gap > median` where `acc_gap = reasoning_score - knowledge_score`).
This caused **data leakage** (label was a direct function of training features).
Session 5 is the final corrected run: label = `gpqa_acc > median(gpqa_acc)`, and normalization
was moved post-split (fit on train only).

---

## Session 1 - 2026-05-03 18:04 (Preliminary - leaked label)

**Label:** `acc_gap > median` (later found to contain data leakage)  
**Normalization:** Applied before split (leaky)  
**Dataset size:** ~253 models | **Train/Val/Test split:** 72% / 13% / 15%  

### Grid Search - Logistic Regression (Val Set)

| Exp # | Learning Rate | L2 Lambda | Epochs | Batch Size | Val Accuracy | Val F1 | Val ROC-AUC |
|-------|--------------|-----------|--------|------------|-------------|--------|-------------|
| 1     | 0.001        | 0.00      | 500    | 32         | 86.84%      | 0.8649 | 0.9668      |
| 2     | 0.001        | 0.01      | 500    | 32         | 86.84%      | 0.8649 | 0.9668      |
| 3     | 0.001        | 0.10      | 500    | 32         | 86.84%      | 0.8649 | 0.9778      |
| 4     | 0.010        | 0.00      | 500    | 32         | 92.11%      | 0.9189 | 0.9861      |
| 5     | 0.010        | 0.01      | 500    | 32         | 92.11%      | 0.9189 | 0.9834      |
| 6     | 0.010        | 0.10      | 500    | 32         | 86.84%      | 0.8571 | 0.9723      |
| 7     | 0.100        | 0.00      | 500    | 32         | 97.37%      | 0.9730 | 1.0000      |
| 8     | 0.100        | 0.01      | 500    | 32         | 94.74%      | 0.9474 | 0.9917      |
| 9     | 0.100        | 0.10      | 500    | 32         | 86.84%      | 0.8571 | 0.9723      |

**Best val params:** lr=0.1, l2=0.0 (Exp 7)

### Best Model - Test Set

| Model              | Test Accuracy | Test F1 | Test ROC-AUC |
|--------------------|--------------|---------|--------------|
| LogisticRegression | 88.89%       | 0.8980  | 1.0000       |
| ThresholdBaseline  | 51.11%       | 0.0000  | 0.5000       |

---

## Session 2 - 2026-05-03 19:17 (Preliminary - leaked label)

**Label:** `acc_gap > median` (leaky)  
**Normalization:** Applied before split (leaky)  
**Dataset size:** ~253 models | **Train/Val/Test split:** 72% / 13% / 15%  

### Grid Search - Logistic Regression (Val Set)

| Exp # | Learning Rate | L2 Lambda | Epochs | Batch Size | Val Accuracy | Val F1 | Val ROC-AUC |
|-------|--------------|-----------|--------|------------|-------------|--------|-------------|
| 10    | 0.001        | 0.00      | 500    | 32         | 63.16%      | 0.5333 | 0.8116      |
| 11    | 0.001        | 0.01      | 500    | 32         | 68.42%      | 0.6250 | 0.8199      |
| 12    | 0.001        | 0.10      | 500    | 32         | 71.05%      | 0.6667 | 0.8560      |
| 13    | 0.010        | 0.00      | 500    | 32         | 86.84%      | 0.8571 | 0.9391      |
| 14    | 0.010        | 0.01      | 500    | 32         | 86.84%      | 0.8571 | 0.9363      |
| 15    | 0.010        | 0.10      | 500    | 32         | 86.84%      | 0.8571 | 0.8975      |
| 16    | 0.100        | 0.00      | 500    | 32         | 86.84%      | 0.8571 | 0.9723      |
| 17    | 0.100        | 0.01      | 500    | 32         | 86.84%      | 0.8571 | 0.9446      |
| 18    | 0.100        | 0.10      | 500    | 32         | 86.84%      | 0.8571 | 0.8947      |

**Best val params:** lr=0.01, l2=0.0 (Exp 13, tied - first found wins)

### Best Model - Test Set

| Model              | Test Accuracy | Test F1 | Test ROC-AUC |
|--------------------|--------------|---------|--------------|
| LogisticRegression | 84.44%       | 0.8372  | 0.9091       |
| ThresholdBaseline  | 51.11%       | 0.0000  | 0.5000       |

---

## Session 3 - 2026-05-03 19:17 (Preliminary - leaked label)

**Label:** `acc_gap > median` (leaky)  
**Normalization:** Applied before split (leaky)  
**Dataset size:** ~253 models | **Train/Val/Test split:** 72% / 13% / 15%  

### Grid Search - Logistic Regression (Val Set)

| Exp # | Learning Rate | L2 Lambda | Epochs | Batch Size | Val Accuracy | Val F1 | Val ROC-AUC |
|-------|--------------|-----------|--------|------------|-------------|--------|-------------|
| 19    | 0.001        | 0.00      | 500    | 32         | 78.95%      | 0.7778 | 0.7922      |
| 20    | 0.001        | 0.01      | 500    | 32         | 76.32%      | 0.7429 | 0.7978      |
| 21    | 0.001        | 0.10      | 500    | 32         | 76.32%      | 0.7429 | 0.8449      |
| 22    | 0.010        | 0.00      | 500    | 32         | 86.84%      | 0.8571 | 0.9391      |
| 23    | 0.010        | 0.01      | 500    | 32         | 86.84%      | 0.8571 | 0.9363      |
| 24    | 0.010        | 0.10      | 500    | 32         | 86.84%      | 0.8571 | 0.8947      |
| 25    | 0.100        | 0.00      | 500    | 32         | 86.84%      | 0.8571 | 0.9723      |
| 26    | 0.100        | 0.01      | 500    | 32         | 86.84%      | 0.8571 | 0.9418      |
| 27    | 0.100        | 0.10      | 500    | 32         | 86.84%      | 0.8571 | 0.8947      |

**Best val params:** lr=0.01, l2=0.0 (Exp 22, tied - first found wins)

### Best Model - Test Set

| Model              | Test Accuracy | Test F1 | Test ROC-AUC |
|--------------------|--------------|---------|--------------|
| LogisticRegression | 82.22%       | 0.8182  | 0.9032       |
| ThresholdBaseline  | 51.11%       | 0.0000  | 0.5000       |

---

## Session 4 - 2026-05-03 19:28 (Preliminary - leaked label)

**Label:** `acc_gap > median` (leaky)  
**Normalization:** Applied before split (leaky)  
**Dataset size:** ~253 models | **Train/Val/Test split:** 72% / 13% / 15%  

### Grid Search - Logistic Regression (Val Set)

| Exp # | Learning Rate | L2 Lambda | Epochs | Batch Size | Val Accuracy | Val F1 | Val ROC-AUC |
|-------|--------------|-----------|--------|------------|-------------|--------|-------------|
| 28    | 0.001        | 0.00      | 500    | 32         | 82.68%      | 0.8197 | 0.9169      |
| 29    | 0.001        | 0.01      | 500    | 32         | 81.89%      | 0.8130 | 0.9177      |
| 30    | 0.001        | 0.10      | 500    | 32         | 85.04%      | 0.8430 | 0.9132      |
| 31    | 0.010        | 0.00      | 500    | 32         | 88.98%      | 0.8906 | 0.9740      |
| 32    | 0.010        | 0.01      | 500    | 32         | 88.19%      | 0.8819 | 0.9683      |
| 33    | 0.010        | 0.10      | 500    | 32         | 85.04%      | 0.8403 | 0.9291      |
| 34    | 0.100        | 0.00      | 500    | 32         | 96.85%      | 0.9677 | 0.9983      |
| 35    | 0.100        | 0.01      | 500    | 32         | 89.76%      | 0.8976 | 0.9745      |
| 36    | 0.100        | 0.10      | 500    | 32         | 84.25%      | 0.8305 | 0.9296      |

**Best val params:** lr=0.1, l2=0.0 (Exp 34)

### Best Model - Test Set

| Model              | Test Accuracy | Test F1 | Test ROC-AUC |
|--------------------|--------------|---------|--------------|
| LogisticRegression | 95.33%       | 0.9530  | 0.9948       |
| ThresholdBaseline  | 50.00%       | 0.0000  | 0.5000       |

---

## Session 5 - 2026-05-04 22:40 ★ FINAL (Leakage Fixed)

**Changes from prior sessions:**
- Label redefined: `gpqa_acc > median(gpqa_acc)` - held out from features entirely
- `gpqa_acc` removed from feature set
- Normalization moved post-split: mean/std fit on train only, applied to val/test

**Label:** High GPQA knowledge performance (binary, median split)  
**Features (15):** `bbh_acc`, `math_acc`, `musr_acc`, `mmlu_pro_acc`,
10 BBH subtask scores, `score_variance`  
**Dataset size:** 995 models | **Label balance:** 498 negative / 497 positive  
**Train/Val/Test split:** 72% / 13% / 15% → Train=718, Val=127, Test=150  

### Grid Search - Logistic Regression (Val Set)

| Exp # | Learning Rate | L2 Lambda | Epochs | Batch Size | Val Accuracy | Val F1 | Val ROC-AUC |
|-------|--------------|-----------|--------|------------|-------------|--------|-------------|
| 37    | 0.001        | 0.00      | 500    | 32         | 76.38%      | 0.7541 | 0.8676      |
| 38    | 0.001        | 0.01      | 500    | 32         | 77.17%      | 0.7603 | 0.8708      |
| 39    | 0.001        | 0.10      | 500    | 32         | 80.31%      | 0.7899 | 0.8795      |
| 40    | 0.010        | 0.00      | 500    | 32         | 87.40%      | 0.8667 | 0.9449      |
| 41    | 0.010        | 0.01      | 500    | 32         | 88.19%      | 0.8760 | 0.9382      |
| 42    | 0.010        | 0.10      | 500    | 32         | 81.89%      | 0.8034 | 0.9023      |
| 43    | 0.100        | 0.00      | 500    | 32         | **90.55%**  | **0.9016** | **0.9683** |
| 44    | 0.100        | 0.01      | 500    | 32         | 87.40%      | 0.8667 | 0.9422      |
| 45    | 0.100        | 0.10      | 500    | 32         | 81.10%      | 0.7966 | 0.9062      |

**Best val params:** lr=0.1, l2=0.0 (Exp 43)

### Best Model - Test Set

| Model              | Test Accuracy | Test F1 | Test Precision | Test Recall | Test ROC-AUC |
|--------------------|--------------|---------|----------------|-------------|--------------|
| LogisticRegression | **91.33%**   | **0.9091** | 0.9559      | 0.8667      | **0.9712**   |
| ThresholdBaseline  | 50.00%       | 0.0000  | 0.0000         | 0.0000      | 0.5000       |

### Top Feature Weights (Best Model)

| Feature         | Weight  | Direction             |
|-----------------|--------:|----------------------|
| math_acc        | +8.446  | → high GPQA          |
| bbh_acc         | +2.354  | → high GPQA          |
| musr_acc        | +1.862  | → high GPQA          |
| score_variance  | +1.090  | → high GPQA          |
| mmlu_pro_acc    | −9.504  | → low GPQA           |

---

## Summary - Final Session vs Baseline

| Model              | Accuracy | F1     | ROC-AUC | Notes                          |
|--------------------|----------|--------|---------|--------------------------------|
| LogisticRegression | 91.33%   | 0.9091 | 0.9712  | Best: lr=0.1, l2=0.0, ep=500  |
| ThresholdBaseline  | 50.00%   | 0.0000 | 0.5000  | Degenerate - predicts all 0   |

**Key finding:** Strong reasoning benchmark scores (especially MATH-Hard) predict high GPQA
performance. MMLU-Pro is a strong negative predictor - models specializing in broad knowledge
without deep reasoning tend to score lower on GPQA.
