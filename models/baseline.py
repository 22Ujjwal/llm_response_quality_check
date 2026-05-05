import numpy as np


# intentionally dumb rule-based baseline, sets the floor our LR needs to beat
class ThresholdBaseline:
    def __init__(self, feature_names, correct_idx=None, margin_idx=None, margin_threshold=0.0):
        self.feature_names = list(feature_names)
        self.correct_idx = correct_idx
        self.margin_idx = margin_idx
        self.margin_threshold = margin_threshold

    def fit(self, X, y):
        if self.correct_idx is None and "is_correct" in self.feature_names:
            self.correct_idx = self.feature_names.index("is_correct")
        if self.margin_idx is None and "logit_margin" in self.feature_names:
            self.margin_idx = self.feature_names.index("logit_margin")

        if self.margin_idx is not None:
            margins = X[:, self.margin_idx]
            best_acc, best_thresh = 0, 0
            for t in np.linspace(margins.min(), margins.max(), 50):
                preds = self._threshold_predict(X, t)
                acc = np.mean(preds == y)
                if acc > best_acc:
                    best_acc = acc
                    best_thresh = t
            self.margin_threshold = best_thresh

        return self

    def _threshold_predict(self, X, threshold):
        preds = np.zeros(len(X), dtype=int)
        if self.correct_idx is not None and self.margin_idx is not None:
            correct = X[:, self.correct_idx] > 0
            high_margin = X[:, self.margin_idx] > threshold
            preds = (correct & high_margin).astype(int)
        elif self.correct_idx is not None:
            preds = (X[:, self.correct_idx] > 0).astype(int)
        return preds

    def predict(self, X, threshold=None):
        t = threshold if threshold is not None else self.margin_threshold
        return _threshold_predict(X, t) if threshold else self._threshold_predict(X, self.margin_threshold)

    def predict_proba(self, X):
        if self.margin_idx is not None:
            raw = X[:, self.margin_idx]
            shifted = raw - raw.min()
            denom = shifted.max() if shifted.max() > 0 else 1
            return shifted / denom
        return np.ones(len(X)) * 0.5
