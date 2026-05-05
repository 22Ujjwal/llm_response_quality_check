import numpy as np


class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_epochs=1000, batch_size=32,
                 l2_lambda=0.01, verbose=True, random_state=42):
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.l2_lambda = l2_lambda
        self.verbose = verbose
        self.random_state = random_state
        self.weights = None
        self.bias = None
        self.train_losses = []
        self.val_losses = []

    def _sigmoid(self, z):
        # piecewise sigmoid split at 0, keeps gradients from going cooked
        return np.where(z >= 0,
                        1 / (1 + np.exp(-z)),
                        np.exp(z) / (1 + np.exp(z)))

    def _binary_cross_entropy(self, y_true, y_pred):
        eps = 1e-9
        y_pred = np.clip(y_pred, eps, 1 - eps)
        bce = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        l2_penalty = (self.l2_lambda / 2) * np.sum(self.weights ** 2)
        return bce + l2_penalty

    def _forward(self, X):
        z = X @ self.weights + self.bias
        return self._sigmoid(z)

    def _compute_gradients(self, X, y_true, y_pred):
        n = len(y_true)
        error = y_pred - y_true
        dw = (X.T @ error) / n + self.l2_lambda * self.weights
        db = np.mean(error)
        return dw, db

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        np.random.seed(self.random_state)
        n_samples, n_features = X_train.shape

        self.weights = np.random.randn(n_features) * np.sqrt(2.0 / n_features)
        self.bias = 0.0
        self.train_losses = []
        self.val_losses = []

        for epoch in range(self.num_epochs):
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]

            # mini-batch weight updates, this is literally where the model learns
            for start in range(0, n_samples, self.batch_size):
                end = min(start + self.batch_size, n_samples)
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]
                y_pred_batch = self._forward(X_batch)
                dw, db = self._compute_gradients(X_batch, y_batch, y_pred_batch)
                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db

            train_pred = self._forward(X_train)
            train_loss = self._binary_cross_entropy(y_train, train_pred)
            self.train_losses.append(train_loss)

            if X_val is not None and y_val is not None:
                val_pred = self._forward(X_val)
                val_loss = self._binary_cross_entropy(y_val, val_pred)
                self.val_losses.append(val_loss)

            if self.verbose and (epoch + 1) % 100 == 0:
                val_str = f"  val_loss={val_loss:.4f}" if self.val_losses else ""
                print(f"  Epoch {epoch+1}/{self.num_epochs}  train_loss={train_loss:.4f}{val_str}")

        return self

    def predict_proba(self, X):
        return self._forward(X)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

    def get_params(self):
        return {
            "learning_rate": self.learning_rate,
            "num_epochs": self.num_epochs,
            "batch_size": self.batch_size,
            "l2_lambda": self.l2_lambda,
        }
