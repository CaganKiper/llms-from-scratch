"""Chapter 3: Logistic Regression (Softmax Regression)

Core concepts:
- Softmax function: Converts logits to probability distribution
- Cross-entropy loss: Measures distance between predicted and true distributions
- Next-word prediction: Classification over vocabulary
- The LLM skeleton: input → linear model → softmax → probability
"""

import numpy as np


def softmax(logits):
    """Apply softmax function to convert logits to probabilities

    Softmax(x_i) = exp(x_i) / sum(exp(x_j))

    Implementation uses numerical stability trick:
    Softmax(x) = Softmax(x - max(x))

    Args:
        logits: Raw scores (n_classes,) or (batch_size, n_classes)

    Returns:
        probs: Probability distribution (same shape as logits)
    """
    # Subtract max for numerical stability
    logits_shifted = logits - np.max(logits, axis=-1, keepdims=True)

    # Compute exp
    exp_logits = np.exp(logits_shifted)

    # Normalize
    probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

    return probs


def cross_entropy_loss(probs, target_idx):
    """Compute cross-entropy loss

    Loss = -log(p[target_idx])

    Args:
        probs: Predicted probability distribution (n_classes,) or (batch_size, n_classes)
        target_idx: Index of true class (scalar or (batch_size,))

    Returns:
        loss: Scalar loss value
    """
    # Handle single sample vs batch
    if probs.ndim == 1:
        # Single sample
        return -np.log(probs[target_idx] + 1e-10)  # Add epsilon to avoid log(0)
    else:
        # Batch
        batch_size = probs.shape[0]
        # Extract probability of true class for each sample
        true_class_probs = probs[np.arange(batch_size), target_idx]
        return -np.mean(np.log(true_class_probs + 1e-10))


def cross_entropy_loss_with_logits(logits, target_idx):
    """Compute cross-entropy loss directly from logits

    More numerically stable than computing softmax then loss.

    Args:
        logits: Raw scores (n_classes,) or (batch_size, n_classes)
        target_idx: Index of true class (scalar or (batch_size,))

    Returns:
        loss: Scalar loss value
    """
    probs = softmax(logits)
    return cross_entropy_loss(probs, target_idx)


class LogisticRegression:
    """Logistic (Softmax) Regression for next-word prediction

    Model: logits = X @ W + b
           probs = softmax(logits)

    where X is the input (one-hot encoded word),
    W is the weight matrix, and b is the bias vector.
    """

    def __init__(self, vocab_size, learning_rate=0.01):
        """Initialize logistic regression model

        Args:
            vocab_size: Size of vocabulary
            learning_rate: Step size for gradient descent
        """
        self.vocab_size = vocab_size
        self.lr = learning_rate

        # Initialize parameters (small random values)
        self.W = np.random.randn(vocab_size, vocab_size) * 0.01
        self.b = np.zeros(vocab_size)

    def forward(self, x):
        """Forward pass: compute logits and probabilities

        Args:
            x: Input one-hot vector (vocab_size,) or batch (batch_size, vocab_size)

        Returns:
            logits: Raw scores (vocab_size,) or (batch_size, vocab_size)
            probs: Probability distribution (same shape as logits)
        """
        logits = x @ self.W + self.b
        probs = softmax(logits)
        return logits, probs

    def compute_loss(self, probs, target_idx):
        """Compute cross-entropy loss

        Args:
            probs: Predicted probabilities (vocab_size,) or (batch_size, vocab_size)
            target_idx: True class index (scalar or (batch_size,))

        Returns:
            loss: Scalar loss value
        """
        return cross_entropy_loss(probs, target_idx)

    def backward(self, x, probs, target_idx):
        """Compute gradients via backpropagation

        Derivation:
            Loss = -log(probs[target_idx])

            For softmax + cross-entropy, the gradient is simple:
            ∂Loss/∂logits = probs
            ∂Loss/∂logits[target_idx] -= 1

            Then chain rule:
            ∂Loss/∂W = x^T @ ∂Loss/∂logits
            ∂Loss/∂b = ∂Loss/∂logits

        Args:
            x: Input (vocab_size,) or (batch_size, vocab_size)
            probs: Predicted probabilities (vocab_size,) or (batch_size, vocab_size)
            target_idx: True class index (scalar or (batch_size,))

        Returns:
            grad_W: Gradient w.r.t. W
            grad_b: Gradient w.r.t. b
        """
        # Handle single sample vs batch
        if x.ndim == 1:
            # Single sample
            grad_logits = probs.copy()
            grad_logits[target_idx] -= 1

            grad_W = np.outer(x, grad_logits)
            grad_b = grad_logits
        else:
            # Batch
            batch_size = x.shape[0]
            grad_logits = probs.copy()
            grad_logits[np.arange(batch_size), target_idx] -= 1

            grad_W = (x.T @ grad_logits) / batch_size
            grad_b = np.mean(grad_logits, axis=0)

        return grad_W, grad_b

    def update_parameters(self, grad_W, grad_b):
        """Update parameters using gradient descent

        Args:
            grad_W: Gradient w.r.t. W
            grad_b: Gradient w.r.t. b
        """
        self.W -= self.lr * grad_W
        self.b -= self.lr * grad_b

    def train_step(self, x, target_idx):
        """Single training step

        Args:
            x: Input (vocab_size,) or (batch_size, vocab_size)
            target_idx: True class index (scalar or (batch_size,))

        Returns:
            loss: Loss value for this step
        """
        # Forward pass
        logits, probs = self.forward(x)

        # Compute loss
        loss = self.compute_loss(probs, target_idx)

        # Backward pass
        grad_W, grad_b = self.backward(x, probs, target_idx)

        # Update parameters
        self.update_parameters(grad_W, grad_b)

        return loss

    def predict(self, x):
        """Predict next word probabilities

        Args:
            x: Input one-hot vector (vocab_size,) or batch (batch_size, vocab_size)

        Returns:
            probs: Probability distribution over vocabulary
        """
        _, probs = self.forward(x)
        return probs

    def predict_word(self, x):
        """Predict most likely next word index

        Args:
            x: Input one-hot vector (vocab_size,) or batch (batch_size, vocab_size)

        Returns:
            predicted_idx: Index of most likely word
        """
        probs = self.predict(x)
        return np.argmax(probs, axis=-1)


def train_logistic_regression(X_train, y_train, vocab_size, learning_rate=0.01, n_epochs=10):
    """Train logistic regression model

    Args:
        X_train: Training inputs (n_samples, vocab_size) - one-hot encoded
        y_train: Training targets (n_samples,) - word indices
        vocab_size: Size of vocabulary
        learning_rate: Learning rate
        n_epochs: Number of training epochs

    Returns:
        model: Trained LogisticRegression instance
        loss_history: List of losses over training
    """
    model = LogisticRegression(vocab_size, learning_rate)
    loss_history = []

    n_samples = len(X_train)

    for epoch in range(n_epochs):
        epoch_losses = []

        # Train on each sample
        for i in range(n_samples):
            loss = model.train_step(X_train[i], y_train[i])
            epoch_losses.append(loss)

        # Record average loss for epoch
        avg_loss = np.mean(epoch_losses)
        loss_history.append(avg_loss)

        if (epoch + 1) % max(1, n_epochs // 10) == 0:
            print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {avg_loss:.4f}")

    return model, loss_history


# ============================================================================
# Helper Functions for Exercises
# ============================================================================

def softmax_simple(logits):
    """Simple softmax implementation (exercise helper)"""
    exp_logits = np.exp(logits - np.max(logits))
    return exp_logits / np.sum(exp_logits)


def cross_entropy_simple(probs, target_idx):
    """Simple cross-entropy implementation (exercise helper)"""
    return -np.log(probs[target_idx] + 1e-10)
