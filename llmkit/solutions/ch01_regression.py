"""Chapter 1: Linear Regression

Core concepts:
- Model: y_hat = w * x + b
- Loss function: Mean Squared Error (MSE)
- Gradient descent: Update parameters to minimize loss
- Training loop: forward → compute loss → backward → update
"""

import numpy as np


class LinearRegression:
    """Simple linear regression with gradient descent

    Model: y_hat = w * x + b

    Attributes:
        w: Weight parameter
        b: Bias parameter
    """

    def __init__(self, learning_rate=0.01):
        """Initialize linear regression model

        Args:
            learning_rate: Step size for gradient descent
        """
        self.lr = learning_rate
        self.w = 0.0
        self.b = 0.0

    def forward(self, x):
        """Forward pass: compute predictions

        Args:
            x: Input values (n,)

        Returns:
            y_hat: Predictions (n,)
        """
        return self.w * x + self.b

    def mse_loss(self, y_pred, y_true):
        """Compute Mean Squared Error loss

        Args:
            y_pred: Predicted values (n,)
            y_true: True values (n,)

        Returns:
            loss: Scalar MSE value
        """
        return np.mean((y_pred - y_true) ** 2)

    def compute_gradients(self, x, y_true, y_pred):
        """Compute gradients of loss with respect to parameters

        Derivation:
            loss = mean((y_pred - y_true)^2)
                 = mean((w*x + b - y)^2)

            ∂loss/∂w = mean(2 * (w*x + b - y) * x)
                     = mean(2 * (y_pred - y_true) * x)

            ∂loss/∂b = mean(2 * (w*x + b - y))
                     = mean(2 * (y_pred - y_true))

        Args:
            x: Input values (n,)
            y_true: True values (n,)
            y_pred: Predicted values (n,)

        Returns:
            grad_w: Gradient with respect to w
            grad_b: Gradient with respect to b
        """
        n = len(x)
        error = y_pred - y_true

        grad_w = (2.0 / n) * np.sum(error * x)
        grad_b = (2.0 / n) * np.sum(error)

        return grad_w, grad_b

    def update_parameters(self, grad_w, grad_b):
        """Update parameters using gradient descent

        Args:
            grad_w: Gradient with respect to w
            grad_b: Gradient with respect to b
        """
        self.w -= self.lr * grad_w
        self.b -= self.lr * grad_b

    def train(self, x, y, n_iterations=100):
        """Train the model using gradient descent

        Args:
            x: Input values (n,)
            y: True values (n,)
            n_iterations: Number of training iterations

        Returns:
            Dictionary with training history
        """
        w_history = []
        b_history = []
        loss_history = []

        for i in range(n_iterations):
            # Forward pass
            y_pred = self.forward(x)

            # Compute loss
            loss = self.mse_loss(y_pred, y)

            # Backward pass (compute gradients)
            grad_w, grad_b = self.compute_gradients(x, y, y_pred)

            # Update parameters
            self.update_parameters(grad_w, grad_b)

            # Record history
            w_history.append(self.w)
            b_history.append(self.b)
            loss_history.append(loss)

        return {
            'x': x,
            'y': y,
            'w_history': np.array(w_history),
            'b_history': np.array(b_history),
            'loss_history': np.array(loss_history),
            'final_w': self.w,
            'final_b': self.b
        }


def gradient_descent(x, y, learning_rate=0.01, n_iterations=100):
    """Standalone function for gradient descent on linear regression

    This is an alternative interface to the LinearRegression class.

    Args:
        x: Input values (n,)
        y: True values (n,)
        learning_rate: Step size for gradient descent
        n_iterations: Number of training iterations

    Returns:
        Dictionary with training history
    """
    model = LinearRegression(learning_rate=learning_rate)
    return model.train(x, y, n_iterations)


# ============================================================================
# Helper Functions for Exercises
# ============================================================================

def forward_pass(x, w, b):
    """Compute forward pass

    Args:
        x: Input values (n,)
        w: Weight
        b: Bias

    Returns:
        y_hat: Predictions (n,)
    """
    return w * x + b


def compute_mse(y_pred, y_true):
    """Compute Mean Squared Error

    Args:
        y_pred: Predicted values (n,)
        y_true: True values (n,)

    Returns:
        loss: Scalar MSE value
    """
    return np.mean((y_pred - y_true) ** 2)


def compute_gradients_manual(x, y_true, y_pred):
    """Compute gradients manually

    Args:
        x: Input values (n,)
        y_true: True values (n,)
        y_pred: Predicted values (n,)

    Returns:
        grad_w: Gradient with respect to w
        grad_b: Gradient with respect to b
    """
    n = len(x)
    error = y_pred - y_true
    grad_w = (2.0 / n) * np.sum(error * x)
    grad_b = (2.0 / n) * np.sum(error)
    return grad_w, grad_b
