"""Chapter 5: Multi-Layer Perceptron + Backpropagation

Core concepts:
- Hidden layers: Stack layers of neurons
- Activation functions: Non-linearity (ReLU, tanh, sigmoid)
- Why non-linearity matters: Without it, stacking is pointless
- Backpropagation: Chain rule applied systematically
- Universal approximation: MLP can approximate any function
"""

import numpy as np


# ============================================================================
# Activation Functions
# ============================================================================

def relu(x):
    """ReLU activation: max(0, x)

    Args:
        x: Input array

    Returns:
        output: ReLU(x)
    """
    return np.maximum(0, x)


def relu_derivative(x):
    """Derivative of ReLU

    Args:
        x: Input array (pre-activation)

    Returns:
        derivative: 1 if x > 0, else 0
    """
    return (x > 0).astype(float)


def tanh(x):
    """Tanh activation

    Args:
        x: Input array

    Returns:
        output: tanh(x)
    """
    return np.tanh(x)


def tanh_derivative(x):
    """Derivative of tanh

    tanh'(x) = 1 - tanh(x)^2

    Args:
        x: Input array (pre-activation)

    Returns:
        derivative: 1 - tanh(x)^2
    """
    return 1 - np.tanh(x) ** 2


def sigmoid(x):
    """Sigmoid activation: 1 / (1 + exp(-x))

    Args:
        x: Input array

    Returns:
        output: sigmoid(x)
    """
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))  # Clip for numerical stability


def sigmoid_derivative(x):
    """Derivative of sigmoid

    sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))

    Args:
        x: Input array (pre-activation)

    Returns:
        derivative: sigmoid(x) * (1 - sigmoid(x))
    """
    sig = sigmoid(x)
    return sig * (1 - sig)


def softmax(x):
    """Softmax activation (for output layer)

    Args:
        x: Input array (n_classes,) or (batch_size, n_classes)

    Returns:
        probs: Probability distribution
    """
    x_shifted = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


# ============================================================================
# Multi-Layer Perceptron
# ============================================================================

class MLP:
    """Multi-Layer Perceptron for classification

    Architecture: input → hidden layers → output
    Each layer: linear transformation + activation function
    """

    def __init__(self, layer_sizes, activation='relu', learning_rate=0.01):
        """Initialize MLP

        Args:
            layer_sizes: List of layer sizes [input_dim, hidden1, hidden2, ..., output_dim]
            activation: Activation function ('relu', 'tanh', or 'sigmoid')
            learning_rate: Learning rate for gradient descent
        """
        self.layer_sizes = layer_sizes
        self.n_layers = len(layer_sizes) - 1
        self.lr = learning_rate

        # Set activation function
        if activation == 'relu':
            self.activation = relu
            self.activation_derivative = relu_derivative
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_derivative = tanh_derivative
        elif activation == 'sigmoid':
            self.activation = sigmoid
            self.activation_derivative = sigmoid_derivative
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Initialize weights and biases
        self.weights = []
        self.biases = []

        for i in range(self.n_layers):
            # He initialization for ReLU, Xavier for tanh/sigmoid
            if activation == 'relu':
                scale = np.sqrt(2.0 / layer_sizes[i])
            else:
                scale = np.sqrt(1.0 / layer_sizes[i])

            W = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * scale
            b = np.zeros(layer_sizes[i + 1])

            self.weights.append(W)
            self.biases.append(b)

    def forward(self, x):
        """Forward pass through network

        Args:
            x: Input (input_dim,) or (batch_size, input_dim)

        Returns:
            output: Final output (output_dim,) or (batch_size, output_dim)
            cache: Dictionary with intermediate values for backprop
        """
        cache = {'activations': [x], 'pre_activations': []}

        current = x

        # Pass through hidden layers
        for i in range(self.n_layers - 1):
            # Linear transformation
            z = current @ self.weights[i] + self.biases[i]
            cache['pre_activations'].append(z)

            # Activation
            current = self.activation(z)
            cache['activations'].append(current)

        # Output layer (linear transformation only, no activation here)
        z_out = current @ self.weights[-1] + self.biases[-1]
        cache['pre_activations'].append(z_out)

        # Apply softmax for classification
        output = softmax(z_out)
        cache['output'] = output

        return output, cache

    def backward(self, x, y_true, cache):
        """Backward pass (backpropagation)

        Computes gradients for all parameters using chain rule.

        Args:
            x: Input
            y_true: True labels (indices) or one-hot vectors
            cache: Forward pass cache

        Returns:
            grads: Dictionary with gradients for weights and biases
        """
        output = cache['output']
        activations = cache['activations']
        pre_activations = cache['pre_activations']

        # Handle single sample vs batch
        if x.ndim == 1:
            batch_size = 1
            # Convert scalar label to array
            if np.isscalar(y_true):
                y_true = np.array([y_true])
        else:
            batch_size = x.shape[0]

        # Initialize gradients
        grad_weights = [np.zeros_like(W) for W in self.weights]
        grad_biases = [np.zeros_like(b) for b in self.biases]

        # Gradient of loss w.r.t. output (softmax + cross-entropy)
        grad_output = output.copy()

        if batch_size == 1:
            grad_output[y_true[0]] -= 1
        else:
            grad_output[np.arange(batch_size), y_true] -= 1

        # Backpropagate through layers
        grad_current = grad_output

        for i in range(self.n_layers - 1, -1, -1):
            # Gradient w.r.t. weights and biases at this layer
            if batch_size == 1:
                grad_weights[i] = np.outer(activations[i], grad_current)
                grad_biases[i] = grad_current
            else:
                grad_weights[i] = activations[i].T @ grad_current / batch_size
                grad_biases[i] = np.mean(grad_current, axis=0)

            # Backpropagate to previous layer
            if i > 0:
                # Gradient w.r.t. activation of previous layer
                grad_activation = grad_current @ self.weights[i].T

                # Gradient w.r.t. pre-activation (apply activation derivative)
                grad_current = grad_activation * self.activation_derivative(pre_activations[i - 1])

        return {'weights': grad_weights, 'biases': grad_biases}

    def update_parameters(self, grads):
        """Update parameters using gradients

        Args:
            grads: Dictionary with gradients
        """
        for i in range(self.n_layers):
            self.weights[i] -= self.lr * grads['weights'][i]
            self.biases[i] -= self.lr * grads['biases'][i]

    def train_step(self, x, y_true):
        """Single training step

        Args:
            x: Input
            y_true: True labels

        Returns:
            loss: Loss value
            output: Model predictions
        """
        # Forward pass
        output, cache = self.forward(x)

        # Compute loss (cross-entropy)
        if x.ndim == 1:
            loss = -np.log(output[y_true] + 1e-10)
        else:
            batch_size = x.shape[0]
            true_probs = output[np.arange(batch_size), y_true]
            loss = -np.mean(np.log(true_probs + 1e-10))

        # Backward pass
        grads = self.backward(x, y_true, cache)

        # Update parameters
        self.update_parameters(grads)

        return loss, output

    def predict(self, x):
        """Predict class probabilities

        Args:
            x: Input

        Returns:
            probs: Probability distribution over classes
        """
        output, _ = self.forward(x)
        return output

    def predict_class(self, x):
        """Predict class labels

        Args:
            x: Input

        Returns:
            labels: Predicted class indices
        """
        probs = self.predict(x)
        return np.argmax(probs, axis=-1)


# ============================================================================
# Training Function
# ============================================================================

def train_mlp(X_train, y_train, layer_sizes, activation='relu',
              learning_rate=0.01, n_epochs=50, batch_size=32, verbose=True):
    """Train MLP on dataset

    Args:
        X_train: Training inputs (n_samples, input_dim)
        y_train: Training labels (n_samples,)
        layer_sizes: List of layer sizes [input_dim, hidden, ..., output_dim]
        activation: Activation function
        learning_rate: Learning rate
        n_epochs: Number of epochs
        batch_size: Batch size (None for full batch)
        verbose: Print progress

    Returns:
        model: Trained MLP
        loss_history: List of losses
    """
    model = MLP(layer_sizes, activation, learning_rate)
    loss_history = []

    n_samples = len(X_train)

    for epoch in range(n_epochs):
        # Shuffle data
        indices = np.random.permutation(n_samples)
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]

        epoch_losses = []

        # Mini-batch training
        if batch_size is None:
            batch_size = n_samples

        for i in range(0, n_samples, batch_size):
            X_batch = X_shuffled[i:i + batch_size]
            y_batch = y_shuffled[i:i + batch_size]

            loss, _ = model.train_step(X_batch, y_batch)
            epoch_losses.append(loss)

        avg_loss = np.mean(epoch_losses)
        loss_history.append(avg_loss)

        if verbose and (epoch + 1) % max(1, n_epochs // 10) == 0:
            # Compute accuracy
            predictions = model.predict_class(X_train)
            accuracy = np.mean(predictions == y_train)
            print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.3f}")

    return model, loss_history


# ============================================================================
# Helper Functions for Exercises
# ============================================================================

def initialize_weights(input_dim, output_dim, activation='relu'):
    """Initialize weights for a layer

    Args:
        input_dim: Input dimension
        output_dim: Output dimension
        activation: Activation type for scaling

    Returns:
        W: Weight matrix
        b: Bias vector
    """
    if activation == 'relu':
        scale = np.sqrt(2.0 / input_dim)
    else:
        scale = np.sqrt(1.0 / input_dim)

    W = np.random.randn(input_dim, output_dim) * scale
    b = np.zeros(output_dim)

    return W, b


def forward_layer(x, W, b, activation_fn):
    """Forward pass through a single layer

    Args:
        x: Input
        W: Weights
        b: Bias
        activation_fn: Activation function

    Returns:
        z: Pre-activation
        a: Post-activation
    """
    z = x @ W + b
    a = activation_fn(z)
    return z, a
