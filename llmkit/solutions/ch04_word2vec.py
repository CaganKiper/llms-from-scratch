"""Chapter 4: Word2Vec (Dense Embeddings)

Core concepts:
Part A: Autoencoder - Learning compressed representations via bottleneck
Part B: Word2Vec - Learning word embeddings from context
  - CBOW (Continuous Bag of Words): context → center word
  - Skip-gram: center word → context
  - Distributional hypothesis: Similar contexts = similar meanings
"""

import numpy as np


# ============================================================================
# Part A: Simple Autoencoder
# ============================================================================

class SimpleAutoencoder:
    """Simple autoencoder with one hidden layer

    Architecture: input → hidden (bottleneck) → output
    Goal: Reconstruct input, forcing hidden layer to learn compressed representation
    """

    def __init__(self, input_dim, hidden_dim, learning_rate=0.01):
        """Initialize autoencoder

        Args:
            input_dim: Dimension of input/output
            hidden_dim: Dimension of hidden layer (bottleneck)
            learning_rate: Learning rate for training
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lr = learning_rate

        # Initialize weights
        self.W_encode = np.random.randn(input_dim, hidden_dim) * 0.01
        self.b_encode = np.zeros(hidden_dim)

        self.W_decode = np.random.randn(hidden_dim, input_dim) * 0.01
        self.b_decode = np.zeros(input_dim)

    def encode(self, x):
        """Encode input to hidden representation

        Args:
            x: Input (input_dim,)

        Returns:
            h: Hidden representation (hidden_dim,)
        """
        return np.tanh(x @ self.W_encode + self.b_encode)

    def decode(self, h):
        """Decode hidden representation to output

        Args:
            h: Hidden representation (hidden_dim,)

        Returns:
            output: Reconstructed output (input_dim,)
        """
        return h @ self.W_decode + self.b_decode

    def forward(self, x):
        """Full forward pass

        Args:
            x: Input (input_dim,)

        Returns:
            h: Hidden representation
            output: Reconstructed output
        """
        h = self.encode(x)
        output = self.decode(h)
        return h, output

    def reconstruction_loss(self, x, output):
        """Compute reconstruction loss (MSE)

        Args:
            x: Original input
            output: Reconstructed output

        Returns:
            loss: Scalar loss value
        """
        return np.mean((x - output) ** 2)


# ============================================================================
# Part B: Word2Vec (CBOW and Skip-gram)
# ============================================================================

class CBOW:
    """Continuous Bag of Words model

    Predicts center word from context words.
    The embedding matrix is what we learn and use.
    """

    def __init__(self, vocab_size, embed_dim, learning_rate=0.01):
        """Initialize CBOW model

        Args:
            vocab_size: Size of vocabulary
            embed_dim: Dimension of word embeddings
            learning_rate: Learning rate
        """
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.lr = learning_rate

        # Embedding matrix: each row is a word's embedding
        self.W_embed = np.random.randn(vocab_size, embed_dim) * 0.01

        # Output weights: project from embedding back to vocabulary
        self.W_out = np.random.randn(embed_dim, vocab_size) * 0.01

    def get_embedding(self, word_idx):
        """Get embedding for a word

        Args:
            word_idx: Index of word in vocabulary

        Returns:
            embedding: Word embedding (embed_dim,)
        """
        return self.W_embed[word_idx]

    def forward(self, context_indices):
        """Forward pass: predict center word from context

        Args:
            context_indices: List of indices for context words

        Returns:
            context_embedding: Average embedding of context words
            logits: Scores for each word in vocabulary
            probs: Probability distribution over vocabulary
        """
        # Get embeddings for context words
        context_embeddings = self.W_embed[context_indices]

        # Average them (bag of words - order doesn't matter)
        context_embedding = np.mean(context_embeddings, axis=0)

        # Project to vocabulary space
        logits = context_embedding @ self.W_out

        # Softmax to get probabilities
        probs = softmax(logits)

        return context_embedding, logits, probs

    def train_step(self, context_indices, target_idx):
        """Single training step

        Args:
            context_indices: Indices of context words
            target_idx: Index of target (center) word

        Returns:
            loss: Loss value
        """
        # Forward pass
        context_embedding, logits, probs = self.forward(context_indices)

        # Compute loss (cross-entropy)
        loss = -np.log(probs[target_idx] + 1e-10)

        # Backward pass
        # Gradient of cross-entropy + softmax
        grad_logits = probs.copy()
        grad_logits[target_idx] -= 1

        # Gradients for output weights
        grad_W_out = np.outer(context_embedding, grad_logits)

        # Gradient flowing back to context embedding
        grad_context_embedding = grad_logits @ self.W_out.T

        # Distribute gradient to each context word equally
        grad_embed = grad_context_embedding / len(context_indices)

        # Update embeddings for context words
        for idx in context_indices:
            self.W_embed[idx] -= self.lr * grad_embed

        # Update output weights
        self.W_out -= self.lr * grad_W_out

        return loss

    def get_all_embeddings(self):
        """Get embedding matrix

        Returns:
            embeddings: (vocab_size, embed_dim)
        """
        return self.W_embed.copy()


class SkipGram:
    """Skip-gram model

    Predicts context words from center word.
    Similar to CBOW but reversed task.
    """

    def __init__(self, vocab_size, embed_dim, learning_rate=0.01):
        """Initialize Skip-gram model

        Args:
            vocab_size: Size of vocabulary
            embed_dim: Dimension of word embeddings
            learning_rate: Learning rate
        """
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.lr = learning_rate

        # Embedding matrix
        self.W_embed = np.random.randn(vocab_size, embed_dim) * 0.01

        # Output weights
        self.W_out = np.random.randn(embed_dim, vocab_size) * 0.01

    def get_embedding(self, word_idx):
        """Get embedding for a word"""
        return self.W_embed[word_idx]

    def forward(self, center_idx):
        """Forward pass: predict context from center word

        Args:
            center_idx: Index of center word

        Returns:
            center_embedding: Embedding of center word
            logits: Scores for each word in vocabulary
            probs: Probability distribution over vocabulary
        """
        # Get embedding for center word
        center_embedding = self.W_embed[center_idx]

        # Project to vocabulary space
        logits = center_embedding @ self.W_out

        # Softmax
        probs = softmax(logits)

        return center_embedding, logits, probs

    def train_step(self, center_idx, context_indices):
        """Single training step

        Args:
            center_idx: Index of center word
            context_indices: Indices of context words

        Returns:
            loss: Average loss across context words
        """
        # Forward pass
        center_embedding, logits, probs = self.forward(center_idx)

        # Compute loss for each context word
        total_loss = 0
        grad_W_out = np.zeros_like(self.W_out)
        grad_center_embedding = np.zeros_like(center_embedding)

        for context_idx in context_indices:
            # Loss for this context word
            loss = -np.log(probs[context_idx] + 1e-10)
            total_loss += loss

            # Gradient
            grad_logits = probs.copy()
            grad_logits[context_idx] -= 1

            # Accumulate gradients
            grad_W_out += np.outer(center_embedding, grad_logits)
            grad_center_embedding += grad_logits @ self.W_out.T

        # Average gradients
        n_context = len(context_indices)
        grad_W_out /= n_context
        grad_center_embedding /= n_context

        # Update parameters
        self.W_embed[center_idx] -= self.lr * grad_center_embedding
        self.W_out -= self.lr * grad_W_out

        return total_loss / n_context

    def get_all_embeddings(self):
        """Get embedding matrix"""
        return self.W_embed.copy()


# ============================================================================
# Training Functions
# ============================================================================

def train_cbow(training_pairs, vocab_size, embed_dim=50, learning_rate=0.01, n_epochs=5):
    """Train CBOW model

    Args:
        training_pairs: List of (context_indices, target_idx) tuples
        vocab_size: Size of vocabulary
        embed_dim: Embedding dimension
        learning_rate: Learning rate
        n_epochs: Number of training epochs

    Returns:
        model: Trained CBOW model
        loss_history: List of losses
    """
    model = CBOW(vocab_size, embed_dim, learning_rate)
    loss_history = []

    for epoch in range(n_epochs):
        epoch_losses = []

        for context_indices, target_idx in training_pairs:
            loss = model.train_step(context_indices, target_idx)
            epoch_losses.append(loss)

        avg_loss = np.mean(epoch_losses)
        loss_history.append(avg_loss)

        if (epoch + 1) % max(1, n_epochs // 5) == 0:
            print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {avg_loss:.4f}")

    return model, loss_history


def train_skipgram(training_pairs, vocab_size, embed_dim=50, learning_rate=0.01, n_epochs=5):
    """Train Skip-gram model

    Args:
        training_pairs: List of (center_idx, context_indices) tuples
        vocab_size: Size of vocabulary
        embed_dim: Embedding dimension
        learning_rate: Learning rate
        n_epochs: Number of training epochs

    Returns:
        model: Trained Skip-gram model
        loss_history: List of losses
    """
    model = SkipGram(vocab_size, embed_dim, learning_rate)
    loss_history = []

    for epoch in range(n_epochs):
        epoch_losses = []

        for center_idx, context_indices in training_pairs:
            loss = model.train_step(center_idx, context_indices)
            epoch_losses.append(loss)

        avg_loss = np.mean(epoch_losses)
        loss_history.append(avg_loss)

        if (epoch + 1) % max(1, n_epochs // 5) == 0:
            print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {avg_loss:.4f}")

    return model, loss_history


# ============================================================================
# Embedding Analysis Functions
# ============================================================================

def cosine_similarity(vec1, vec2):
    """Compute cosine similarity between two vectors

    Args:
        vec1: First vector
        vec2: Second vector

    Returns:
        similarity: Cosine similarity (-1 to 1)
    """
    dot = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot / (norm1 * norm2 + 1e-10)


def find_similar_words(word_idx, embeddings, vocab, top_k=5):
    """Find most similar words to given word

    Args:
        word_idx: Index of query word
        embeddings: Embedding matrix (vocab_size, embed_dim)
        vocab: List of words
        top_k: Number of similar words to return

    Returns:
        similar_words: List of (word, similarity) tuples
    """
    query_embedding = embeddings[word_idx]

    similarities = []
    for i, word in enumerate(vocab):
        if i == word_idx:
            continue
        sim = cosine_similarity(query_embedding, embeddings[i])
        similarities.append((word, sim))

    # Sort by similarity
    similarities.sort(key=lambda x: x[1], reverse=True)

    return similarities[:top_k]


def analogy(word_a, word_b, word_c, embeddings, word2idx, vocab, top_k=5):
    """Solve word analogy: a is to b as c is to ?

    Uses vector arithmetic: embedding(b) - embedding(a) + embedding(c) ≈ embedding(d)

    Args:
        word_a, word_b, word_c: Words for analogy
        embeddings: Embedding matrix
        word2idx: Word to index mapping
        vocab: List of words
        top_k: Number of candidates to return

    Returns:
        candidates: List of (word, similarity) tuples
    """
    # Get embeddings
    emb_a = embeddings[word2idx[word_a]]
    emb_b = embeddings[word2idx[word_b]]
    emb_c = embeddings[word2idx[word_c]]

    # Compute target vector: b - a + c
    target = emb_b - emb_a + emb_c

    # Find closest words
    similarities = []
    exclude = {word_a, word_b, word_c}

    for word, idx in word2idx.items():
        if word in exclude:
            continue
        sim = cosine_similarity(target, embeddings[idx])
        similarities.append((word, sim))

    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]


# ============================================================================
# Standalone Forward Functions (for exercise verification)
# ============================================================================

def cbow_forward(context_indices, W_embed, W_out):
    """Standalone CBOW forward pass — used to verify exercise implementations.

    Args:
        context_indices: List of indices for context words
        W_embed: Embedding matrix (vocab_size, embed_dim)
        W_out: Output weights (embed_dim, vocab_size)

    Returns:
        context_embedding: Average embedding of context words (embed_dim,)
        logits: Scores for each word in vocabulary (vocab_size,)
        probs: Probability distribution over vocabulary (vocab_size,)
    """
    context_embeddings = W_embed[context_indices]
    context_embedding = np.mean(context_embeddings, axis=0)
    logits = context_embedding @ W_out
    probs = softmax(logits)
    return context_embedding, logits, probs


# ============================================================================
# Helper Functions
# ============================================================================

def softmax(logits):
    """Softmax function (same as ch03)"""
    logits_shifted = logits - np.max(logits)
    exp_logits = np.exp(logits_shifted)
    return exp_logits / np.sum(exp_logits)
