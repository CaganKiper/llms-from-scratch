"""Visualization functions for LLM textbook

All matplotlib code lives here. Each visualization is a single function call
from the notebook.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import warnings


# ============================================================================
# General Utilities
# ============================================================================

def compare(your_result, reference_result, tolerance=1e-5):
    """Compare your implementation with reference implementation

    Args:
        your_result: Output from student implementation
        reference_result: Output from reference implementation
        tolerance: Numerical tolerance for floating point comparison
    """
    # Check if student hasn't implemented yet (ellipsis placeholder)
    if your_result is Ellipsis or your_result is ... or \
       (isinstance(your_result, np.ndarray) and your_result.dtype == object and
        len(your_result) > 0 and your_result.flat[0] is Ellipsis):
        print("[NOT IMPLEMENTED] Found ... placeholder")
        return False

    # Handle different types
    if isinstance(your_result, dict) and isinstance(reference_result, dict):
        # Compare dictionaries
        if set(your_result.keys()) != set(reference_result.keys()):
            print("[FAIL] Keys don't match!")
            print(f"   Your keys: {set(your_result.keys())}")
            print(f"   Reference keys: {set(reference_result.keys())}")
            return False

        all_match = True
        for key in your_result.keys():
            if not compare(your_result[key], reference_result[key], tolerance):
                print(f"   Mismatch in key: {key}")
                all_match = False

        if all_match:
            print("[PASS] All results match!")
        return all_match

    elif isinstance(your_result, (list, tuple)):
        # Compare sequences
        if len(your_result) != len(reference_result):
            print(f"[FAIL] Length mismatch: {len(your_result)} vs {len(reference_result)}")
            return False

        all_match = True
        for i, (y, r) in enumerate(zip(your_result, reference_result)):
            if not compare(y, r, tolerance):
                print(f"   Mismatch at index {i}")
                all_match = False

        if all_match:
            print("[PASS] All results match!")
        return all_match

    elif isinstance(your_result, np.ndarray):
        # Compare numpy arrays
        if your_result.shape != reference_result.shape:
            print(f"[FAIL] Shape mismatch: {your_result.shape} vs {reference_result.shape}")
            return False

        if not np.allclose(your_result, reference_result, rtol=tolerance, atol=tolerance):
            print(f"[FAIL] Values don't match!")
            print(f"   Max difference: {np.max(np.abs(your_result - reference_result))}")
            print(f"   Your result (first 5): {your_result.flatten()[:5]}")
            print(f"   Reference (first 5): {reference_result.flatten()[:5]}")
            return False

        print("[PASS] Results match!")
        return True

    else:
        # Compare scalars or other types
        try:
            if isinstance(your_result, (int, float, np.number)):
                match = abs(your_result - reference_result) < tolerance
            else:
                match = your_result == reference_result

            if match:
                print("[PASS] Results match!")
            else:
                print(f"[FAIL] Mismatch: {your_result} vs {reference_result}")
            return match
        except:
            print(f"[FAIL] Cannot compare types: {type(your_result)} vs {type(reference_result)}")
            return False


def _has_ellipsis(value):
    """Recursively check if a value is or contains an unimplemented ... placeholder."""
    if value is Ellipsis:
        return True
    if isinstance(value, dict):
        return any(_has_ellipsis(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return any(_has_ellipsis(v) for v in value)
    if isinstance(value, np.ndarray) and value.dtype == object and value.size > 0:
        return any(v is Ellipsis for v in value.flat)
    return False


def resolve(your_result, reference):
    """Return reference if your_result contains unimplemented ... placeholders.

    Use this in verify cells to ensure downstream code always has a valid value,
    even when the exercise hasn't been implemented yet.
    """
    if _has_ellipsis(your_result):
        print("[SKIPPED] Using reference implementation")
        return reference
    return your_result


# ============================================================================
# Chapter 1: Linear Regression
# ============================================================================

def plot_gradient_descent(results, figsize=(16, 4), true_w=None, true_b=None):
    """Plot gradient descent results

    Args:
        results: Dictionary with keys 'x', 'y', 'w_history', 'b_history', 'loss_history'
        figsize: Figure size
        true_w: True slope (if known), shown as dashed reference line
        true_b: True intercept (if known), shown as dashed reference line
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Plot 1: Loss curve
    axes[0].plot(results['loss_history'], linewidth=2)
    axes[0].set_xlabel('Iteration', fontsize=12)
    axes[0].set_ylabel('Loss (MSE)', fontsize=12)
    axes[0].set_title('Training Loss Over Time', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Final fit
    x = results['x']
    y = results['y']
    w_final = results['w_history'][-1]
    b_final = results['b_history'][-1]
    y_pred = w_final * x + b_final

    axes[1].scatter(x, y, alpha=0.5, s=30, label='Data')
    axes[1].plot(x, y_pred, 'r-', linewidth=2, label=f'Fit: y = {w_final:.2f}x + {b_final:.2f}')
    if true_w is not None and true_b is not None:
        y_true_line = true_w * x + true_b
        axes[1].plot(x, y_true_line, '--', color='gray', linewidth=1.5,
                     alpha=0.8, label=f'True: y = {true_w:.2f}x + {true_b:.2f}')
    axes[1].set_xlabel('x', fontsize=12)
    axes[1].set_ylabel('y', fontsize=12)
    axes[1].set_title('Final Linear Fit', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Parameter trajectory
    w_hist = results['w_history']
    b_hist = results['b_history']
    n_iter = len(w_hist)
    sc = axes[2].scatter(w_hist, b_hist, c=range(n_iter), cmap='viridis', s=20, zorder=3)
    axes[2].plot(w_hist, b_hist, '-', alpha=0.3, linewidth=1, zorder=2)
    axes[2].scatter(w_hist[0], b_hist[0], marker='^', s=120, color='red', zorder=4, label='Start')
    axes[2].scatter(w_hist[-1], b_hist[-1], marker='*', s=200, color='gold', zorder=4, label='End')
    if true_w is not None and true_b is not None:
        axes[2].scatter(true_w, true_b, marker='x', s=150, color='white', linewidths=2,
                        zorder=5, label='True')
    plt.colorbar(sc, ax=axes[2], label='Iteration')
    axes[2].set_xlabel('w (slope)', fontsize=12)
    axes[2].set_ylabel('b (intercept)', fontsize=12)
    axes[2].set_title('Parameter Trajectory', fontsize=14, fontweight='bold')
    axes[2].legend(fontsize=9)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_loss_curve(losses, figsize=(8, 4), xlabel='Epoch'):
    """Plot training loss curve

    Args:
        losses: Array or list of loss values over iterations
        figsize: Figure size
        xlabel: Label for the x-axis (default: 'Epoch')
    """
    plt.figure(figsize=figsize)
    plt.plot(losses, linewidth=2)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training Loss', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ============================================================================
# Chapter 2: One-Hot Encoding
# ============================================================================

def plot_onehot_heatmap(vectors, words, figsize=(10, 6)):
    """Plot heatmap of one-hot vectors

    Args:
        vectors: One-hot encoded vectors (n_words, vocab_size)
        words: List of words corresponding to each vector
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    plt.imshow(vectors, cmap='Blues', aspect='auto')
    plt.colorbar(label='Value')
    plt.xlabel('Vocabulary Index', fontsize=12)
    plt.ylabel('Word', fontsize=12)
    plt.yticks(range(len(words)), words)
    plt.title('One-Hot Encoding', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_distance_matrix(vectors, words, figsize=(8, 8)):
    """Plot pairwise distance matrix between vectors

    Args:
        vectors: Encoded vectors (n_words, dim)
        words: List of words
        figsize: Figure size
    """
    # Compute pairwise Euclidean distances
    n = len(vectors)
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            distances[i, j] = np.linalg.norm(vectors[i] - vectors[j])

    plt.figure(figsize=figsize)
    im = plt.imshow(distances, cmap='viridis', aspect='auto')
    plt.colorbar(im, label='Euclidean Distance  (off-diagonal = √2 ≈ 1.414)')
    plt.xlabel('Word Index', fontsize=12)
    plt.ylabel('Word Index', fontsize=12)
    plt.xticks(range(len(words)), words, rotation=45, ha='right')
    plt.yticks(range(len(words)), words)
    plt.title('Pairwise Distance Matrix\n(All words equidistant — every off-diagonal cell = √2)',
              fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()


# ============================================================================
# Chapter 3: Logistic Regression
# ============================================================================

def plot_probability_dist(probs, vocab, figsize=(10, 4)):
    """Plot probability distribution over vocabulary

    Args:
        probs: Probability distribution (vocab_size,)
        vocab: List of words in vocabulary
        figsize: Figure size
    """
    # Show top 20 words if vocabulary is large
    n_show = min(20, len(vocab))
    top_indices = np.argsort(probs)[-n_show:][::-1]

    plt.figure(figsize=figsize)
    plt.bar(range(n_show), probs[top_indices])
    plt.xlabel('Word', fontsize=12)
    plt.ylabel('Probability', fontsize=12)
    plt.xticks(range(n_show), [vocab[i] for i in top_indices], rotation=45, ha='right')
    plt.title('Next Word Probability Distribution', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()


# ============================================================================
# Chapter 4: Word2Vec
# ============================================================================

def plot_embeddings_2d(embeddings, words, method='pca', figsize=(10, 8),
                       word_indices=None, n_words=50):
    """Plot word embeddings in 2D space

    Args:
        embeddings: Word embedding matrix (vocab_size, embed_dim)
        words: List of words
        method: Dimensionality reduction method ('pca' or 'tsne')
        figsize: Figure size
        word_indices: Explicit list of indices to plot and label. If None,
                      selects n_words most frequent non-punctuation tokens.
        n_words: Number of words to show when word_indices is None
    """
    # Determine which words to annotate
    if word_indices is None:
        # Pick content words: alphabetic, length > 1, from the front of vocab
        content_indices = [i for i, w in enumerate(words) if w.isalpha() and len(w) > 1]
        word_indices = content_indices[:n_words]

    # Reduce to 2D
    if embeddings.shape[1] > 2:
        centered = embeddings - np.mean(embeddings, axis=0)
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, idx]
        embeddings_2d = centered @ eigenvectors[:, :2]
    else:
        embeddings_2d = embeddings

    # Select subset
    subset_2d = embeddings_2d[word_indices]
    subset_words = [words[i] for i in word_indices]

    # Plot
    plt.figure(figsize=figsize)
    plt.scatter(subset_2d[:, 0], subset_2d[:, 1], alpha=0.7, s=60)

    for i, word in enumerate(subset_words):
        plt.annotate(word, (subset_2d[i, 0], subset_2d[i, 1]),
                     fontsize=9, alpha=0.85,
                     xytext=(4, 4), textcoords='offset points')

    plt.xlabel('Dimension 1', fontsize=12)
    plt.ylabel('Dimension 2', fontsize=12)
    plt.title(f'Word Embeddings ({method.upper()}) — {len(word_indices)} content words',
              fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_word_analogies(embeddings, vocab, analogies, figsize=(12, 4)):
    """Visualize word analogies

    Args:
        embeddings: Word embedding matrix
        vocab: List of words
        analogies: List of (word_a, word_b, word_c, word_d) tuples
                  representing "a is to b as c is to d"
        figsize: Figure size per analogy
    """
    word2idx = {word: idx for idx, word in enumerate(vocab)}

    for a, b, c, d in analogies:
        if not all(w in word2idx for w in [a, b, c, d]):
            print(f"Skipping {a}:{b}::{c}:{d} - not all words in vocabulary")
            continue

        # Get embeddings
        vec_a = embeddings[word2idx[a]]
        vec_b = embeddings[word2idx[b]]
        vec_c = embeddings[word2idx[c]]
        vec_d = embeddings[word2idx[d]]

        # Compute analogy: b - a + c should ≈ d
        predicted = vec_b - vec_a + vec_c

        # Compute similarity
        similarity = np.dot(predicted, vec_d) / (np.linalg.norm(predicted) * np.linalg.norm(vec_d))

        print(f"{a}:{b} :: {c}:{d}")
        print(f"  Cosine similarity between predicted and {d}: {similarity:.3f}")


def plot_similarity_heatmap(embeddings, words, figsize=(10, 10), words_to_show=None):
    """Plot cosine similarity heatmap

    Args:
        embeddings: Word embedding matrix (vocab_size, embed_dim)
        words: List of all words in vocabulary
        figsize: Figure size
        words_to_show: Explicit list of (index, word) pairs to display.
                       If None, uses the first min(30, n) entries — which may
                       be alphabetically sorted punctuation. Prefer passing
                       a curated list of semantically interesting words.
    """
    if words_to_show is not None:
        indices = [i for i, _ in words_to_show]
        display_words = [w for _, w in words_to_show]
        sub_embeddings = embeddings[indices]
    else:
        n_show = min(30, len(words))
        sub_embeddings = embeddings[:n_show]
        display_words = words[:n_show]

    # Compute cosine similarity matrix for the selected words
    n = len(sub_embeddings)
    similarities = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                similarities[i, j] = 1.0
            else:
                dot = np.dot(sub_embeddings[i], sub_embeddings[j])
                norm_i = np.linalg.norm(sub_embeddings[i])
                norm_j = np.linalg.norm(sub_embeddings[j])
                similarities[i, j] = dot / (norm_i * norm_j + 1e-10)

    plt.figure(figsize=figsize)
    im = plt.imshow(similarities, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    plt.colorbar(im, label='Cosine Similarity')
    plt.xlabel('Word', fontsize=12)
    plt.ylabel('Word', fontsize=12)
    plt.xticks(range(n), display_words, rotation=45, ha='right')
    plt.yticks(range(n), display_words)
    plt.title('Word Embedding Similarity', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


# ============================================================================
# Chapter 5: MLP
# ============================================================================

def plot_activations(activations, layer_names=None, figsize=(12, 4)):
    """Plot activation patterns across layers

    Args:
        activations: List of activation matrices for each layer
        layer_names: Names of layers
        figsize: Figure size
    """
    n_layers = len(activations)
    if layer_names is None:
        layer_names = [f"Layer {i+1}" for i in range(n_layers)]

    fig, axes = plt.subplots(1, n_layers, figsize=figsize)
    if n_layers == 1:
        axes = [axes]

    for i, (act, name) in enumerate(zip(activations, layer_names)):
        # Show first sample's activations
        if len(act.shape) == 2:
            act_to_plot = act[0]
        else:
            act_to_plot = act

        axes[i].bar(range(len(act_to_plot)), act_to_plot)
        axes[i].set_title(name, fontweight='bold')
        axes[i].set_xlabel('Neuron Index')
        axes[i].set_ylabel('Activation')
        axes[i].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.show()


def plot_hidden_activations(model, X_sample, layer_idx=0, figsize=(10, 4)):
    """Plot hidden layer activations for a few input samples

    Args:
        model: Trained MLP with a forward() method that returns (output, cache)
        X_sample: Small batch of inputs (n_samples, input_dim)
        layer_idx: Which hidden layer to visualize (0 = first hidden)
        figsize: Figure size
    """
    _, cache = model.forward(X_sample)
    # pre_activations[layer_idx] is the pre-activation at the chosen hidden layer
    activations = cache['pre_activations'][layer_idx]  # (n_samples, n_neurons)

    n_samples = activations.shape[0]
    fig, axes = plt.subplots(1, n_samples, figsize=figsize, sharey=True)
    if n_samples == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        vals = activations[i]
        colors = ['#e05c5c' if v < 0 else '#5c9ee0' for v in vals]
        ax.barh(range(len(vals)), vals, color=colors, edgecolor='none')
        ax.axvline(0, color='white', linewidth=0.8, alpha=0.5)
        ax.set_title(f'Sample {i+1}', fontsize=10)
        ax.set_xlabel('Pre-activation', fontsize=9)
        if i == 0:
            ax.set_ylabel('Neuron', fontsize=10)
        ax.grid(True, alpha=0.2, axis='x')

    fig.suptitle(f'Hidden Layer {layer_idx+1} Pre-Activations\n'
                 f'(red = negative → ReLU kills it, blue = positive → passes through)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_decision_boundary(model_predict_fn, X, y, figsize=(8, 8)):
    """Plot decision boundary for 2D classification

    Args:
        model_predict_fn: Function that takes X and returns predictions
        X: Input data (n_samples, 2)
        y: Labels (n_samples,)
        figsize: Figure size
    """
    # Create mesh
    h = 0.1
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # Predict on mesh
    Z = model_predict_fn(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot
    plt.figure(figsize=figsize)
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
    # Draw the actual decision boundary line at probability = 0.5
    plt.contour(xx, yy, Z, levels=[0.5], colors='white', linewidths=2)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu', edgecolors='black', s=50)
    plt.xlabel('Feature 1', fontsize=12)
    plt.ylabel('Feature 2', fontsize=12)
    plt.title('Decision Boundary', fontsize=14, fontweight='bold')
    plt.colorbar(label='P(class 1)')
    plt.tight_layout()
    plt.show()


# ============================================================================
# Chapter 6: BPE Tokenization
# ============================================================================

def plot_tokenization(text, tokens, figsize=(14, 3)):
    """Visualize how text is tokenized — renders tokens as colored spans.

    Args:
        text: Original text string
        tokens: List of tokens
        figsize: Figure size
    """
    # Print summary
    print(f"Tokenized ({len(tokens)} tokens from {len(text.split())} words):")
    print(f"  {' | '.join(tokens[:40])}{'...' if len(tokens) > 40 else ''}")

    # Build a colored token display using matplotlib
    # Show only first ~20 tokens to keep it readable
    display_tokens = tokens[:20]
    n = len(display_tokens)

    # Generate distinct colors
    cmap = plt.get_cmap('tab20')
    colors = [cmap(i % 20) for i in range(n)]

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    x = 0.02
    y = 0.5
    max_width = 0.96
    for i, token in enumerate(display_tokens):
        color = colors[i]
        # Estimate token width (characters * fixed width)
        token_width = max(0.03, len(token) * 0.018 + 0.01)
        if x + token_width > max_width:
            break  # Stop if we run out of space
        # Draw colored rectangle
        rect = plt.Rectangle((x, y - 0.28), token_width, 0.56,
                              facecolor=color, alpha=0.7, edgecolor='white', linewidth=1.5)
        ax.add_patch(rect)
        # Draw token text
        ax.text(x + token_width / 2, y, token, ha='center', va='center',
                fontsize=11, fontweight='bold', color='white',
                bbox=dict(facecolor='none', edgecolor='none'))
        x += token_width + 0.005

    if len(tokens) > len(display_tokens):
        ax.text(x + 0.01, y, f'... +{len(tokens) - len(display_tokens)} more',
                va='center', fontsize=10, color='gray', style='italic')

    ax.set_title(f'Tokenization — first {min(n, len(display_tokens))} tokens shown',
                 fontsize=13, fontweight='bold', pad=8)
    plt.tight_layout()
    plt.show()


def plot_vocab_composition(vocab, figsize=(10, 6)):
    """Plot vocabulary composition by token type

    Args:
        vocab: List of tokens in vocabulary
        figsize: Figure size
    """
    # Categorize tokens
    single_char = sum(1 for token in vocab if len(token) == 1)
    short = sum(1 for token in vocab if 2 <= len(token) <= 3)
    medium = sum(1 for token in vocab if 4 <= len(token) <= 6)
    long_tokens = sum(1 for token in vocab if len(token) > 6)

    categories = ['Single\nChar', 'Short\n(2-3)', 'Medium\n(4-6)', 'Long\n(>6)']
    counts = [single_char, short, medium, long_tokens]

    plt.figure(figsize=figsize)
    plt.bar(categories, counts, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    plt.xlabel('Token Length', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title(f'Vocabulary Composition ({len(vocab)} total tokens)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()


def plot_token_lengths(tokens, figsize=(10, 5)):
    """Plot distribution of token lengths

    Args:
        tokens: List of tokens
        figsize: Figure size
    """
    lengths = [len(token) for token in tokens]

    plt.figure(figsize=figsize)
    plt.hist(lengths, bins=range(1, max(lengths) + 2), edgecolor='black', alpha=0.7)
    plt.xlabel('Token Length (characters)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Token Length Distribution', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()


# ============================================================================
# Chapter 7: RNN
# ============================================================================

def plot_hidden_states(h_seq, figsize=(12, 4)):
    """Plot the L2 norm of hidden states over timesteps.

    Shows how much information the RNN is carrying at each position.

    Args:
        h_seq: numpy array or torch.Tensor (seq_len, hidden_size)
               — one sequence (no batch dimension)
        figsize: Figure size
    """
    import torch
    if isinstance(h_seq, torch.Tensor):
        h_seq = h_seq.detach().cpu().numpy()

    norms = np.linalg.norm(h_seq, axis=-1)   # (seq_len,)

    plt.figure(figsize=figsize)
    plt.plot(norms, linewidth=2, color='steelblue')
    plt.fill_between(range(len(norms)), norms, alpha=0.2, color='steelblue')
    plt.xlabel('Timestep', fontsize=12)
    plt.ylabel('||h_t||  (hidden state norm)', fontsize=12)
    plt.title('Hidden State Magnitude Over the Sequence', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ============================================================================
# Chapter 8: LSTM
# ============================================================================

def plot_gate_activations(gates_dict, n_show=20, figsize=(14, 8)):
    """Plot LSTM gate activation heatmaps.

    Args:
        gates_dict: dict with keys 'forget', 'input', 'gate', 'output'
                    each mapping to a numpy/tensor array of shape
                    (seq_len, hidden_size) — for one example sequence
        n_show:     Number of hidden units to show (columns)
        figsize:    Figure size
    """
    import torch
    gate_names  = ['forget', 'input', 'gate', 'output']
    gate_titles = ['Forget Gate (f)', 'Input Gate (i)',
                   'Cell Candidate (g)', 'Output Gate (o)']
    cmaps = ['Blues', 'Greens', 'RdBu_r', 'Oranges']

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()

    for ax, name, title, cmap in zip(axes, gate_names, gate_titles, cmaps):
        data = gates_dict[name]
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
        data = data[:, :n_show]   # (seq_len, n_show)
        im = ax.imshow(data.T, aspect='auto', cmap=cmap, vmin=0, vmax=1)
        plt.colorbar(im, ax=ax)
        ax.set_xlabel('Timestep', fontsize=10)
        ax.set_ylabel('Hidden unit', fontsize=10)
        ax.set_title(title, fontsize=12, fontweight='bold')

    plt.suptitle('LSTM Gate Activations', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.show()


# ============================================================================
# Chapter 9: Attention
# ============================================================================

def plot_attention_weights(weights, tokens, figsize=(10, 8)):
    """Plot an attention weight matrix as an annotated heatmap.

    Rows = query positions (what is being predicted),
    Columns = key positions (what is being attended to).

    Args:
        weights: numpy array or torch.Tensor (seq_len, seq_len)
        tokens:  List[str] of length seq_len (token labels)
        figsize: Figure size
    """
    import torch
    if isinstance(weights, torch.Tensor):
        weights = weights.detach().cpu().numpy()

    n = len(tokens)
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(weights, cmap='Blues', vmin=0, vmax=weights.max())
    plt.colorbar(im, ax=ax, label='Attention weight')

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(tokens, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(tokens, fontsize=9)
    ax.set_xlabel('Key (attending to)', fontsize=11)
    ax.set_ylabel('Query (being predicted)', fontsize=11)
    ax.set_title('Attention Weights', fontsize=14, fontweight='bold')

    # Annotate cells with values
    for i in range(n):
        for j in range(n):
            val = weights[i, j]
            color = 'white' if val > weights.max() * 0.6 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=7, color=color)

    plt.tight_layout()
    plt.show()


# ============================================================================
# Chapter 10: Transformer
# ============================================================================

def plot_multi_head_attention(all_weights, tokens, figsize=None):
    """Plot per-head attention heatmaps in a grid.

    Args:
        all_weights: numpy array or torch.Tensor (n_heads, seq_len, seq_len)
                     — for a single example, single layer
        tokens:      List[str] of length seq_len
        figsize:     Figure size (auto-computed if None)
    """
    import torch
    if isinstance(all_weights, torch.Tensor):
        all_weights = all_weights.detach().cpu().numpy()

    n_heads = all_weights.shape[0]
    n_cols  = min(n_heads, 4)
    n_rows  = (n_heads + n_cols - 1) // n_cols
    if figsize is None:
        figsize = (4 * n_cols, 3.5 * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = np.array(axes).flatten() if n_heads > 1 else [axes]

    for head_idx, ax in enumerate(axes):
        if head_idx >= n_heads:
            ax.axis('off')
            continue
        w = all_weights[head_idx]
        im = ax.imshow(w, cmap='Blues', vmin=0, vmax=w.max())
        ax.set_title(f'Head {head_idx + 1}', fontsize=11, fontweight='bold')
        ax.set_xticks(range(len(tokens)))
        ax.set_yticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=45, ha='right', fontsize=7)
        ax.set_yticklabels(tokens, fontsize=7)

    plt.suptitle('Multi-Head Attention Patterns', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_positional_encoding(pe, figsize=(12, 5)):
    """Plot sinusoidal positional encodings as a heatmap.

    Args:
        pe:      numpy array or torch.Tensor (seq_len, d_model)
        figsize: Figure size
    """
    import torch
    if isinstance(pe, torch.Tensor):
        pe = pe.detach().cpu().numpy()
    if pe.ndim == 3:
        pe = pe[0]   # remove batch dim if present

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Heatmap
    im = axes[0].imshow(pe, aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1)
    plt.colorbar(im, ax=axes[0])
    axes[0].set_xlabel('Dimension', fontsize=11)
    axes[0].set_ylabel('Position', fontsize=11)
    axes[0].set_title('Positional Encoding Heatmap', fontsize=12, fontweight='bold')

    # A few individual dimensions
    for dim in [0, 1, 4, 8]:
        if dim < pe.shape[1]:
            axes[1].plot(pe[:, dim], label=f'dim {dim}', linewidth=1.5)
    axes[1].set_xlabel('Position', fontsize=11)
    axes[1].set_ylabel('Value', fontsize=11)
    axes[1].set_title('Encoding Values by Dimension', fontsize=12, fontweight='bold')
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# ============================================================================
# Chapter 11: Generation
# ============================================================================

def plot_generation_samples(samples_dict, figsize=(12, 4)):
    """Display generated text samples for different decoding strategies.

    Args:
        samples_dict: dict mapping strategy label → generated string
                      e.g. {'greedy': '...', 'temp=0.8': '...', ...}
        figsize:      Figure size (used for the enclosing figure)
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('off')

    n = len(samples_dict)
    row_height = 0.85 / max(n, 1)
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for idx, (label, text) in enumerate(samples_dict.items()):
        y = 0.95 - idx * row_height
        color = colors[idx % len(colors)]
        ax.text(0.0, y, f'{label}:', fontsize=11, fontweight='bold',
                color=color, transform=ax.transAxes, va='top')
        # Wrap long text
        max_chars = 110
        display = text if len(text) <= max_chars else text[:max_chars] + '…'
        ax.text(0.18, y, display, fontsize=10, color='black',
                transform=ax.transAxes, va='top',
                fontfamily='monospace', wrap=True)

    ax.set_title('Generated Text — Decoding Strategy Comparison',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()
