"""Data generators and loaders for LLM textbook

Provides:
- Synthetic data generators for Chapters 1-3, 5
- WikiText-2 downloader and loader for Chapters 4, 6-11
"""

import numpy as np
import os
import urllib.request
from pathlib import Path


# ============================================================================
# Synthetic Data Generators (Chapters 1-3, 5)
# ============================================================================

def generate_linear_sequence(n=100, slope=2, intercept=0, noise=0.1, seed=42):
    """Generate simple linear sequence for Chapter 1: Linear Regression

    Args:
        n: Number of data points
        slope: Slope of the line
        intercept: Y-intercept
        noise: Standard deviation of Gaussian noise
        seed: Random seed for reproducibility

    Returns:
        x: Input values (n,)
        y: Output values (n,)
    """
    np.random.seed(seed)
    x = np.linspace(0, 1, n)
    y = slope * x + intercept + np.random.normal(0, noise, n)
    return x, y


def generate_simple_text(n_words=100, vocab_size=10, seed=42):
    """Generate simple text with tiny vocabulary for Chapter 2: One-Hot Encoding

    Creates deterministic "text" with a small vocabulary for clear demonstrations.

    Args:
        n_words: Number of words to generate
        vocab_size: Size of vocabulary
        seed: Random seed for reproducibility

    Returns:
        text: List of words
        vocab: List of unique words in vocabulary
    """
    np.random.seed(seed)
    # Create simple vocabulary using real common words
    _base_vocab = ["the", "cat", "sat", "on", "mat", "a", "dog", "ran", "big", "fast",
                   "old", "new", "red", "hot", "saw"]
    vocab = _base_vocab[:vocab_size]
    # Generate text by sampling from vocabulary
    indices = np.random.randint(0, vocab_size, n_words)
    text = [vocab[i] for i in indices]
    return text, vocab


def generate_next_word_pairs(text, window_size=1):
    """Generate (context, target) pairs for next-word prediction

    Used in Chapter 3: Logistic Regression for classification tasks.

    Args:
        text: List of words
        window_size: Number of context words to use

    Returns:
        contexts: List of context windows (each a list of words)
        targets: List of target words
    """
    contexts = []
    targets = []

    for i in range(window_size, len(text)):
        context = text[i-window_size:i]
        target = text[i]
        contexts.append(context)
        targets.append(target)

    return contexts, targets


def generate_nonlinear_data(n=200, noise=0.2, seed=42):
    """Generate data with non-linear patterns for Chapter 5: MLP

    Creates data that requires non-linear model to fit well.

    Args:
        n: Number of data points
        noise: Standard deviation of Gaussian noise
        seed: Random seed for reproducibility

    Returns:
        X: Input features (n, 2)
        y: Binary labels (n,)
    """
    np.random.seed(seed)
    # Generate two clusters in a non-linearly separable pattern
    n_per_class = n // 2

    # Class 0: Inner circle
    r0 = np.random.uniform(0, 2, n_per_class)
    theta0 = np.random.uniform(0, 2 * np.pi, n_per_class)
    x0 = r0 * np.cos(theta0)
    y0 = r0 * np.sin(theta0)

    # Class 1: Outer ring
    r1 = np.random.uniform(3, 5, n_per_class)
    theta1 = np.random.uniform(0, 2 * np.pi, n_per_class)
    x1 = r1 * np.cos(theta1)
    y1 = r1 * np.sin(theta1)

    # Combine
    X = np.vstack([
        np.column_stack([x0, y0]),
        np.column_stack([x1, y1])
    ])
    y = np.hstack([np.zeros(n_per_class), np.ones(n_per_class)])

    # Add noise
    X += np.random.normal(0, noise, X.shape)

    # Shuffle
    indices = np.random.permutation(n)
    return X[indices], y[indices]


# ============================================================================
# WikiText-2 Loader (Chapters 4, 6-11)
# ============================================================================

def get_cache_dir():
    """Get cache directory for downloaded datasets"""
    cache_dir = Path.home() / '.llmkit_cache'
    cache_dir.mkdir(exist_ok=True)
    return cache_dir


def download_wikitext2(cache_dir=None):
    """Download WikiText-2 dataset if not already cached

    Downloads plain text files from the PyTorch examples repository.

    Args:
        cache_dir: Directory to cache the dataset (default: ~/.llmkit_cache)

    Returns:
        Path to the dataset directory
    """
    if cache_dir is None:
        cache_dir = get_cache_dir()
    else:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(exist_ok=True)

    dataset_dir = cache_dir / 'wikitext-2'

    # Check if already fully downloaded
    expected_files = [dataset_dir / f'wiki.{s}.tokens' for s in ['train', 'valid', 'test']]
    if dataset_dir.exists() and all(f.exists() and f.stat().st_size > 0 for f in expected_files):
        return dataset_dir

    # Clean up any partial download before retrying
    if dataset_dir.exists():
        import shutil
        shutil.rmtree(dataset_dir)

    dataset_dir.mkdir()

    base_url = 'https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2'
    splits = ['train', 'valid', 'test']

    for split in splits:
        url = f'{base_url}/{split}.txt'
        dest = dataset_dir / f'wiki.{split}.tokens'
        print(f"Downloading {split}.txt...")
        urllib.request.urlretrieve(url, dest)

    print("Download complete!")
    return dataset_dir


def load_wikitext2(split='train', cache_dir=None):
    """Load WikiText-2 dataset

    Args:
        split: One of 'train', 'valid', or 'test'
        cache_dir: Cache directory (default: ~/.llmkit_cache)

    Returns:
        text: Raw text as a string
    """
    if split not in ['train', 'valid', 'test']:
        raise ValueError(f"split must be 'train', 'valid', or 'test', got {split}")

    # Ensure dataset is downloaded
    dataset_dir = download_wikitext2(cache_dir)

    # Load the appropriate file
    file_path = dataset_dir / f'wiki.{split}.tokens'

    if not file_path.exists():
        raise FileNotFoundError(f"Could not find {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    return text


def get_small_corpus(n_sentences=500, split='train', cache_dir=None):
    """Get a small subset of WikiText-2 for fast exercises

    Args:
        n_sentences: Number of sentences to include (approximate)
        split: Dataset split to use
        cache_dir: Cache directory

    Returns:
        text: Subset of text as a string
    """
    full_text = load_wikitext2(split=split, cache_dir=cache_dir)

    # Split by periods (rough sentence boundary)
    sentences = []
    current = []
    for line in full_text.split('\n'):
        line = line.strip()
        if line:
            current.append(line)
            # Consider paragraph as sentence boundary
            if len(current) >= 2:
                sentences.append(' '.join(current))
                current = []
                if len(sentences) >= n_sentences:
                    break

    return ' '.join(sentences)


def get_full_corpus(split='train', cache_dir=None):
    """Get the full WikiText-2 corpus

    Args:
        split: Dataset split to use
        cache_dir: Cache directory

    Returns:
        text: Full text as a string
    """
    return load_wikitext2(split=split, cache_dir=cache_dir)


def prepare_word2vec_data(text, window_size=2):
    """Prepare data for Word2Vec training (CBOW or Skip-gram)

    Args:
        text: Raw text string
        window_size: Context window size on each side

    Returns:
        pairs: List of (context_words, target_word) tuples
        vocab: List of unique words
        word2idx: Dictionary mapping words to indices
        idx2word: Dictionary mapping indices to words
    """
    # Tokenize (simple whitespace split)
    words = text.lower().split()

    # Build vocabulary
    vocab = sorted(set(words))
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for word, idx in word2idx.items()}

    # Generate training pairs
    pairs = []
    for i in range(window_size, len(words) - window_size):
        target = words[i]
        # Context: words before and after (excluding target)
        context = (
            words[i-window_size:i] +
            words[i+1:i+1+window_size]
        )
        pairs.append((context, target))

    return pairs, vocab, word2idx, idx2word


def tokenize_simple(text):
    """Simple whitespace tokenization

    Args:
        text: Raw text string

    Returns:
        tokens: List of tokens (words)
    """
    return text.lower().split()


# ============================================================================
# Sequence Data Helpers (Chapters 7-11)
# ============================================================================

def prepare_lm_sequences(token_ids, seq_len=64):
    """Create (input, target) pairs for language modeling.

    Slides a window of length seq_len over token_ids.
    Each input sequence is paired with itself shifted one step right as target.

    Args:
        token_ids: List or 1-D array of integer token ids
        seq_len:   Length of each input/target sequence

    Returns:
        inputs:  LongTensor (n_sequences, seq_len)
        targets: LongTensor (n_sequences, seq_len)
    """
    import torch
    if not isinstance(token_ids, torch.Tensor):
        token_ids = torch.tensor(token_ids, dtype=torch.long)

    inputs, targets = [], []
    for i in range(len(token_ids) - seq_len):
        inputs.append(token_ids[i : i + seq_len])
        targets.append(token_ids[i + 1 : i + seq_len + 1])

    return torch.stack(inputs), torch.stack(targets)


def get_token_ids(corpus, tokenizer):
    """Encode a raw text corpus to a flat list of integer token ids.

    Args:
        corpus:    Raw text string
        tokenizer: Object with an .encode_ids(str) → list[int] method
                   (e.g. the BytePairEncoding instance from ch06_bpe)

    Returns:
        token_ids: List[int]
    """
    if hasattr(tokenizer, 'encode_ids'):
        return tokenizer.encode_ids(corpus)
    return tokenizer.encode(corpus)
