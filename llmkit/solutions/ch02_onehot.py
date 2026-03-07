"""Chapter 2: One-Hot Encoding

Core concepts:
- Vocabulary: Set of unique words
- One-hot encoding: Binary vector with 1 at word's index, 0s elsewhere
- Properties: Sparse, equidistant, no semantic meaning
"""

import numpy as np


def build_vocabulary(text):
    """Build vocabulary from text

    Args:
        text: List of words or string

    Returns:
        vocab: Sorted list of unique words
        word2idx: Dictionary mapping words to indices
        idx2word: Dictionary mapping indices to words
    """
    if isinstance(text, str):
        words = text.lower().split()
    else:
        words = [w.lower() for w in text]

    # Get unique words and sort
    vocab = sorted(set(words))

    # Create mappings
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for word, idx in word2idx.items()}

    return vocab, word2idx, idx2word


def word_to_index(word, word2idx):
    """Convert word to index

    Args:
        word: Word string
        word2idx: Word-to-index dictionary

    Returns:
        idx: Index of word in vocabulary
    """
    return word2idx.get(word.lower(), None)


def index_to_word(idx, idx2word):
    """Convert index to word

    Args:
        idx: Index in vocabulary
        idx2word: Index-to-word dictionary

    Returns:
        word: Word string
    """
    return idx2word.get(idx, None)


def one_hot_encode(word, word2idx, vocab_size=None):
    """Encode a single word as one-hot vector

    Args:
        word: Word string
        word2idx: Word-to-index dictionary
        vocab_size: Size of vocabulary (if None, inferred from word2idx)

    Returns:
        vector: One-hot encoded vector (vocab_size,)
    """
    if vocab_size is None:
        vocab_size = len(word2idx)

    vector = np.zeros(vocab_size, dtype=np.float32)
    idx = word_to_index(word, word2idx)

    if idx is not None:
        vector[idx] = 1.0

    return vector


def one_hot_encode_sequence(words, word2idx, vocab_size=None):
    """Encode sequence of words as matrix of one-hot vectors

    Args:
        words: List of words
        word2idx: Word-to-index dictionary
        vocab_size: Size of vocabulary (if None, inferred from word2idx)

    Returns:
        matrix: One-hot encoded matrix (n_words, vocab_size)
    """
    if vocab_size is None:
        vocab_size = len(word2idx)

    matrix = np.zeros((len(words), vocab_size), dtype=np.float32)

    for i, word in enumerate(words):
        matrix[i] = one_hot_encode(word, word2idx, vocab_size)

    return matrix


def one_hot_decode(vector, idx2word):
    """Decode one-hot vector back to word

    Args:
        vector: One-hot encoded vector (vocab_size,)
        idx2word: Index-to-word dictionary

    Returns:
        word: Decoded word string
    """
    idx = np.argmax(vector)
    return index_to_word(idx, idx2word)


def compute_euclidean_distance(vec1, vec2):
    """Compute Euclidean distance between two vectors

    Args:
        vec1: First vector
        vec2: Second vector

    Returns:
        distance: Euclidean distance
    """
    return np.linalg.norm(vec1 - vec2)


def compute_pairwise_distances(vectors):
    """Compute pairwise Euclidean distances

    Args:
        vectors: Matrix of vectors (n_vectors, dim)

    Returns:
        distances: Distance matrix (n_vectors, n_vectors)
    """
    n = len(vectors)
    distances = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            distances[i, j] = compute_euclidean_distance(vectors[i], vectors[j])

    return distances


class OneHotEncoder:
    """One-hot encoder for text

    Encapsulates vocabulary building and encoding/decoding operations.
    """

    def __init__(self, vocab=None):
        """Initialize encoder

        Args:
            vocab: Optional pre-defined vocabulary (list of words)
        """
        if vocab is not None:
            self.vocab = sorted(vocab)
            self.word2idx = {word: idx for idx, word in enumerate(self.vocab)}
            self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        else:
            self.vocab = None
            self.word2idx = None
            self.idx2word = None

    def fit(self, text):
        """Build vocabulary from text

        Args:
            text: List of words or string
        """
        self.vocab, self.word2idx, self.idx2word = build_vocabulary(text)
        return self

    def encode(self, word):
        """Encode single word

        Args:
            word: Word string

        Returns:
            vector: One-hot vector (vocab_size,)
        """
        if self.word2idx is None:
            raise ValueError("Encoder not fitted. Call fit() first.")
        return one_hot_encode(word, self.word2idx)

    def encode_sequence(self, words):
        """Encode sequence of words

        Args:
            words: List of words

        Returns:
            matrix: One-hot matrix (n_words, vocab_size)
        """
        if self.word2idx is None:
            raise ValueError("Encoder not fitted. Call fit() first.")
        return one_hot_encode_sequence(words, self.word2idx)

    def decode(self, vector):
        """Decode one-hot vector to word

        Args:
            vector: One-hot vector (vocab_size,)

        Returns:
            word: Decoded word
        """
        if self.idx2word is None:
            raise ValueError("Encoder not fitted. Call fit() first.")
        return one_hot_decode(vector, self.idx2word)

    @property
    def vocab_size(self):
        """Get vocabulary size"""
        return len(self.vocab) if self.vocab is not None else 0
