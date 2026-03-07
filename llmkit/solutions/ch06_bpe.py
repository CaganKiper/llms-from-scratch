"""Chapter 6: Byte Pair Encoding (BPE) Tokenization

Core concepts:
- Out-of-vocabulary (OOV) problem
- Granularity spectrum: character ↔ subword ↔ word
- BPE algorithm: Iteratively merge most frequent pairs
- Token vs word: Reframing prediction task
"""

import numpy as np
from collections import Counter, defaultdict
import re


# ============================================================================
# BPE Algorithm
# ============================================================================

class BytePairEncoding:
    """Byte Pair Encoding tokenizer

    Learns subword vocabulary by iteratively merging frequent character pairs.
    """

    def __init__(self, vocab_size=1000):
        """Initialize BPE tokenizer

        Args:
            vocab_size: Target vocabulary size
        """
        self.vocab_size = vocab_size
        self.vocab = []
        self.merges = []  # List of (pair, merged_token) in order learned
        self.token2idx = {}
        self.idx2token = {}

    def _get_word_freqs(self, text):
        """Get word frequencies from text

        Args:
            text: Raw text string

        Returns:
            word_freqs: Dictionary mapping words to frequencies
        """
        # Simple tokenization: split on whitespace and punctuation
        words = re.findall(r'\w+|[^\w\s]', text.lower())
        return Counter(words)

    def _get_char_vocab(self, word_freqs):
        """Get initial character vocabulary

        Args:
            word_freqs: Dictionary mapping words to frequencies

        Returns:
            vocab: Set of characters
        """
        vocab = set()
        for word in word_freqs.keys():
            # Represent word as character sequence with end marker
            chars = list(word) + ['</w>']
            vocab.update(chars)
        return sorted(vocab)

    def _get_pairs(self, word_splits):
        """Get all adjacent pairs from word splits

        Args:
            word_splits: Dictionary mapping words to their current split representation

        Returns:
            pairs: Counter of (token1, token2) pairs
        """
        pairs = Counter()

        for word, freq in word_splits.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += freq

        return pairs

    def _merge_pair(self, pair, word_splits):
        """Merge a pair in all words

        Args:
            pair: Tuple (token1, token2) to merge
            word_splits: Dictionary mapping words to their split representation

        Returns:
            new_word_splits: Updated word splits
        """
        new_word_splits = {}
        bigram = ' '.join(pair)
        replacement = ''.join(pair)

        for word, freq in word_splits.items():
            # Replace pair in word
            new_word = word.replace(bigram, replacement)
            new_word_splits[new_word] = freq

        return new_word_splits

    def train(self, text):
        """Train BPE on text

        Args:
            text: Raw text string
        """
        # Get word frequencies
        word_freqs = self._get_word_freqs(text)

        # Initialize with character vocabulary
        vocab = self._get_char_vocab(word_freqs)

        # Represent each word as space-separated characters with end marker
        word_splits = {}
        for word, freq in word_freqs.items():
            chars = list(word) + ['</w>']
            word_splits[' '.join(chars)] = freq

        # Iteratively merge most frequent pairs
        while len(vocab) < self.vocab_size:
            # Get all pairs
            pairs = self._get_pairs(word_splits)

            if not pairs:
                break

            # Find most frequent pair
            best_pair = max(pairs, key=pairs.get)

            # Merge this pair in all words
            word_splits = self._merge_pair(best_pair, word_splits)

            # Add merged token to vocabulary
            new_token = ''.join(best_pair)
            vocab.append(new_token)

            # Record this merge
            self.merges.append(best_pair)

        # Build final vocabulary and mappings
        self.vocab = sorted(vocab)
        self.token2idx = {token: idx for idx, token in enumerate(self.vocab)}
        self.idx2token = {idx: token for token, idx in self.token2idx.items()}

        return self

    def _apply_merges(self, word):
        """Apply learned merges to a word

        Args:
            word: Word string

        Returns:
            tokens: List of subword tokens
        """
        # Start with characters + end marker
        chars = list(word) + ['</w>']
        word_str = ' '.join(chars)

        # Apply each merge in order
        for pair in self.merges:
            bigram = ' '.join(pair)
            replacement = ''.join(pair)
            word_str = word_str.replace(bigram, replacement)

        # Split into tokens
        tokens = word_str.split()
        return tokens

    def encode(self, text):
        """Encode text into tokens

        Args:
            text: Raw text string

        Returns:
            tokens: List of subword tokens
        """
        # Tokenize into words
        words = re.findall(r'\w+|[^\w\s]', text.lower())

        # Apply BPE to each word
        all_tokens = []
        for word in words:
            tokens = self._apply_merges(word)
            all_tokens.extend(tokens)

        return all_tokens

    def encode_ids(self, text):
        """Encode text into token IDs

        Args:
            text: Raw text string

        Returns:
            ids: List of token IDs
        """
        tokens = self.encode(text)
        ids = [self.token2idx.get(token, 0) for token in tokens]  # 0 for unknown
        return ids

    def decode(self, tokens):
        """Decode tokens back to text

        Args:
            tokens: List of subword tokens

        Returns:
            text: Reconstructed text
        """
        text = ''.join(tokens)
        text = text.replace('</w>', ' ')
        return text.strip()

    def decode_ids(self, ids):
        """Decode token IDs back to text

        Args:
            ids: List of token IDs

        Returns:
            text: Reconstructed text
        """
        tokens = [self.idx2token.get(idx, '') for idx in ids]
        return self.decode(tokens)


# ============================================================================
# Simple BPE Functions (for exercises)
# ============================================================================

def get_pair_frequencies(word_splits):
    """Count frequencies of adjacent token pairs

    Args:
        word_splits: Dictionary mapping word representations to frequencies
                     e.g., {'h e l l o </w>': 5, 'w o r l d </w>': 3}

    Returns:
        pair_freqs: Counter of (token1, token2) pairs
    """
    pair_freqs = Counter()

    for word, freq in word_splits.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pair = (symbols[i], symbols[i + 1])
            pair_freqs[pair] += freq

    return pair_freqs


def merge_pair_in_word(word, pair):
    """Merge a pair in a single word representation

    Args:
        word: Space-separated token string (e.g., 'h e l l o </w>')
        pair: Tuple of tokens to merge (e.g., ('l', 'l'))

    Returns:
        new_word: Word with pair merged (e.g., 'h e ll o </w>')
    """
    bigram = ' '.join(pair)
    replacement = ''.join(pair)
    return word.replace(bigram, replacement)


def build_bpe_vocab(text, n_merges=100):
    """Build BPE vocabulary with simple interface

    Args:
        text: Raw text string
        n_merges: Number of merge operations to perform

    Returns:
        vocab: List of tokens in vocabulary
        merges: List of (token1, token2) pairs in merge order
    """
    # Get word frequencies
    words = re.findall(r'\w+', text.lower())
    word_freqs = Counter(words)

    # Initialize: each word as characters + end marker
    word_splits = {}
    vocab = set()

    for word, freq in word_freqs.items():
        chars = list(word) + ['</w>']
        word_splits[' '.join(chars)] = freq
        vocab.update(chars)

    merges = []

    # Perform merges
    for _ in range(n_merges):
        pair_freqs = get_pair_frequencies(word_splits)

        if not pair_freqs:
            break

        # Get most frequent pair
        best_pair = max(pair_freqs, key=pair_freqs.get)

        # Merge in all words
        new_word_splits = {}
        for word, freq in word_splits.items():
            new_word = merge_pair_in_word(word, best_pair)
            new_word_splits[new_word] = freq

        word_splits = new_word_splits

        # Add to vocab and record merge
        new_token = ''.join(best_pair)
        vocab.add(new_token)
        merges.append(best_pair)

    return sorted(vocab), merges


# ============================================================================
# Tokenization Statistics
# ============================================================================

def compute_compression_ratio(text, tokens):
    """Compute compression ratio of tokenization

    Args:
        text: Original text
        tokens: List of tokens

    Returns:
        ratio: Compression ratio (n_words / n_tokens)
    """
    words = text.split()
    return len(words) / len(tokens) if tokens else 0


def get_token_statistics(tokens):
    """Get statistics about tokens

    Args:
        tokens: List of tokens

    Returns:
        stats: Dictionary with statistics
    """
    lengths = [len(token.replace('</w>', '')) for token in tokens]

    return {
        'n_tokens': len(tokens),
        'unique_tokens': len(set(tokens)),
        'avg_length': np.mean(lengths),
        'min_length': min(lengths) if lengths else 0,
        'max_length': max(lengths) if lengths else 0,
        'token_freq': Counter(tokens)
    }


def compare_tokenizations(text):
    """Compare different tokenization strategies

    Args:
        text: Raw text string

    Returns:
        comparison: Dictionary with results from different strategies
    """
    # Word-level
    words = text.split()

    # Character-level
    chars = list(text.replace(' ', '▁'))  # Use ▁ to mark spaces

    # BPE (with moderate vocab size)
    bpe = BytePairEncoding(vocab_size=500)
    bpe.train(text)
    bpe_tokens = bpe.encode(text)

    return {
        'word': {
            'n_tokens': len(words),
            'vocab_size': len(set(words)),
            'sample': words[:20]
        },
        'character': {
            'n_tokens': len(chars),
            'vocab_size': len(set(chars)),
            'sample': chars[:20]
        },
        'bpe': {
            'n_tokens': len(bpe_tokens),
            'vocab_size': len(bpe.vocab),
            'sample': bpe_tokens[:20]
        }
    }
