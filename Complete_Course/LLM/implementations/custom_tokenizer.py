"""
Byte Pair Encoding (BPE) Tokenizer from Scratch.
This script demonstrates the training of a subword tokenizer on a corpus,
generating the vocabulary, and using the learned merges to tokenize new text.

Run: python Complete_Course/LLM/implementations/custom_tokenizer.py
"""

from collections import defaultdict

class ScratchBPETokenizer:
    def __init__(self, vocab_size=30):
        self.vocab_size = vocab_size
        self.merges = {}  # Stores learned merges e.g., ('h', 'e') -> 'he'
        self.vocab = []   # Stores final subwords

    def _get_stats(self, ids):
        """Count frequencies of adjacent symbol pairs."""
        counts = defaultdict(int)
        for word, freq in ids.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                counts[(symbols[i], symbols[i+1])] += freq
        return counts

    def _merge_vocab(self, pair, v_in):
        """Merge all occurrences of the most frequent pair in the vocabulary."""
        v_out = {}
        bigram = " ".join(pair)
        replacement = "".join(pair)
        for word in v_in:
            w_out = word.replace(bigram, replacement)
            v_out[w_out] = v_in[word]
        return v_out

    def train(self, corpus):
        # 1. Initialize vocabulary with individual characters + special space marker
        # We use a space suffix or prefix to represent spaces between words.
        # E.g., we split words and append a trailing '</w>' to mark word boundaries.
        words = corpus.strip().split()
        
        # Word count frequency map
        vocab_counts = defaultdict(int)
        for word in words:
            # Represent word as characters separated by spaces, ending with word-end marker
            spaced_word = " ".join(list(word)) + " </w>"
            vocab_counts[spaced_word] += 1

        # Calculate initial characters to seed vocab
        initial_chars = set()
        for word in vocab_counts:
            for char in word.split():
                initial_chars.add(char)
        self.vocab = list(initial_chars)

        # 2. Iteratively merge the most frequent pairs until target vocab size is reached
        num_merges = self.vocab_size - len(self.vocab)
        print(f"Initial unique characters count: {len(self.vocab)}")
        print(f"Goal vocabulary size: {self.vocab_size}. Performing {num_merges} merges...")

        for i in range(num_merges):
            pairs = self._get_stats(vocab_counts)
            if not pairs:
                break
            best_pair = max(pairs, key=pairs.get)
            
            # Record merge rule
            self.merges[best_pair] = "".join(best_pair)
            self.vocab.append("".join(best_pair))
            
            # Apply merge to all words in training corpus
            vocab_counts = self._merge_vocab(best_pair, vocab_counts)
            print(f"Merge #{i+1}: {best_pair} -> {''.join(best_pair)} (Frequency: {pairs[best_pair]})")

    def tokenize(self, text):
        """Tokenize a new string using the learned merges."""
        words = text.strip().split()
        tokenized_words = []
        
        for word in words:
            # Start by splitting the word into individual characters + word-end marker
            w_chars = list(word) + ["</w>"]
            
            # Keep merging characters as long as we have valid rules in our merges dictionary
            while True:
                # Find all current adjacent pairs
                pairs = [(w_chars[i], w_chars[i+1]) for i in range(len(w_chars) - 1)]
                
                # Check which of these pairs can be merged based on our learned merges list
                # We want to merge the ones that were learned earliest (highest priority)
                mergeable = [pair for pair in pairs if pair in self.merges]
                if not mergeable:
                    break
                
                # Pick the merge with the highest priority (first learned)
                pair_to_merge = min(mergeable, key=lambda p: list(self.merges.keys()).index(p))
                new_symbol = self.merges[pair_to_merge]
                
                # Create the merged array of characters
                new_w_chars = []
                i = 0
                while i < len(w_chars):
                    if i < len(w_chars) - 1 and (w_chars[i], w_chars[i+1]) == pair_to_merge:
                        new_w_chars.append(new_symbol)
                        i += 2
                    else:
                        new_w_chars.append(w_chars[i])
                        i += 1
                w_chars = new_w_chars
                
            tokenized_words.extend(w_chars)
            
        return tokenized_words


def main():
    # Training corpus
    corpus = (
        "hug hug hug pug pug hug hug hug </w> "
        "pun pun pun pun pun pun pun pun </w> "
        "hugger hugger hugger hugger hugger </w>"
    )
    
    print("--- Training BPE Tokenizer ---")
    tokenizer = ScratchBPETokenizer(vocab_size=15)
    tokenizer.train(corpus)
    
    print("\n--- Final Learned Vocabulary ---")
    print(tokenizer.vocab)
    
    print("\n--- Final Learned Merges ---")
    print(tokenizer.merges)

    # Tokenize test phrase
    test_phrase = "hug pug pun hugger"
    tokens = tokenizer.tokenize(test_phrase)
    print(f"\nOriginal Text:  '{test_phrase}'")
    print(f"Tokenized Subwords: {tokens}")

if __name__ == '__main__':
    main()
