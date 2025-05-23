from collections import defaultdict
import math
import numpy as np

class Ngram:
    def __init__(self, n):
        self.n = n

    def get_next_word_prob(self, word, vocab_size, previous_ngram, n_gram_counts, nplus1_gram_counts, smoothing_factor=1.0):
        """
        Return the conditional probability of a word occurring after a given n-gram.
        """
        previous_ngram = tuple(previous_ngram)
        previous_count = n_gram_counts.get(previous_ngram, 0)
        nplus1_gram = previous_ngram + (word,)
        nplus1_count = nplus1_gram_counts.get(nplus1_gram, 0)

        # Laplace smoothing
        prob = (nplus1_count + smoothing_factor) / (previous_count + vocab_size * smoothing_factor)
        return prob

    def return_probabilities(self, previous_ngram, n_gram_counts, nplus1_gram_counts, vocabulary, smoothing_factor=1.0):
        previous_ngram = tuple(previous_ngram)
        vocab_size = len(vocabulary)

        probabilities = {
            word: self.get_next_word_prob(word, vocab_size, previous_ngram, n_gram_counts, nplus1_gram_counts, smoothing_factor)
            for word in vocabulary
        }

        return probabilities

    def complete_word(self, previous_tokens, n_gram_counts, nplus1_gram_counts, vocabulary, smoothing_factor=1.0, started_word=None):
        previous_ngram = previous_tokens[-self.n:]
        probabilities = self.return_probabilities(previous_ngram, n_gram_counts, nplus1_gram_counts, vocabulary, smoothing_factor)

        suggestion = None
        max_prob = 0.0

        for word, prob in probabilities.items():
            if started_word and not word.startswith(started_word):
                continue
            if prob > max_prob:
                suggestion = word
                max_prob = prob

        return suggestion, max_prob

    def get_suggestions(self, previous_tokens, n_gram_counts_list, vocabulary, smoothing_factor=1.0, started_word=None):
        suggestions = []
        used_words = set()

        for i in range(len(n_gram_counts_list) - 1):
            n_gram_counts = n_gram_counts_list[i]
            nplus1_gram_counts = n_gram_counts_list[i + 1]

            suggestion, prob = self.complete_word(previous_tokens, n_gram_counts, nplus1_gram_counts, vocabulary,
                                            smoothing_factor, started_word)
            
            # If suggestion already exists, don't suggest it again
            if suggestion and suggestion not in used_words:
                suggestions.append((suggestion, prob))
                used_words.add(suggestion)

        return sorted(suggestions, reverse=True)

             
def build_ngram_counts(data, max_n):
    ngram_counts_list = []
    
    for n in range(1, max_n + 1):
        counts = defaultdict(int)
        for sentence in data:
            padded = ["<s>"] * (n - 1) + sentence + ["<eos>"]
            for i in range(len(padded) - n + 1):
                ngram = tuple(padded[i:i+n])
                counts[ngram] += 1
        ngram_counts_list.append(counts)
    
    return ngram_counts_list

        
        
if __name__ == '__main__':
    # Test
    data = [["hi", "there"], ["i", "like", "dogs"]]
    ngram_counts_list = build_ngram_counts(data, max_n=5)
    vocab = {"hi", "there", "i", "like", "dogs", "and", "i", "like", "cats", "<eos>", "<unk>"}

    model = Ngram(n=3)
    suggestions = model.get_suggestions(["i", "like"], ngram_counts_list, vocab, started_word="d")
    print(suggestions)

    