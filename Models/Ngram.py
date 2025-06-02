from collections import defaultdict
import math
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Data.data_preparation import get_sentence_tokens
                                   
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

    
    def get_all_suggestions(self, previous_tokens, n_gram_counts_list, vocabulary, smoothing_factor=1.0, started_word="", filter_specials=False):
        suggestions = {}
        
        for i in range(len(n_gram_counts_list) - 1):
            n_gram_counts = n_gram_counts_list[i]
            nplus1_gram_counts = n_gram_counts_list[i + 1]

            previous_ngram = previous_tokens[-(i+1):] 
                
            probabilities = self.return_probabilities(previous_ngram, n_gram_counts, nplus1_gram_counts, vocabulary, smoothing_factor)
            specials = {'<unk>', '<pad>', '<sos>', '<eos>'}

            for word, prob in probabilities.items():
                if started_word and not word.startswith(started_word):
                    continue
                if word not in suggestions or prob > suggestions[word]:
                    if filter_specials and word in specials:  # special tokens will be filtered out in the GUI 
                        continue
                    # exclude the full word itself from suggestions
                    if word != started_word:
                        suggestions[word] = prob

        return suggestions
    
    def get_top_suggestions(self, previous_tokens, n_gram_counts_list, vocabulary, k=3, smoothing_factor=1.0, started_word=""):
        if started_word == "" and (not previous_tokens or len(previous_tokens) < self.n - 1):
            previous_tokens = ["<sos>"]   # if no context is given, assume start of sentence
            
        suggestions_dict = self.get_all_suggestions(previous_tokens, n_gram_counts_list, vocabulary, smoothing_factor, started_word, filter_specials=True)

        # Sort by probability (descending)
        word_candidates = [word for word, _ in sorted(suggestions_dict.items(), key=lambda x: -x[1])]
        
        return word_candidates[:k]


    def build_vocab(self, data):
            vocab = set()
            for sentence in data:
                for word in sentence:
                    vocab.add(word)
            vocab.update(["<eos>", "<unk>", "<pad>", "<sos>"])
            return vocab
             
    def build_ngram_counts(self, data, max_n):
        ngram_counts_list = []
        
        for n in range(1, max_n + 1):
            counts = defaultdict(int)
            for sentence in data:
                # Add <sos> and <eos> tokens and pad shorter contexts with <pad>
                padded = ["<pad>"] * (n - 2) + ["<sos>"] + sentence + ["<eos>"]
                for i in range(len(padded) - n + 1):
                    ngram = tuple(padded[i:i+n])
                    counts[ngram] += 1
            ngram_counts_list.append(counts)
        
        return ngram_counts_list

        
if __name__ == '__main__':
    # Test
    # data = [["hi", "there"], ["i", "like", "dogs", "and", "cats"], ["i", "like", "cats"], ["i", "like", "dog"], ["i", "like", "dogs"], ["i", "like", "dinner"], ["i", "like", "dinner"]]
    data = get_sentence_tokens("Data/Datasets/conv_train.csv")
    model = Ngram(n=3)
    
    ngram_counts_list = model.build_ngram_counts(data, max_n=5)
    vocab = model.build_vocab(data)
    
    word_candidates = model.get_top_suggestions(['are', 'you', 'going'], ngram_counts_list, vocab, k=5, started_word="")
    print(word_candidates)
    
    word_candidates = model.get_top_suggestions(['are', 'you'], ngram_counts_list, vocab, k=5, started_word="")
    print(word_candidates)

    