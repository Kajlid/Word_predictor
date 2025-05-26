import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torch
from nltk.tokenize import word_tokenize

def create_dataframe(filename):
    df = pd.read_csv(filename, skiprows=0)
    
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    
    # Concatenate questions and answers vertically 
    stacked_df = pd.concat([df['question'], df['answer']]).dropna().reset_index(drop=True)
    
    return stacked_df


def get_sentence_tokens(filename):
    """ Get a list of sentences as lists of tokens for n-gram.

    Args:
        filename (string): path to raw text dataset.

    Returns:
        sentence_tokens (list): list of lists of tokens (list of sentences).
    """
    conv_data = create_dataframe(filename)
    sentence_tokens = conv_data.apply(word_tokenize)     # Lists of tokens
    
    return sentence_tokens.tolist()


def word_index_mappings():
    pass


def read_embeddings(pth, vocab, dim):
    emb = np.random.uniform(-0.05, 0.05, (len(vocab), dim)).astype(np.float32)
    found_tokens = 0
    with open(pth, 'r') as f:
        for line in f:
            parts = line.split(' ', 1)
            token, vec = parts[0], parts[1]
            if token in vocab:
                emb[vocab[token]] = np.fromstring(vec, sep=' ', dtype=np.float32)
                found_tokens += 1

        print(f'Found embeddings for {found_tokens} / {len(vocab)}')
    return torch.tensor(emb, dtype=torch.float32)

    
class FullSentenceDataset(Dataset):
    """
    Each sample is one sentence:
      x: all tokens except the last
      y: all tokens except the first (i.e. next-word targets)
    """
    def __init__(self, sentences, pad_idx):
        self.sents = [s for s in sentences if len(s) > 1]
        self.pad_idx = pad_idx


    def __len__(self):
        return len(self.sents)

    def __getitem__(self, idx):
        s = self.sents[idx]
        # input is everything but the last token
        x = torch.tensor(s[:-1], dtype=torch.long)
        # target is everything but the first token
        y = torch.tensor(s[1:],  dtype=torch.long)
        return x, y

def numericalize_sentence(sent, tok2id, unk_idx):
    return [tok2id.get(tok, unk_idx) for tok in sent]

def collate_full_sentences(batch):
    xs, ys = zip(*batch)
    x_padded = pad_sequence(xs, batch_first=True, padding_value=1)
    y_padded = pad_sequence(ys, batch_first=True, padding_value=1)
    return x_padded, y_padded

if __name__== "__main__":
    sentence_tokens = get_sentence_tokens("Data/Datasets/Conversation.csv")
    
    print(len(sentence_tokens))
    print(sentence_tokens[1000])

    
    