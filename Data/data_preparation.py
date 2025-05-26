import pandas as pd
import numpy as np
import random
import csv
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torch
from nltk.tokenize import word_tokenize

def create_dataframe(filename):
    df = pd.read_csv(filename, skiprows=0)
    
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    
    # Concatenate questions and answers vertically (for creating train and val data)
    stacked_df = pd.concat([df['question'], df['answer']]).dropna().reset_index(drop=True)
    
    return stacked_df


def get_sentence_tokens(filename):
    """ Get a list of sentences as lists of tokens for n-gram.

    Args:
        filename (string): path to raw text dataset.

    Returns:
        sentence_tokens (list): list of lists of tokens (list of sentences).
    """
    sentence_tokens = []
    with open(filename, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if row: 
                sentence = row[0]
                tokens = word_tokenize(sentence)
                sentence_tokens.append(['<sos>'] + tokens + ['<eos>'])
    return sentence_tokens


def word_index_mappings(file_path ='Data/Datasets/conv_train.csv'):
    train_sentences = get_sentence_tokens(file_path)
    id2tok = ['<unk>', '<pad>']
    tok2id = {'<unk>': 0, '<pad>': 1}
    max_sentence_length = 0
    for s in train_sentences:
        max_sentence_length = max(max_sentence_length, len(s))
        for t in s:
            if t not in tok2id:
                tok2id[t] = len(id2tok)
                id2tok.append(t)

    return tok2id, id2tok


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

def write_sentences(path, data):
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            for sentence in data:
                writer.writerow([''.join(sentence)])

def train_val_split(filename, train_ratio=0.95, seed=42, train_out="conv_train.csv", val_out="conv_val.csv"):
    sentences = create_dataframe(filename)
            
    random.seed(seed)
    random.shuffle(sentences)

    split_idx = int(len(sentences) * train_ratio)
    train_sentences = sentences[:split_idx]
    val_sentences = sentences[split_idx:]

    write_sentences(train_out, train_sentences)
    write_sentences(val_out, val_sentences)

    print(f"Train size: {len(train_sentences)}")
    print(f"Validation size: {len(val_sentences)}")
    
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
    sentence_tokens = get_sentence_tokens("Data/Datasets/conv_train.csv")
    
    print(len(sentence_tokens))
    print(sentence_tokens[1000])
    
    #train_val_split("Data/Datasets/Conversation.csv", train_ratio=0.95, seed=42, train_out="Data/Datasets/conv_train.csv", val_out="Data/Datasets/conv_val.csv")
    

    
    