import pandas as pd
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
    conv_data = create_dataframe(filename)
    sentence_tokens = conv_data.apply(word_tokenize)     # Lists of tokens
    
    token_arr = sentence_tokens.to_numpy()
    
    return token_arr


def word_index_mappings():
    pass


def read_embeddings():
    pass
    
class FullSentenceDataset(Dataset):
    """
    Each sample is one sentence:
      x: all tokens except the last
      y: all tokens except the first (i.e. next-word targets)
    """
    def __init__(self, sentences, pad_idx, device='cpu'):
        """
        sentences: List[List[int]]  # each is a numericalized sentence
        pad_idx:   int              # index of your padding token
        """
        self.sents = [s for s in sentences if len(s) > 1]
        self.pad_idx = pad_idx
        self.device = device

    def __len__(self):
        return len(self.sents)

    def __getitem__(self, idx):
        s = self.sents[idx]
        # input is everything but the last token
        x = torch.tensor(s[:-1], dtype=torch.long, device=self.device)
        # target is everything but the first token
        y = torch.tensor(s[1:],  dtype=torch.long, device=self.device)
        return x, y

def numericalize_sentence(sent, tok2id, unk_idx):
    return [tok2id.get(tok, unk_idx) for tok in sent]

def collate_full_sentences(batch):
    """
    batch: list of (x, y) pairs where
      x: [L_i]  (LongTensor)
      y: [L_i]  (LongTensor)
    Returns:
      x_padded: [B, L_max]
      y_padded: [B, L_max]
      lengths:  [B] (lengths of each original x)
    """
    xs, ys = zip(*batch)
    lengths = torch.tensor([x.size(0) for x in xs], dtype=torch.long)
    x_padded = pad_sequence(xs, batch_first=True, padding_value=1)
    y_padded = pad_sequence(ys, batch_first=True, padding_value=1)
    return x_padded, y_padded

if __name__== "__main__":
    sentence_tokens = get_sentence_tokens("Data/Datasets/Conversation.csv")
    
    print(sentence_tokens)

    
    