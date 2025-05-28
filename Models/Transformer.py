import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from nltk.tokenize import word_tokenize
import math
import random
from tqdm import tqdm

# ======================= Dataset ======================= #

class WordDataset(Dataset):
    def __init__(self, text_file, maxlen=16):
        with open(text_file, encoding='utf-8') as f:
            tokens = word_tokenize(f.read().lower())
        self.maxlen = maxlen
        self.vocab = sorted(set(tokens))
        self.word_to_id = {w: i+1 for i, w in enumerate(self.vocab)}  # 0 for PAD
        self.id_to_word = {i+1: w for i, w in enumerate(self.vocab)}
        self.pad_id = 0

        self.X = []
        self.y = []
        for i in range(len(tokens) - maxlen):
            input_seq = tokens[i:i+maxlen]
            label = tokens[i+maxlen]
            self.X.append([self.word_to_id.get(w, 0) for w in input_seq])
            self.y.append(self.word_to_id.get(label, 0))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])

# ======================= Transformer Blocks ======================= #

class SelfAttention(nn.Module):
    def __init__(self, vector_dim, att_dim):
        super().__init__()
        assert vector_dim % att_dim == 0
        self.no_of_heads = vector_dim // att_dim
        self.vector_dim = vector_dim
        self.att_dim = att_dim
        self.wq = nn.Linear(att_dim, att_dim, bias=False)
        self.wk = nn.Linear(att_dim, att_dim, bias=False)
        self.wv = nn.Linear(att_dim, att_dim, bias=False)
        self.wo = nn.Linear(vector_dim, vector_dim, bias=False)

    def compute_attention(self, q, k, v):
        dk = k.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(dk)
        alpha = torch.softmax(scores, dim=-1)
        return torch.matmul(alpha, v)

    def reshape_for_multihead(self, x):
        batch_size, seq_length, vector_dim = x.size()
        x = x.view(batch_size, seq_length, self.no_of_heads, self.att_dim)
        return x.permute(0, 2, 1, 3)

    def reshape_back(self, x):
        batch_size, no_of_heads, seq_length, att_dim = x.size()
        x = x.permute(0, 2, 1, 3)
        return x.reshape(batch_size, seq_length, no_of_heads * att_dim)

    def forward(self, x):
        x = self.reshape_for_multihead(x)
        q, k, v = self.wq(x), self.wk(x), self.wv(x)
        att = self.compute_attention(q, k, v)
        att = self.reshape_back(att)
        return self.wo(att)

class PositionwiseFFN(nn.Module):
    def __init__(self, hidden_size, dropout):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size)
        )

    def forward(self, x):
        return self.ffn(x)

class EncoderBlock(nn.Module):
    def __init__(self, vector_dim, att_dim, dropout):
        super().__init__()
        self.attn = SelfAttention(vector_dim, att_dim)
        self.ffn = PositionwiseFFN(vector_dim, dropout)
        self.ln1 = nn.LayerNorm(vector_dim)
        self.ln2 = nn.LayerNorm(vector_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x2 = x + self.dropout(self.attn(self.ln1(x)))
        x3 = x2 + self.dropout(self.ffn(self.ln2(x2)))
        return x3

# ======================= Model ======================= #

class Transformer(nn.Module):
    def __init__(self, config, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, config.vector_dim, padding_idx=0)
        self.positional = nn.Parameter(torch.randn(1, config.maxlen, config.vector_dim))
        self.transformers = nn.ModuleList([
            EncoderBlock(config.vector_dim, config.att_dim, config.dropout_prob)
            for _ in range(config.num_layers)
        ])
        self.final = nn.Linear(config.vector_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x) + self.positional[:, :x.size(1), :]
        for layer in self.transformers:
            x = layer(x)
        last_token = x[:, -1, :]
        return self.final(last_token)

# ======================= Config ======================= #

class Config:
    vector_dim = 64
    att_dim = 64
    dropout_prob = 0.1
    num_layers = 2
    batch_size = 32
    learning_rate = 0.0001
    num_epochs = 30
    maxlen = 16

# ======================= Training ======================= #

def train_model():
    device = torch.device("mps" if torch.backends.mps.is_available() else
                          "cuda" if torch.cuda.is_available() else "cpu") 
    config = Config()
    dataset = WordDataset("Data/Datasets/conv_train.csv", config.maxlen)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    val_dataset = WordDataset("Data/Datasets/conv_val.csv", config.maxlen)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    vocab_size = len(dataset.word_to_id) + 1
    model = Transformer(config, vocab_size=vocab_size)
    model.to(device)
    model.embedding.weight.data = model.embedding.weight.data.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    best_val_loss = float('inf')
    patience = 2  
    epochs_no_improve = 0

    for epoch in tqdm(range(config.num_epochs), desc="Training Progress"):
        model.train()
        total_loss = 0
        for x_batch, y_batch in dataloader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_train_loss = total_loss / len(dataloader)
        val_loss = evaluate(model, val_dataloader, criterion, device)

        print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # For early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
        
            torch.save({
                'model_state_dict': model.state_dict(),
                'vocab': dataset.word_to_id,
                'config': vars(config)
            }, f"transformer_epoch{epoch+1}_val{val_loss:.2f}_input64_numlayers2_attdim64_lr0.0001_batchsize32_maxlen16_dropout0.1.pth")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
    
    
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            total_loss += loss.item()
    return total_loss / len(dataloader)



if __name__ == "__main__":
    train_model()
