import torch
import torch.nn.functional as F

class LSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, embeddings, tok2id, id2tok, device=torch.device('cpu')):
        super(LSTM, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, device=device)
        self.fc = torch.nn.Linear(hidden_size, output_size, device=device)

        if embeddings is None:
            self.embedding = torch.nn.Embedding(output_size, input_size, padding_idx=1, device=device)
        else:
            self.embedding = torch.nn.Embedding.from_pretrained(embeddings, freeze=False, padding_idx=1).to(self.device)
        self.tok2id = tok2id
        self.id2tok = id2tok

    def init_hidden(self, batch_size):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size, device=self.device),
                torch.zeros(self.num_layers, batch_size, self.hidden_size, device=self.device))


    def forward(self, x, h_prev=None, c_prev=None):
        x = x.to(self.device)
        if h_prev is None or c_prev is None:
            h_prev, c_prev = self.init_hidden(x.size(0))
        emb = self.embedding(x)
        logits, (h_prev, c_prev) = self.lstm(emb, (h_prev, c_prev))
        logits = self.fc(logits.reshape(-1, self.hidden_size))
        return logits, h_prev, c_prev

    def train_model(self, train_loader, optimizer, criterion, epochs):
        self.train()
        for epoch in range(epochs):
            for i, (x, y) in enumerate(train_loader):
                x, y = x.to(self.device), y.to(self.device)

                optimizer.zero_grad()
                logits, _, _ = self(x)

                loss = criterion(logits, y.view(-1))
                loss.backward()
                optimizer.step()
                if i % 100 == 0:
                    print(f'Epoch: {epoch+1}/{epochs}, Step: {i+1}/{len(train_loader)}, Loss: {loss.item():.4f}')

    def predict(self, tokens, k=None):
        self.eval()
        #prime with context tokens
        h, c = self.init_hidden(1)
        with torch.no_grad():
            for t in tokens:
                # carry over h and c here
                idx = self.tok2id.get(t, self.tok2id['<unk>'])
                x = torch.tensor([[idx]], device=self.device)
                logits, h, c = self(x, h, c)

            probs = F.softmax(logits, dim=-1).squeeze(0)  # (V)

        if k is None:
            k = probs.size(0)
        topk = torch.topk(probs, k)
        return [(self.id2tok[idx.item()], topk.values[i].item())
                for i, idx in enumerate(topk.indices)]

    def complete_current_word(self, context_tokens, prefix, k=5):

        # P(w | context) for entire vocab
        probs = dict(self.predict(context_tokens, k=None))

        candidates = [(w, p) for w, p in probs.items() if w.startswith(prefix) and w != prefix]

        candidates.sort(key=lambda cand: cand[1], reverse=True)

        return candidates[:k]