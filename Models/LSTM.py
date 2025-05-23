import torch

class LSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, embeddings, tok2id, device=torch.device('cpu')):
        super(LSTM, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, device=device)
        self.fc = torch.nn.Linear(hidden_size, output_size, device=device)

        if embeddings is None:
            self.embeddings = torch.nn.Embedding(len(tok2id), input_size, padding_idx=1)
        else:
            self.embeddings = torch.nn.Embedding.from_pretrained(embeddings, freeze=False, padding_idx=1)
        self.tok2id = tok2id


    def forward(self, x, h_prev=None, c_prev=None):
        emb = self.embeddings(x)
        logits, h, c = self(emb, h_prev, c_prev)
        logits = self.fc(logits.reshape(-1, self.hidden_size))
        return logits, h, c

    def train_model(self, train_loader, optimizer, criterion, epochs):
        self.train()
        for epoch in range(epochs):
            for i, (x, y) in enumerate(train_loader):
                x, y = x.to(self.device), y.to(self.device)

                optimizer.zero_grad()
                logits, _, _ = self(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
                if i % 100 == 0:
                    print(f'Epoch: {epoch+1}/{epochs}, Step: {i+1}/{len(train_loader)}, Loss: {loss.item():.4f}')

