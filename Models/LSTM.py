import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

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

        if h_prev is None or c_prev is None:
            h_prev, c_prev = self.init_hidden(x.size(0))
        emb = self.embedding(x)
        logits, (h_prev, c_prev) = self.lstm(emb, (h_prev, c_prev))
        logits = self.fc(logits.reshape(-1, self.hidden_size))
        return logits, h_prev, c_prev

    def train_model(self, train_loader, optimizer, criterion, epochs, val_loader = None, plot_training=False):
        if plot_training:
            # For continuous plotting
            plt.ion()
            fig, ax = plt.subplots()
        
        train_losses = []
        val_losses = []
        
        best_val_loss = float('inf')
        patience = 2
        epochs_no_improve = 0
        

        for epoch in range(epochs):
            self.train()
            total_loss = 0
            total_tokens = 0
            for i, (x, y) in enumerate(train_loader):
                x, y = x.to(self.device), y.to(self.device)

                optimizer.zero_grad()
                logits, _, _ = self(x)

                loss = criterion(logits, y.view(-1))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                tokens_this_batch = (y.view(-1) != 1).sum().item()
                total_tokens += tokens_this_batch
                if i % 1000 == 0:
                    print(f'Epoch: {epoch+1}/{epochs}, Step: {i+1}/{len(train_loader)}, Loss: {loss.item()/tokens_this_batch:.4f}')
                    
            avg_train_loss = total_loss / total_tokens
            train_losses.append(avg_train_loss)
            
            if val_loader:
                self.eval()
                with torch.no_grad():
                    val_loss = 0
                    val_tokens = 0
                    for x_val, y_val in val_loader:
                        x_val, y_val = x_val.to(self.device), y_val.to(self.device)

                        logits, _, _ = self(x_val)
                        loss = criterion(logits, y_val.view(-1))
                        
                        val_loss += loss.item()
                        val_tokens += (y_val.view(-1) != 1).sum().item()
                    avg_val_loss = val_loss / val_tokens
                    val_losses.append(avg_val_loss)
                    print(
                        f'Epoch: {epoch + 1}/{epochs}, Validation Loss: {avg_val_loss:.4f}, Train Loss: {avg_train_loss:.4f} \n')

                    # Early stopping
                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        epochs_no_improve = 0
                    else:
                        epochs_no_improve += 1
                        if epochs_no_improve >= patience:
                            print("Early stopping triggered.")
                            break
            
            if plot_training:                     
                # Update plot
                ax.clear()
                ax.plot(train_losses, label="Train Loss", color='blue')
                if val_loader:
                    ax.plot(val_losses, label="Validation Loss", color='red')
                ax.set_xlabel("Epoch")
                ax.set_ylabel("Loss")
                ax.set_title("Loss Curve")
                ax.legend()
                ax.grid(True)
                plt.pause(0.1)
                # plt.show()

    def _prime_model(self, context_tokens):

        if not context_tokens:
            context_tokens = ['<sos>']

        h, c = self.init_hidden(1)
        for t in context_tokens:
            # carry over h and c here
            idx = self.tok2id.get(t, self.tok2id['<unk>'])
            x = torch.tensor([[idx]], device=self.device)
            logits, h, c = self(x, h, c)

        return logits, h, c

    def predict(self, context_tokens, n_candidates=None):
        self.eval()
        with torch.no_grad():
            logits, h, c = self._prime_model(context_tokens)

            probs = F.softmax(logits, dim=-1).squeeze(0)  # (V)

            if n_candidates is None:
                n_candidates = probs.size(0)
            topk = torch.topk(probs, n_candidates)
            return [(self.id2tok[idx.item()], topk.values[i].item())
                    for i, idx in enumerate(topk.indices)]

    def complete_current_word(self, context_tokens, prefix, k=5):
        
        if not context_tokens:
            context_tokens = ['<sos>']

        with torch.no_grad():

            logits, h, c = self._prime_model(context_tokens)
            probs = F.softmax(logits.squeeze(0), dim=-1)

            mask = torch.zeros_like(probs, dtype=torch.bool)
            for id, tok in enumerate(self.id2tok):
                if tok.startswith(prefix) and tok != prefix:
                    mask[id] = True


            NEG_INF = -1e9
            # the ~ flips the boolean tensor.
            masked = probs.masked_fill(~mask, NEG_INF)  # [V]

            top_vals, top_ids = torch.topk(masked, min(k, mask.sum().item()))

        return [(self.id2tok[i], top_vals[j].item())
                for j, i in enumerate(top_ids)]


