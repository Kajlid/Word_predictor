import torch
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Models.LSTM import LSTM
from Data.data_preparation import FullSentenceDataset, get_sentence_tokens, collate_full_sentences, numericalize_sentence
from torch.optim import Adam

def main():
    input_size = 50
    hidden_size = 128
    num_layers = 2
    output_size = 1
    embeddings = None

    device = torch.device(input('Enter device: cpu or mps: '))
    epochs = int(input('Enter number of epochs: '))

    sentences = get_sentence_tokens('Data/Datasets/Conversation.csv')
    id2tok = ['<unk>', '<pad>']
    tok2id = {'<unk>': 0, '<pad>': 1}
    max_sentence_length = 0
    for s in sentences:
        max_sentence_length = max(max_sentence_length, len(s))
        for t in s:
            if t not in tok2id:
                tok2id[t] = len(id2tok)
                id2tok.append(t)

    sentence_ids = [numericalize_sentence(s, tok2id, tok2id['<unk>']) for s in sentences]
    ds = FullSentenceDataset(sentence_ids, tok2id['<pad>'], device=device)
    train_loader = torch.utils.data.DataLoader(
        ds, batch_size=4, shuffle=True, num_workers=4, pin_memory=True, collate_fn=collate_full_sentences
    )

    model = LSTM(input_size, hidden_size, num_layers, output_size, embeddings, tok2id, device=device)

    model.train_model(train_loader, Adam(model.parameters()), torch.nn.CrossEntropyLoss(), epochs)

    model.loss_plot()

if __name__ == "__main__":
    main()