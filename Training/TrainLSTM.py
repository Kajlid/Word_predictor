import torch
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Models.LSTM import LSTM
from Data.data_preparation import (FullSentenceDataset, get_sentence_tokens,
                                   collate_full_sentences, numericalize_sentence,
                                   read_embeddings)
from torch.optim import Adam
import timeit

def main():
    input_size = 50
    hidden_size = 128
    num_layers = 2

    device = torch.device(input('Enter device: cpu or mps: '))
    epochs = int(input('Enter number of epochs: '))

    train_sentences = get_sentence_tokens('Data/Datasets/conv_train.csv')
    id2tok = ['<unk>', '<pad>']
    tok2id = {'<unk>': 0, '<pad>': 1}
    max_sentence_length = 0
    for s in train_sentences:
        max_sentence_length = max(max_sentence_length, len(s))
        for t in s:
            if t not in tok2id:
                tok2id[t] = len(id2tok)
                id2tok.append(t)
                
                
    val_sentences = get_sentence_tokens('Data/Datasets/conv_val.csv')

    output_size = len(id2tok)
    embeddings = read_embeddings('Data/glove.6B.50d.txt', tok2id, input_size)
    train_sentence_ids = [numericalize_sentence(s, tok2id, tok2id['<unk>']) for s in train_sentences]
    train_ds = FullSentenceDataset(train_sentence_ids, tok2id['<pad>'])
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=4, shuffle=True, num_workers=4, pin_memory=True, collate_fn=collate_full_sentences
    )
    
    val_sentence_ids = [numericalize_sentence(s, tok2id, tok2id['<unk>']) for s in val_sentences]
    val_ds = FullSentenceDataset(val_sentence_ids, tok2id['<pad>'])
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=4, shuffle=True, num_workers=4, pin_memory=True, collate_fn=collate_full_sentences
    )

    model = LSTM(input_size, hidden_size, num_layers, output_size, embeddings, tok2id, id2tok, device=device)

    model.train_model(train_loader, Adam(model.parameters(), lr=0.0001), torch.nn.CrossEntropyLoss(), epochs, val_loader)
    file_path = 'lstm_model_30_epochs_input50_numlayers2_hidden128_lr0.0001_batchsize4.pth'
    torch.save(model.state_dict(), file_path)
    print(f"Model saved to {file_path}")
    start = timeit.default_timer()
    print(model.predict(['are', 'you', 'going'], n_candidates=5))
    print(timeit.default_timer() - start)


    start = timeit.default_timer()
    print(model.complete_current_word(['are', 'you'], prefix='go',  k=5))
    print(timeit.default_timer() - start)

if __name__ == "__main__":
    main()