from datasets import load_dataset
import sys
import os
import torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Data.data_preparation import (word_index_mappings, read_embeddings, get_sentence_tokens,
                                   FullSentenceDataset, numericalize_sentence, collate_full_sentences)
from nltk import word_tokenize
from torch.optim import Adam
from Models.LSTM import LSTM
import timeit

def main():
    input_size = 50
    hidden_size = 512
    num_layers = 2

    train_data = load_dataset('lm1b', split='train[:100000]', trust_remote_code=True)

    input_size = 50
    hidden_size = 512
    num_layers = 2

    device = torch.device(input('Enter device: cpu or mps: '))
    epochs = int(input('Enter number of epochs: '))

    train_sentences = train_data.map(
        lambda ex: {
            "tokens": ["<sos>"]
                      + word_tokenize(ex["text"].lower())
                      + ["<eos>"]
        },
        remove_columns=["text"]
    )


    train_sentences = train_sentences.filter(lambda ex: len(ex["tokens"]) <= 60)

    from collections import Counter

    # 1) Count all tokens
    ctr = Counter(w for ex in train_sentences for w in ex["tokens"])
    total_tokens = sum(ctr.values())

    # 2) Walk down the sorted list until we hit our coverage target
    coverage_target = 0.90
    cumulative = 0
    most_common = []
    for tok, freq in ctr.most_common():  # sorted descending by freq
        cumulative += freq
        most_common.append(tok)
        if cumulative / total_tokens >= coverage_target:
            break

    # 3) Build your vocab
    specials = ["<unk>", "<pad>"]
    id2tok = specials + most_common
    tok2id = {w: i for i, w in enumerate(id2tok)}

    output_size = len(id2tok)
    print(output_size)

    embeddings = read_embeddings('Data/glove.6B.50d.txt', tok2id, input_size)

    train_conv_sentences = get_sentence_tokens('Data/Datasets/conv_train.csv')
    val_conv_sentences = get_sentence_tokens('Data/Datasets/conv_val.csv')

    train_sentence_ids = [numericalize_sentence(s, tok2id, tok2id['<unk>']) for s in train_conv_sentences]
    train_ds = FullSentenceDataset(train_sentence_ids, tok2id['<pad>'])
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=16, shuffle=True, num_workers=4, pin_memory=True, collate_fn=collate_full_sentences
    )

    val_sentence_ids = [numericalize_sentence(s, tok2id, tok2id['<unk>']) for s in val_conv_sentences]
    val_ds = FullSentenceDataset(val_sentence_ids, tok2id['<pad>'])
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=2, pin_memory=True,
                                             collate_fn=collate_full_sentences)

    saved_model = torch.load('lstm_model_30_epochs_input50_numlayers2_hidden512_lr0.0001_batchsize64_L2_1e-6.pth')

    model = LSTM(input_size, hidden_size, num_layers, output_size, embeddings, tok2id, id2tok, device=device)
    model.load_state_dict(saved_model)

    model.train_model(train_loader, Adam(model.parameters(), lr=0.0001, weight_decay=1e-6),
                      torch.nn.CrossEntropyLoss(ignore_index=1, reduction='sum'),
                      epochs, val_loader, plot_training=False)

    torch.save(model.state_dict(), 'finetuned.pth')

    start = timeit.default_timer()
    print(model.predict(['are', 'you', 'going'], n_candidates=5))
    print(timeit.default_timer() - start)

    start = timeit.default_timer()
    print(model.complete_current_word(['are', 'you'], prefix='go', k=5))
    print(timeit.default_timer() - start)

if __name__ == '__main__':
    main()