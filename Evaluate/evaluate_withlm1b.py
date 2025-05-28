from test_predictions import evaluate_saved_keystrokes_lstm
import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Data.data_preparation import get_sentence_tokens, word_index_mappings, read_embeddings
from Models.LSTM import LSTM
from nltk import word_tokenize
from datasets import load_dataset
from collections import Counter

def main():


    saved_model = torch.load('finetuned.pth')
    train_data = load_dataset('lm1b', split='train[:100000]', trust_remote_code=True)


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

    embeddings = read_embeddings('Data/glove.6B.50d.txt', tok2id, 50)
    model = LSTM(
        input_size=50,
        hidden_size=512,
        num_layers=2,
        output_size=len(id2tok),
        embeddings=embeddings,
        tok2id=tok2id,
        id2tok=id2tok,
        device=torch.device('mps'),
    )

    model.load_state_dict(saved_model)
    sentences_to_test = get_sentence_tokens('Data/Datasets/conv_val.csv')
    num_words = sum(len(s) for s in sentences_to_test)
    print(f"Number of words in validation set: {num_words}")
    result = evaluate_saved_keystrokes_lstm(model, sentences_to_test, k_full=3, k_char=3)
    print("LSTM performance:")
    print(result)
    print(f"Average total keystrokes per sentence: {result['avg_keystrokes_per_sentence']:.2f}")
    print(f"Average saved keystrokes per sentence: {result['avg_saved_per_sentence']:.2f}")
    print(
        f"Percentage saved per sentence: {100 * result['avg_saved_per_sentence'] / result['avg_keystrokes_per_sentence']:.2f}%")

if __name__ == '__main__':
    main()