import sys
import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Data.data_preparation import get_sentence_tokens, word_index_mappings, read_embeddings
from Models.LSTM import LSTM


def build_prefix_index(id2tok):
    prefix_index = defaultdict(list)
    for tok in id2tok:
        for i in range(1, len(tok) + 1):
            prefix_index[tok[:i]].append(tok)
    return prefix_index


def evaluate_saved_keystrokes(model, sentences, k_full=3, k_char=3):
    model.eval()
    prefix_index = build_prefix_index(model.id2tok)
    
    # Count total possible keystrokes 
    total_keystrokes = sum(len(word) for sent in sentences for word in sent)
    saved_keystrokes = 0
    saved_keystrokes_per_sentence = []

    with torch.no_grad():
        for sent in tqdm(sentences, desc="Evaluating Sentences"):
            if not sent:
                saved_keystrokes_per_sentence.append(0)
                continue
            
            # full sentence excluding the last word (for prediction)
            input_tokens = ['<sos>'] + sent[:-1]
            input_ids = [model.tok2id.get(t, model.tok2id['<unk>']) for t in input_tokens]
            x = torch.tensor([input_ids], device=model.device)

            logits, _, _ = model(x)  # shape: (1, T, V)
            logits = logits.squeeze(0)  # shape: (T, V)
            probs = F.softmax(logits, dim=-1)

            saved_in_sentence = 0
            for i, word in enumerate(sent):
                word_probs = probs[i]  # shape: (V,)
                vocab_size = word_probs.shape[0]
                topk_full = torch.topk(word_probs, min(k_full, vocab_size))
                full_suggestions = [model.id2tok[idx.item()] for idx in topk_full.indices]

                if word in full_suggestions:
                    saved_keystrokes += len(word)
                    saved_in_sentence += len(word)
                    continue

                for j in range(1, len(word) + 1):
                    prefix = word[:j]
                    candidates = prefix_index.get(prefix, [])

                    filtered = [
                        (w, word_probs[model.tok2id.get(w, 0)].item())
                        for w in candidates if w != prefix and w in model.tok2id
                    ]

                    topk_prefix = sorted(filtered, key=lambda x: x[1], reverse=True)[:min(k_char, len(filtered))]

                    if word in [w for w, _ in topk_prefix]:
                        saved_keystrokes += len(word) - j
                        saved_in_sentence += len(word) - j
                        break
                    
            saved_keystrokes_per_sentence.append(saved_in_sentence)
    
    # total_keystrokes: what a user would have typed without suggestions
    # saved_keystrokes: how many keystrokes were actually avoided
    # saved_rate: % typing saved
    saved_rate = saved_keystrokes / total_keystrokes if total_keystrokes > 0 else 0.0
    return {
        "total_keystrokes": total_keystrokes,
        "saved_keystrokes": saved_keystrokes,
        "saved_rate": saved_rate,
        "avg_saved_per_sentence": sum(saved_keystrokes_per_sentence) / len(saved_keystrokes_per_sentence) if saved_keystrokes_per_sentence else 0,
        "avg_keystrokes_per_sentence": total_keystrokes / len(sentences),
    }


if __name__ == '__main__':
    tok2id, id2tok = word_index_mappings()
    embeddings = read_embeddings('Data/glove.6B.50d.txt', tok2id, 50)
    saved_model = torch.load('lstm_model_30_epochs_input50_numlayers2_hidden128_lr0.0001_batchsize4_L2_1e-6.pth')

    model = LSTM(
        input_size=50,
        hidden_size=128,
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
    result = evaluate_saved_keystrokes(model, sentences_to_test, k_full=3, k_char=3)
    print(result)
    print()
    print(f"Average total keystrokes per sentence: {result['avg_keystrokes_per_sentence']:.2f}")
    print(f"Average saved keystrokes per sentence: {result['avg_saved_per_sentence']:.2f}")
    print(f"Percentage saved per sentence: {100 * result['avg_saved_per_sentence'] / result['avg_keystrokes_per_sentence']:.2f}%")