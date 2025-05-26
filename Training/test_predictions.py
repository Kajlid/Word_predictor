import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Data.data_preparation import get_sentence_tokens, word_index_mappings, read_embeddings
from Models.LSTM import LSTM
import torch


def evaluate_saved_keystrokes(model: LSTM, sentences, k_full=3, k_char=3):
    total_keystrokes = 0
    saved_keystrokes = 0

    for sent in sentences:   # for every list of words
        context = []
        for word in sent:
            if len(context) == 0:
                vocab = [w for w in model.tok2id.keys()]
                full_suggestions = [w for w, _ in model.predict(vocab, k_full)]
            else:
                full_suggestions = [w for w, _ in model.predict(context, k_full)]
            if word in full_suggestions:
                # user clicks immediately ( one click to select suggestion so - 1 )
                saved_keystrokes += len(word) - 1
                
                context.append(word)
                continue

            for i in range(1, len(word) + 1):
                total_keystrokes += 1
                prefix = word[:i]
                
                char_suggestions = [w for w, _ in model.complete_current_word(context, prefix, k_char)]
                if word in char_suggestions:
                    # click on suggestion
                    saved_keystrokes += (len(word) - i) - 1
                    break
            
            context.append(word)

    saved_rate = saved_keystrokes / total_keystrokes if total_keystrokes > 0 else 0.0
    return {
        "total_keystrokes": total_keystrokes,
        "saved_keystrokes": saved_keystrokes,
        "saved_rate": saved_rate,
    }

if __name__ == '__main__':
    # Load saved model and sentences from validation set
    tok2id, id2tok = word_index_mappings()
    
    embeddings = read_embeddings('Data/glove.6B.50d.txt', tok2id, 50)
    saved_model = torch.load('lstm_model_30_epochs_input50_numlayers2_hidden128_lr0.0001_batchsize4.pth')
    model = LSTM(input_size=50, hidden_size=128, num_layers=2, output_size=len(id2tok), embeddings=embeddings, tok2id=tok2id, id2tok=id2tok, device=torch.device('mps'))
    
    model.load_state_dict(saved_model)
    sentences_to_test = get_sentence_tokens('Data/Datasets/conv_val.csv')
    print(evaluate_saved_keystrokes(model, sentences_to_test, k_full=3, k_char=3))