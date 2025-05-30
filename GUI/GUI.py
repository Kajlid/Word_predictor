import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tkinter as tk
from Models.LSTM import LSTM
from Data.data_preparation import word_index_mappings, read_embeddings
import torch

class GUI:
    def __init__(self, model, nr_suggestions):
        self.model  = model
        self.nr_suggestions = nr_suggestions

        self.root = tk.Tk()
        self.root.title("Live Word Predictor")

        self.entry = tk.Entry(self.root, font=("Helvetica", 16))
        self.entry.pack(fill="x", padx=10, pady=10)
        self.entry.bind("<KeyRelease>", self.on_key)

        self.sugg_frame = tk.Frame(self.root)
        self.sugg_frame.pack(fill="x", padx=10)

        self.sugg_buttons = []
        self.current_suggestions = []


        for _ in range(nr_suggestions):
            btn = tk.Button(self.sugg_frame, text="", font=("Helvetica", 14),
                            command=lambda b=_: self.on_suggest(b))
            btn.pack(side="left", padx=5)
            self.sugg_buttons.append(btn)

        self.update_suggestions()


    def split_input(self, text):
        tokens = text.strip().split()
        if text.endswith(" "):
            return tokens, ""
        elif tokens:
            return tokens[:-1], tokens[-1]
        else:
            return [], ""


    def update_suggestions(self):
        text = self.entry.get().lower()
        context, prefix = self.split_input(text)


        if prefix == "":
            suggs = self.model.predict(context, self.nr_suggestions)
        else:
            suggs = self.model.complete_current_word(context, prefix, self.nr_suggestions)

        specials = {'<pad>', '<unk>', '<sos>', '<eos>'}
        suggs = [(w, p) for (w, p) in suggs if w not in specials]

        # Only take the words for display
        display_words = [w for w, _ in suggs]

        # Update buttons
        for i, btn in enumerate(self.sugg_buttons):
            if i < len(suggs):
                btn.config(text=display_words[i])
                btn.pack(side="left", padx=5)
            else:
                btn.pack_forget()

        self.current_suggestions = display_words

    def on_key(self, event=None):
        self.update_suggestions()

    def on_suggest(self, idx):
        word = self.current_suggestions[idx]
        text = self.entry.get().lower()

        # Remove any partial prefix
        if not text.endswith(" ") and " " in text:
            text = text[:text.rfind(" ")+1]
        elif not text.endswith(" "):
            text = ""

        # Insert the chosen word + a space
        new_text = text + word + " "
        self.entry.delete(0, tk.END)
        self.entry.insert(0, new_text)

        # Refresh suggestions for the new text
        self.update_suggestions()

    def run(self):
        self.root.mainloop()

# Usage:
# from your model module import LSTM
# model = LSTM(...); model.load_state_dict(...); model.to(device)
# app = WordPredictorApp(model)
# app.run()

if __name__ == "__main__":
    tok2id, id2tok = word_index_mappings()
    embeddings = read_embeddings('Data/glove.6B.50d.txt', tok2id, 50)
    saved_model = torch.load('lstm_model_30_epochs_input50_numlayers2_hidden128_lr0.0001_batchsize4.pth')

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

    app = GUI(model, 3)

    app.run()