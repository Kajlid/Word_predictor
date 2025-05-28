import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tkinter as tk
from Models.LSTM import LSTM
from Data.data_preparation import word_index_mappings, read_embeddings
import torch
from datasets import load_dataset
from nltk import word_tokenize
import tkinter as tk
import torch.nn.functional as F

class SMSPredictorApp:
    def __init__(self, model, nr_suggestions=3):
        self.model  = model
        self.nr_suggestions = nr_suggestions

        # Main window
        self.root = tk.Tk()
        self.root.title("SMS Word Predictor")
        self.root.geometry("360x640")  # phone-like aspect

        # Chat history (scrollable)
        self.chat_frame = tk.Frame(self.root, bg="#ECE5DD")
        self.chat_frame.pack(fill="both", expand=True)
        self.canvas    = tk.Canvas(self.chat_frame, bg="#ECE5DD", highlightthickness=0)
        self.scrollbar = tk.Scrollbar(self.chat_frame, orient="vertical", command=self.canvas.yview)
        self.inner     = tk.Frame(self.canvas, bg="#ECE5DD")
        self.inner.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        self.canvas.create_window((0,0), window=self.inner, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        self.entry      = tk.Entry(self.root, font=("Helvetica", 16))
        self.entry.pack(fill="x", padx=8, pady=(4,0))
        self.entry.bind("<KeyRelease>", self.on_key)

        self.sugg_frame = tk.Frame(self.root, bg="#f0f0f0")
        self.sugg_frame.pack(fill="x", padx=8, pady=(0,8))


        self.buttons = []
        for i in range(self.nr_suggestions):
            b = tk.Button(self.sugg_frame,
                          font=("Helvetica", 14),
                          command=lambda j=i: self.on_suggest(j))
            b.pack(side="left", padx=4)
            self.buttons.append(b)

        self.current_suggestions = []

    def split_input(self, text):
        toks = text.strip().split()
        if text.endswith(" "):
            return toks, ""
        elif toks:
            return toks[:-1], toks[-1]
        else:
            return [], ""

    def update_suggestions(self):
        text = self.entry.get().lower()
        context, prefix = self.split_input(text)

        if prefix == "":
            raw = self.model.predict(context, self.nr_suggestions)
        else:
            raw = self.model.complete_current_word(context, prefix, self.nr_suggestions)

        # filter out special tokens
        specials = {'<pad>','<unk>','<sos>','<eos>'}
        suggs = [w for w,_ in raw if w not in specials]


        for i, b in enumerate(self.buttons):
            if i < len(suggs):
                b.config(text=suggs[i])
                b.pack(side="left", padx=4)
            else:
                b.pack_forget()

        self.current_suggestions = suggs

    def add_message(self, text, user=True):
        bubble = tk.Label(self.inner, text=text,
                          wraplength=240, justify="left",
                          font=("Helvetica",12),
                          padx=10, pady=6, bd=0)
        if user:
            bubble.config(bg="#32a852")
            bubble.pack(anchor="e", pady=2, padx=6)
        else:
            bubble.config(bg="#FFFFFF")
            bubble.pack(anchor="w", pady=2, padx=6)

        self.canvas.update_idletasks()
        self.canvas.yview_moveto(1.0)

    def on_key(self, event=None):
        if event and event.keysym == "Return":
            msg = self.entry.get().strip()
            if msg:
                self.add_message(msg, user=True)
                self.entry.delete(0, tk.END)
            return
        self.update_suggestions()

    def on_suggest(self, idx):
        word = self.current_suggestions[idx]
        text = self.entry.get().lower()

        if not text.endswith(" ") and " " in text:
            text = text[:text.rfind(" ")+1]
        elif not text.endswith(" "):
            text = ""

        new_text = text + word + " "
        self.entry.delete(0, tk.END)
        self.entry.insert(0, new_text)


        #self.add_message(word, user=True)
        self.update_suggestions()

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":

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
    saved_model = torch.load('finetuned.pth')

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

    app = SMSPredictorApp(model, 3)

    app.run()