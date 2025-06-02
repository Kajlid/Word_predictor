import tkinter as tk
import torch.nn.functional as F
import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tkinter as tk
from tkinter import ttk
from Models.LSTM import LSTM
from Models.Ngram import Ngram
from Data.data_preparation import word_index_mappings, read_embeddings, get_sentence_tokens

class SMSPredictorApp:
    def __init__(self, lstm_model, ngram_model=None, ngram_counts=None, vocab=None, nr_suggestions=3):
        self.lstm_model = lstm_model
        self.ngram_model = ngram_model
        self.ngram_counts = ngram_counts
        self.vocab = vocab
        self.nr_suggestions = nr_suggestions

        self.total_keystrokes = 0
        self.saved_keystrokes = 0
        # Main window
        self.root = tk.Tk()
        self.root.title("SMS Word Predictor")
        self.root.geometry("360x640")  # phone-like aspect
        
        # Dropdown model choice
        dropdown_frame = tk.Frame(self.root, bg="#ECE5DD")
        dropdown_frame.pack(fill="x", padx=8, pady=5)

        self.model_var = tk.StringVar(value="Choose a model")
        self.model_dropdown = ttk.OptionMenu(
            dropdown_frame,
            self.model_var,
            "Choose a model",
            "LSTM",
            "Bigram",
            "Trigram",
            command=lambda _: self.on_model_change()  # trigger logic when changed
        )
        self.model_dropdown.pack(side="right")

        # Chat history (scrollable)
        self.chat_frame = tk.Frame(self.root, bg="#ECE5DD")
        self.chat_frame.pack(fill="both", expand=True)
        self.canvas = tk.Canvas(self.chat_frame, bg="#ECE5DD", highlightthickness=0)
        self.scrollbar = tk.Scrollbar(self.chat_frame, orient="vertical", command=self.canvas.yview)
        self.inner = tk.Frame(self.canvas, bg="#ECE5DD")
        self.inner.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        self.canvas.create_window((0,0), window=self.inner, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        self.entry = tk.Entry(self.root, font=("Helvetica", 16), fg="grey")
        self.entry.pack(fill="x", padx=8, pady=(4,0))
        self.entry.bind("<KeyRelease>", self.on_key)
        self.entry.insert(0, "Write something here")
        self.entry.bind("<FocusIn>", self.clear_placeholder)
        self.entry.bind("<FocusOut>", self.add_placeholder)
        self.placeholder_text = "Write something here"
        self.placeholder_active = True

        self.sugg_frame = tk.Frame(self.root, bg="#f0f0f0")

        self.status_bar = tk.StringVar(value="Saved 0/0  (0.00%)")
        status_label = tk.Label(self.root, textvariable=self.status_bar,
                                font=("Helvetica", 10), anchor="w")
        status_label.pack(fill="x", side="bottom", padx=8, pady=4)

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
        
    def clear_placeholder(self, event=None):
        if self.placeholder_active:
            self.entry.delete(0, tk.END)
            self.entry.config(fg="black")
            self.placeholder_active = False

    def add_placeholder(self, event=None):
        if not self.entry.get():
            self.entry.insert(0, self.placeholder_text)
            self.entry.config(fg="grey")
            self.placeholder_active = True

    def update_status(self):
        if self.total_keystrokes > 0:
            rate = 100 * self.saved_keystrokes / (self.total_keystrokes + self.saved_keystrokes)
        else:
            rate = 0.0
        self.status_bar.set(
            f"Saved {self.saved_keystrokes}/{self.total_keystrokes + self.saved_keystrokes}  ({rate:.2f}%)"
        )

    def update_suggestions(self):
        if self.model_var.get() == "Choose a model":
            return   # skip suggestion update if a model is not selected
        
        text = self.entry.get().lower()
        # Treat placeholder as empty input
        if self.placeholder_active or text == self.placeholder_text.lower():
            text = ""  
            
        context, prefix = self.split_input(text)
        
        specials = {'<pad>','<unk>','<sos>','<eos>'}
        selected_model = self.model_var.get()

        if selected_model == "LSTM":
            if prefix == "":
                raw = self.lstm_model.predict(context, self.nr_suggestions, filter_specials=True)
            else:
                raw = self.lstm_model.complete_current_word(context, prefix, self.nr_suggestions)

            # filter out special tokens
            suggs = [w for w,_ in raw if w not in specials]
            
        elif selected_model in ("Bigram", "Trigram"):
            ngram_level = 2 if selected_model == "Bigram" else 3
            if prefix == "":
                raw = self.ngram_model.get_top_suggestions(context, self.ngram_counts[:ngram_level], self.vocab, k=self.nr_suggestions)
            else:
               raw = self.ngram_model.get_top_suggestions(context, self.ngram_counts[:ngram_level], self.vocab, k=self.nr_suggestions, started_word=prefix)
                
            # filter out special tokens
            suggs = [w for w in raw if w not in specials]
            
        else:
            suggs = []


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


    def on_model_change(self):
        if self.model_var.get() == "Choose a model":
            for b in self.buttons:
                b.config(state="disabled", text="")
                
            self.sugg_frame.pack_forget()
        else:
            for b in self.buttons:
                b.config(state="normal")
            self.sugg_frame.pack(fill="x", padx=8, pady=(0, 8)) 
            self.update_suggestions()

    def on_key(self, event=None):
        if self.placeholder_active:
            return

        if event and len(event.char) == 1 and event.keysym != "Return":
            self.total_keystrokes += 1
            self.update_status()

        if event and event.keysym == "Return":
            msg = self.entry.get().strip()
            if msg:
                self.add_message(msg, user=True)
                self.entry.delete(0, tk.END)
                self.update_suggestions()
            return
        self.update_suggestions()

    def on_suggest(self, idx):
        word = self.current_suggestions[idx]
        text = self.entry.get().lower()

        if not text.endswith(" ") and " " in text:
            base = text[:text.rfind(" ") + 1]
            prefix = text[text.rfind(" ") + 1:]
        elif not text.endswith(" "):
            base = ""
            prefix = text
        else:
            base = text
            prefix = ""

        saved = max(0, len(word) - len(prefix))
        self.saved_keystrokes += saved
        self.update_status()

        # complete the word
        new_text = base + word + " "
        self.entry.delete(0, tk.END)
        self.entry.insert(0, new_text)
        self.update_suggestions()

        # self.add_message(word, user=True)

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    tok2id, id2tok = word_index_mappings()
    embeddings = read_embeddings('Data/glove.6B.50d.txt', tok2id, 50)
    
    # Load LSTM model
    saved_model = torch.load('final.pth')

    lstm_model = LSTM(
        input_size=50,
        hidden_size=256,
        num_layers=2,
        output_size=len(id2tok),
        embeddings=embeddings,
        tok2id=tok2id,
        id2tok=id2tok,
        device=torch.device('mps'),
    )

    lstm_model.load_state_dict(saved_model)
    
    # Load Ngram model
    data = get_sentence_tokens("Data/Datasets/conv_train.csv")
    ngram_model = Ngram(n=2)
    ngram_counts = ngram_model.build_ngram_counts(data, max_n=5)
    vocab = ngram_model.build_vocab(data)

    app = SMSPredictorApp(lstm_model, ngram_model=ngram_model, ngram_counts=ngram_counts, vocab=vocab, nr_suggestions=3)

    app.run()