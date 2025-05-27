from GUI import GUI
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tkinter as tk
from Models.LSTM import LSTM
from Data.data_preparation import word_index_mappings, read_embeddings
import torch

class SMS(GUI):
    def __init__(self, model, nr_suggestions):
        super().__init__(model, nr_suggestions)

        for widget in (self.entry, self.sugg_frame):
            widget.pack_forget()

        self.chat_frame = tk.Frame(self.root, bg="#ECE5DD")
        self.chat_frame.pack(fill="both", expand=True)

        self.canvas = tk.Canvas(self.chat_frame, bg="#ECE5DD", highlightthickness=0)
        self.scrollbar = tk.Scrollbar(self.chat_frame, orient="vertical",
                                      command=self.canvas.yview)
        self.inner_frame = tk.Frame(self.canvas, bg="#ECE5DD")

        self.inner_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(
                scrollregion=self.canvas.bbox("all")
            )
        )

        self.canvas.create_window((0,0), window=self.inner_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        # 3) Re-pack entry & suggestions at bottom
        self.entry.pack(fill="x", padx=10, pady=(5,0))
        self.sugg_frame.pack(fill="x", padx=10, pady=(0,5))

    def add_message(self, text, sent_by_user=True):
        """
        Add a message bubble to the scrollable chat.
        sent_by_user: True=right aligned (blue), False=left (grey)
        """
        bubble = tk.Label(self.inner_frame,
                          text=text,
                          wraplength=250,
                          justify="left",
                          font=("Helvetica", 14),
                          padx=10, pady=5,
                          bd=0)
        if sent_by_user:
            bubble.config(bg="#32a852")   # light-green bubble
            bubble.pack(anchor="e", pady=2, padx=10)
        else:
            bubble.config(bg="#FFFFFF")   # white bubble
            bubble.pack(anchor="w", pady=2, padx=10)

        # auto-scroll to bottom
        self.canvas.update_idletasks()
        self.canvas.yview_moveto(1.0)

    def on_suggest(self, idx):
        # 1) get the chosen word
        word = self.current_suggestions[idx]

        # 2) insert into entry + space
        text = self.entry.get()
        if not text.endswith(" ") and " " in text:
            text = text[:text.rfind(" ")+1]
        elif not text.endswith(" "):
            text = ""
        new_text = text + word + " "
        self.entry.delete(0, tk.END)
        self.entry.insert(0, new_text)

        # 4) refresh suggestions
        self.update_suggestions()

    def on_key(self, event=None):
        if event and event.keysym == "Return":
            msg = self.entry.get().strip()
            if msg:
                self.add_message(msg, sent_by_user=True)
                self.entry.delete(0, tk.END)
            return


        self.update_suggestions()

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

    app = SMS(model, 3)

    app.run()