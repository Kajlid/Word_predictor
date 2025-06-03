# Word Predictor for Conversational Systems

In this project, we have built a chat-like graphical user interface integrating three different models (a two-layer LSTM, a bigram and a trigram) for word prediction and suggestion of word completion.


## Dependencies
Make sure you have Python 3.7+ installed. The code was primarily built and run in Python 3.12.9.

The project depends on the following libraries:

- csv - for reading the dataset

- pandas - for creating a pandas DataFrame from the csv data

- nltk.tokenize - for tokenizing the data

- torch (PyTorch) — for LSTM model loading and inference

- tqdm - for monitoring progress of certain operations

- numpy — for numerical operations

- matplotlib - for plotting graphs

Other standard libraries: tkinter, sys, os, collections, timeit, math, random.

If you don't have any of the packages installed, you can install them via pip or conda, preferably within an environment. If you are downloading nltk for the first time, you need to run ```nltk.download('punkt')``` in the codebase to be able to run ```word_tokenize``` in ```Data/data_preparation.py```. 

The LSTM model was trained on a Macbook with MPS, but it also supports CUDA.

## Setup

1) Clone the repository

```
git clone https://github.com/Kajlid/Word_predictor.git
cd Word_predictor
```

2) Install dependencies, as described above.

3) Download GloVe embeddings:
To run the code, you will need a file with pretrained GloVe embeddings that are available online.
    You can download the GloVe embeddings from:
    https://nlp.stanford.edu/projects/glove/

    Download 6B.zip and move the file ```glove.6b.50b.txt``` to the ```Data``` directory to make sure that it gets the relative path: 
```Data/glove.6B.50d.txt```.

4) Make sure the following files exist:
- final.pth (trained and saved final LSTM model if you don't wish to train a new model)

- Data/glove.6B.50d.txt (pretrained GloVe embeddings, has to be downloaded and placed in Data/ manually, see step 3)

- Data/Datasets/conv_train.csv (train set)

- Data/Datasets/conv_val.csv (validation set)

## Run the GUI
To start the GUI window, run:

```python GUI/SMS.py```

From the newly opened GUI window, you can:

- Choose a model for prediction (LSTM, Bigram, or Trigram) [Required]. 

    The script will automatically assume that your LSTM model is final.pth, that your train set is conv_train.csv and that the GloVe embeddings used are glove.6B.50d.txt, but this can be changed in the SMS.py file.

- Use the input box for suggestions, although inital suggestions will be displayed as soon as a model has been chosen.

- Press the suggestion boxes to accept suggested completions and view saved keystroke stats at the bottom.

- Press Reset to clear saved and total keystroke counts (does not clear chat history).

If you wish to start on a new message, press the enter key.


## Contact

For questions, write to kajsalid@kth.se or ehedlin@kth.se.




