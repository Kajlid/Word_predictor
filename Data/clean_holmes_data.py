import random
import re
import csv

def clean_and_split_sentences(text):
    # Remove extra whitespace lines
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    cleaned_text = ' '.join(lines)

    # Split into sentences (approximately)
    sentences = re.split(r'(?<=[.!?]) +', cleaned_text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences

def split_data(sentences, train_ratio=0.95):
    random.seed(1)
    random.shuffle(sentences)
    split_idx = int(len(sentences) * train_ratio)
    return sentences[:split_idx], sentences[split_idx:]

def save_to_file(filename, data):
    with open(filename, 'w', encoding='utf-8') as f:
        for sentence in data:
            f.write(sentence + '\n')
            

def main():
    with open('Data/Datasets/sherlock_holmes.txt', 'r', encoding='utf-8') as f:
        raw_text = f.read()
        
    sentences = clean_and_split_sentences(raw_text)

    # Split into train and validation sets
    train_data, val_data = split_data(sentences)

    # Save to separate files
    save_to_file('Data/Datasets/sherlock_train.txt', train_data)
    save_to_file('Data/Datasets/sherlock_val.txt', val_data)

    print(f"Saved {len(train_data)} sentences to train.txt and {len(val_data)} to val.txt.")

if __name__ == "__main__":
    main()
