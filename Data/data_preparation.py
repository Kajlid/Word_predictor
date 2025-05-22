import pandas as pd
from nltk.tokenize import word_tokenize

def create_dataframe(filename):
    df = pd.read_csv(filename, skiprows=0)
    
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    
    # Concatenate questions and answers vertically 
    stacked_df = pd.concat([df['question'], df['answer']]).dropna().reset_index(drop=True)
    
    return stacked_df


def get_sentence_tokens(filename):
    conv_data = create_dataframe(filename)
    sentence_tokens = conv_data.apply(word_tokenize)     # Lists of tokens
    
    token_arr = sentence_tokens.to_numpy()
    
    return token_arr



if __name__== "__main__":
    sentence_tokens = get_sentence_tokens("Data/Datasets/Conversation.csv")
    
    print(sentence_tokens)

    
    