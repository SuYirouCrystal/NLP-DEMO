import pandas as pd
import re
import html
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
from transformers import BertTokenizerFast

def preprocess_tweet(text):
    """
    Clean a tweet by unescaping HTML entities, removing URLs, user mentions,
    hashtag symbols (but keeping the word), and extra whitespace.
    """
    text = html.unescape(text)
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'\s-\s', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_and_preprocess_data(file_path):
    """
    Loads the Sentiment140 CSV dataset, assigns column names, drops unnecessary columns,
    maps sentiment labels to binary (0 = negative, 1 = positive), and cleans the tweet text.
    """
    df = pd.read_csv(file_path, encoding='latin-1', header=None)
    df.columns = ['target', 'id', 'date', 'query', 'user', 'text']
    df = df.drop(columns=['id', 'date', 'query', 'user'])
    
    df['target'] = df['target'].apply(lambda x: 1 if x == 4 else 0)
    df['text'] = df['text'].apply(preprocess_tweet)
    return df

def split_data(df, test_size=0.1, random_state=42):
    """
    Splits the data into training and validation sets, stratifying on the labels.
    """
    texts = df['text'].values
    labels = df['target'].values
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=test_size, stratify=labels, random_state=random_state
    )
    return train_texts, val_texts, train_labels, val_labels

def tokenize_data(train_texts, val_texts, max_length=128):
    """
    Tokenizes the training and validation texts using the BERT tokenizer.
    """
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    train_encodings = tokenizer(list(train_texts), truncation=True, padding=True, max_length=max_length)
    val_encodings = tokenizer(list(val_texts), truncation=True, padding=True, max_length=max_length)
    return train_encodings, val_encodings