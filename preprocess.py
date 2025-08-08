import os
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string

nltk.download('stopwords')
nltk.download('wordnet')

def load_data(data_dir):
    texts = []
    labels = []
    for label_type in ['pos', 'neg']:
        dir_name = os.path.join(data_dir, label_type)
        for fname in os.listdir(dir_name):
            if fname.endswith('.txt'):
                with open(os.path.join(dir_name, fname), encoding='utf-8') as f:
                    texts.append(f.read())
                labels.append(1 if label_type == 'pos' else 0)
    return pd.DataFrame({'review': texts, 'sentiment': labels})

def clean_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = nltk.word_tokenize(text)
    filtered = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(filtered)

if __name__ == "__main__":
    train_df = load_data('aclImdb/train')
    test_df = load_data('aclImdb/test')

    train_df['clean_review'] = train_df['review'].apply(clean_text)
    test_df['clean_review'] = test_df['review'].apply(clean_text)

    train_df.to_csv('train_clean.csv', index=False)
    test_df.to_csv('test_clean.csv', index=False)
