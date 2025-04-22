# model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier

def load_data():
    true_df = pd.read_csv('data/True.csv')
    fake_df = pd.read_csv('data/Fake.csv')

    true_df['label'] = 'REAL'
    fake_df['label'] = 'FAKE'

    data = pd.concat([true_df, fake_df]).reset_index(drop=True)
    data = data.sample(frac=1).reset_index(drop=True)
    return data

def train_model(data):
    X = data['text']
    y = data['label']

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
    X_train_tfidf = vectorizer.fit_transform(X_train)

    model = PassiveAggressiveClassifier(max_iter=1000)
    model.fit(X_train_tfidf, y_train)

    return vectorizer, model

def predict_article(text, vectorizer, model):
    input_tfidf = vectorizer.transform([text])
    prediction = model.predict(input_tfidf)
    return prediction[0]
