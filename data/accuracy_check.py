
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, classification_report

def load_and_prepare_data():
    true_df = pd.read_csv("True.csv")
    fake_df = pd.read_csv("Fake.csv")
    news_df = pd.read_csv("news.csv")

    true_df['label'] = 'REAL'
    fake_df['label'] = 'FAKE'

    if 'label' not in news_df.columns:
        raise ValueError("news.csv must contain a 'label' column")

    df = pd.concat([true_df[['text', 'label']], fake_df[['text', 'label']], news_df[['text', 'label']]])
    return df.dropna().sample(frac=1).reset_index(drop=True)

def preprocess_text(df):
    tfidf = TfidfVectorizer(stop_words='english', max_df=0.7)
    X = tfidf.fit_transform(df['text'])
    y = df['label']
    return train_test_split(X, y, test_size=0.2, random_state=42), tfidf

def train_models(X_train, X_test, y_train, y_test):
    models = {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(),
        "KNN": KNeighborsClassifier(),
        "AdaBoost": AdaBoostClassifier()
    }

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        print(f"{name} Accuracy: {accuracy_score(y_test, preds):.4f}")
        print("Classification Report:")
        print(classification_report(y_test, preds))

if __name__ == "__main__":
    print("Loading and preparing data...")
    df = load_and_prepare_data()
    print("Vectorizing text and splitting dataset...")
    (X_train, X_test, y_train, y_test), tfidf = preprocess_text(df)
    print("Training and evaluating models...")
    train_models(X_train, X_test, y_train, y_test)
