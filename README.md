# 🕵️ Fake News Detection App

This is a simple Python web app built using **Streamlit** to detect whether a news article is **Fake** or **Real** using machine learning.

## 🧠 Model
- TF-IDF Vectorizer
- PassiveAggressiveClassifier
- Trained on `True.csv` and `Fake.csv` datasets
  
## Required libraries
```bash
pip install lxml[html_clean]
pip install newspaper3k

## 🚀 How to Run

```bash
streamlit run app.py
