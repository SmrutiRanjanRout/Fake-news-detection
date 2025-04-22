# app.py
import streamlit as st
from model import load_data, train_model, predict_article
from utils import clean_text, extract_text_from_url

st.set_page_config(page_title="Fake News Detector", page_icon="🕵️", layout="centered")

st.title("🕵️‍♂️ Fake News Detector")
st.markdown("Check if a news article is **REAL** or **FAKE**.")

# Load data and train model (cache for performance)
@st.cache_data
def get_data():
    return load_data()

@st.cache_resource
def get_model():
    data = get_data()
    return train_model(data)

vectorizer, model = get_model()

# User input
st.subheader("Input Options")
input_type = st.radio("Choose input type:", ["Text", "URL"])
article=""
if input_type == "Text":
    article = st.text_area("📰 Paste the news article text here:")
elif input_type == "URL":
    url = st.text_input("🔗 Paste the article URL:")
    if url:
        with st.spinner("Fetching article..."):
            article = extract_text_from_url(url)
        if article.startswith("Error:"):
            st.error(article)
        else:
            st.text_area("Fetched Article Text:", article, height=300)

if st.button("Check Now"):
    if article:
        cleaned_text = clean_text(article)
        result = predict_article(cleaned_text, vectorizer, model)
        if result == "FAKE":
            st.error(f"🚨 The article is likely **{result}**.")
        else:
            st.success(f"✅ The article appears to be **{result}**.")
    else:
        st.warning("Please enter some text to check.")
