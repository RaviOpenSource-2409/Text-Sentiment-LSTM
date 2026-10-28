
import re
import html
import pickle
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

import os
import requests

MODEL_URL = "https://huggingface.co/theravigupta09/sentiment-lstm-model/resolve/main/sentiment_lstm.keras"
MODEL_PATH = "/tmp/sentiment_lstm.keras"

@st.cache_resource
def get_model():
    if not os.path.exists(MODEL_PATH):
        response = requests.get(MODEL_URL, timeout=300)
        response.raise_for_status()

        with open(MODEL_PATH, "wb") as f:
            f.write(response.content)

    return load_model(MODEL_PATH)

model = get_model()

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

def clean_text(text):
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

st.title("💬 Text Sentiment Analysis")
st.write("Enter a movie review below.")

review = st.text_area("Your Review")

if st.button("Analyze Sentiment"):
    if not review.strip():
        st.warning("Please enter a review.")
    else:
        cleaned = clean_text(review)
        seq = tokenizer.texts_to_sequences([cleaned])
        padded = pad_sequences(
            seq, maxlen=200, padding="post", truncating="post"
        )

        score = float(model.predict(padded, verbose=0)[0][0])

        if score >= 0.5:
            st.success("😊 Positive Sentiment")
        else:
            st.error("😞 Negative Sentiment")

        st.write(f"Model score: {score:.2%}")
