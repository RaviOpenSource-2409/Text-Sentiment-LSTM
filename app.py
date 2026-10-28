
import re
import html
import pickle
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

model = load_model("sentiment_lstm.keras")

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
