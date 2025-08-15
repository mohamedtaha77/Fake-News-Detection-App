# app.py — Streamlit app for Fake News Detection

import joblib
import re
import streamlit as st
from pathlib import Path

# NLTK for preprocessing
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# TensorFlow + Keras
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ========== Download NLTK resources ==========
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

# ========== Preprocessing Setup ==========
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\d+", " ", text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return " ".join(tokens)

# ========== Page setup ==========
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")
st.title("📰 Fake News Detection App")
st.markdown("""
This app classifies a news article as **Fake** or **Real**.

🔘 Choose a model  
✍️ Enter a news article  
📊 Get the prediction  
""")

# ========== Load Models ==========
logreg_model = joblib.load("model/logistic_regression_model.pkl")
svm_model = joblib.load("model/svm_model.pkl")
vectorizer = joblib.load("model/tfidf_vectorizer.pkl")

# Load Keras LSTM model and tokenizer
lstm_model = tf.keras.models.load_model("model/lstm_model.keras")
with open("model/tokenizer.pkl", "rb") as f:
    lstm_tokenizer = pickle.load(f)

# Constants
vocab_size = 5000
max_len = 500

# ========== Label Names ==========
label_names = {
    0: "🧪 Fake News",
    1: "📰 Real News"
}

# ========== User Input ==========
text_input = st.text_area("📝 Enter your news article:", height=150)

model_choice = st.radio("🤖 Choose model:", [
    "Logistic Regression (TF-IDF)",
    "Support Vector Machine (TF-IDF)",
    "LSTM Neural Network"
])

# ========== Prediction ==========
if st.button("🔍 Predict"):
    if not text_input.strip():
        st.warning("Please enter a news article.")
    else:
        processed = preprocess_text(text_input)

        if model_choice == "Logistic Regression (TF-IDF)":
            vectorized = vectorizer.transform([processed])
            pred = logreg_model.predict(vectorized)[0]
            prob = logreg_model.predict_proba(vectorized)[0]
            confidence = f"{prob[pred] * 100:.2f}%"

        elif model_choice == "Support Vector Machine (TF-IDF)":
            vectorized = vectorizer.transform([processed])
            pred = svm_model.predict(vectorized)[0]
            confidence = "N/A"

        else:  # LSTM
            seq = lstm_tokenizer.texts_to_sequences([processed])
            padded = pad_sequences(seq, maxlen=max_len, padding='post')
            pred_prob = lstm_model.predict(padded)[0][0]
            pred = int(pred_prob >= 0.5)
            confidence = f"{pred_prob * 100:.2f}%"

        # ========== Results ==========
        st.markdown("---")
        st.subheader("🧠 Prediction Result")
        st.markdown(f"**Category:** {label_names[pred]}")
        st.markdown(f"**Confidence:** <span style='color:green;'>{confidence}</span>", unsafe_allow_html=True)
        st.caption("Note:")
        st.caption("- Confidence is not available for SVM.")
        st.caption("- LSTM confidence is based on sigmoid output.")

        st.markdown(
    "<span style='font-size: 0.9em; color: gray;'>⚠️ Note: Predictions are based on training data and may not always be accurate due to limitations in dataset quality, linguistic nuance, or real-world ambiguity.</span>",
    unsafe_allow_html=True
)

        st.markdown("---")

# Footer
st.markdown("<div style='text-align: center;'>Made with ❤️ for Elevvo Internship Task 3</div>", unsafe_allow_html=True)
