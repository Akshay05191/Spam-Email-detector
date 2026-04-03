import streamlit as st
import pandas as pd
import re
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# ============================
# TEXT CLEANING FUNCTION
# ============================
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    return text

# ============================
# LOAD DATA
# ============================
@st.cache_data
def load_data():
    data = pd.read_csv("spam.csv", encoding="latin-1")
    data = data.rename(columns={"v1": "label", "v2": "text"})
    data = data[["text", "label"]]
    data["label"] = data["label"].map({"spam": 1, "ham": 0})
    data.dropna(inplace=True)
    return data

data = load_data()

# ============================
# TRAIN MODEL
# ============================
@st.cache_resource
def train_model(data):
    X = data["text"].apply(clean_text)
    y = data["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    vectorizer = TfidfVectorizer(stop_words="english", max_features=3000)
    X_train_tfidf = vectorizer.fit_transform(X_train)

    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)

    return model, vectorizer

model, vectorizer = train_model(data)

# ============================
# STREAMLIT UI
# ============================
st.set_page_config(page_title="Spam Detector", layout="centered")

st.title("📧 AI Spam Detection System")
st.write("Enter a message and check whether it's spam or not.")

user_input = st.text_area("Enter message here:")

if st.button("Check"):
    if user_input.strip() != "":
        cleaned = clean_text(user_input)
        vector = vectorizer.transform([cleaned])

        prediction = model.predict(vector)[0]
        prob = model.predict_proba(vector)[0]

        confidence = max(prob) * 100

        if prediction == 1:
            st.error(f"🚨 Spam ({confidence:.2f}% confidence)")
        else:
            st.success(f"✅ Not Spam ({confidence:.2f}% confidence)")
    else:
        st.warning("Please enter some text.")