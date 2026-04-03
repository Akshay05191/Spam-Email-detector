from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

app = Flask(__name__)

# ============================================================
# LOAD & TRAIN MODEL (runs once on startup)
# ============================================================

data = pd.read_csv("spam.csv", encoding="latin-1")
data = data.rename(columns={"v1": "label", "v2": "text"})
data = data[["text", "label"]]

data["label"] = data["label"].map({"spam": 1, "ham": 0})
data.dropna(inplace=True)

X = data["text"]
y = data["label"]

X_train, _, y_train, _ = train_test_split(
    X, y, test_size=0.2, random_state=42
)

vectorizer = TfidfVectorizer(stop_words="english", max_features=3000)
X_train_tfidf = vectorizer.fit_transform(X_train)

model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# ============================================================
# ROUTES
# ============================================================

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":
        email_text = request.form["email"]
        email_vector = vectorizer.transform([email_text])
        result = model.predict(email_vector)[0]

        prediction = "🚨 Spam Email" if result == 1 else "✅ Not Spam"

    return render_template("index.html", prediction=prediction)

# ============================================================
# RUN APP
# ============================================================

if __name__ == "__main__":
    app.run(debug=True)
