# ============================================================
# 1. IMPORT REQUIRED LIBRARIES
# ============================================================

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report


# ============================================================
# 2. LOAD DATASET
# ============================================================

# Load dataset (latin-1 encoding is required for spam.csv)
data = pd.read_csv("spam.csv", encoding="latin-1")

# Rename columns to meaningful names
data = data.rename(columns={"v1": "label", "v2": "text"})

# Keep only required columns
data = data[["text", "label"]]

print("Dataset loaded successfully!")
print(data.head())


# ============================================================
# 3. DATA CLEANING & PREPROCESSING
# ============================================================

# Convert labels to binary values
# spam -> 1, ham -> 0
data["label"] = data["label"].map({"spam": 1, "ham": 0})

# Remove missing values
data.dropna(inplace=True)

print("\nData preprocessing completed!")
print(data["label"].value_counts())


# ============================================================
# 4. TRAIN-TEST SPLIT
# ============================================================

X = data["text"]     # Email text
y = data["label"]    # Spam or Ham label

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTrain-test split completed!")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# 5. TEXT VECTORIZATION (TF-IDF)
# ============================================================

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=3000
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("\nText vectorization completed!")


# ============================================================
# 6. MODEL TRAINING (NAIVE BAYES)
# ============================================================

model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

print("\nModel training completed!")


# ============================================================
# 7. MODEL EVALUATION
# ============================================================

y_pred = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# ============================================================
# 8. CUSTOM EMAIL PREDICTION FUNCTION
# ============================================================

def predict_email(email_text):
    """
    Predict whether an email is Spam or Not Spam
    """
    email_vector = vectorizer.transform([email_text])
    prediction = model.predict(email_vector)

    if prediction[0] == 1:
        return "🚨 Spam Email"
    else:
        return "✅ Not Spam"


# ============================================================
# 9. TEST WITH SAMPLE EMAILS
# ============================================================

sample_email_1 = "Congratulations! You won a free iPhone. Click now!"
sample_email_2 = "Hey, are we still meeting tomorrow at 10?"

print("\nSample Predictions:")
print("Email 1:", predict_email(sample_email_1))
print("Email 2:", predict_email(sample_email_2))


# ============================================================
# END OF PROGRAM
# ============================================================
