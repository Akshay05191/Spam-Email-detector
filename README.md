# 📧 AI Spam Detection System

A Machine Learning-based web application that classifies messages as **Spam** or **Not Spam** using Natural Language Processing (NLP) techniques.

---

## 🚀 Features

- 🔍 Real-time spam detection
- 📊 Confidence score for predictions
- 🧠 TF-IDF text vectorization
- ⚡ Fast and interactive Streamlit UI
- 📁 Clean and modular code structure

---

## 🛠 Tech Stack

- Python
- Scikit-learn
- Streamlit
- Pandas
- NumPy

---

## 📂 Project Structure
Spam-Email-detector/
│
├── app_streamlit.py # Streamlit web app
├── spam_classifier.py # ML model logic
├── spam.csv # Dataset
├── requirements.txt # Dependencies
└── README.md # Project documentation


## ▶️ Run Locally

### 1. Clone repository
```bash
git clone https://github.com/Akshay05191/Spam-Email-detector.git
cd Spam-Email-detector
2. Install dependencies
pip install -r requirements.txt
3. Run the app
streamlit run app_streamlit.py
🧪 Example Inputs

Spam:

"Congratulations! You won a free iPhone. Click now!"

Not Spam:

"Hey, are we still meeting tomorrow?"

📈 Model Details
Algorithm: Multinomial Naive Bayes
Feature Extraction: TF-IDF Vectorization
Dataset: SMS Spam Dataset
Accuracy: ~96% (approx)

📌 Future Improvements
Add Logistic Regression & SVM models
Improve UI/UX design
Deploy online (Streamlit Cloud)
Add text preprocessing enhancements

👨‍💻 Author

Akshay S

⭐ If you like this project, consider giving it a star!
