
# 📰 Fake News Detection Web App

This is a clean and interactive **Fake News Detector App** built with **Streamlit**.  
It classifies news articles as either **Fake** or **Real**.

Three models are implemented and selectable within the UI:
- **Logistic Regression (TF-IDF based)**
- **Support Vector Machine (TF-IDF based)**
- **LSTM Neural Network (Deep Learning)**

---

## 📌 Features

✅ Clean and lemmatize text input using NLTK  
✅ TF-IDF-based classification (LogReg and SVM)  
✅ LSTM-based deep learning model with Keras  
✅ Displays predicted **label** and **confidence** (when available)  
✅ Beautiful and intuitive **Streamlit UI**  
✅ Word Clouds and Frequency Analysis included in the notebook  
✅ Full pipeline from data to deployment

---

## 🔗 Live Demo

Try the app live here:  
👉 [https://fake-news-detection-app-77.streamlit.app/]

---

## 📁 Files Included

| File | Description |
|------|-------------|
| `app.py` | Streamlit web app for real/fake news classification |
| `Elevvo_NLP_Internship_Task3.ipynb` | Full notebook for data processing, training, evaluation, and visualization |
| `requirements.txt` | All required Python packages |
| `model/logistic_regression_model.pkl` | Saved Logistic Regression model |
| `model/svm_model.pkl` | Saved Support Vector Machine model |
| `model/lstm_model.keras` | Trained LSTM model saved in Keras format |
| `model/tfidf_vectorizer.pkl` | TF-IDF vectorizer used in both classical models |
| `model/tokenizer.pkl` | Tokenizer used for LSTM sequences |

---

## 🚀 How to Run Locally

1. **Clone the repo**:

```bash
git clone https://github.com/yourusername/fake-news-detector.git
cd fake-news-detector
```

2. **(Optional) Create a virtual environment**:

```bash
python -m venv venv
venv\Scripts\activate  # on Windows
```

3. **Install dependencies**:

```bash
pip install -r requirements.txt
```

4. **Run the app**:

```bash
streamlit run app.py
```

---

## 🌍 Deployment (Optional)

To deploy on **Streamlit Cloud**:

- Go to [https://streamlit.io/cloud](https://streamlit.io/cloud)
- Connect your GitHub repo
- Set the main file to: `app.py`
- Add model and vectorizer files to `/model` directory in the repo

---

## 📓 Notebook Workflow

> [`Elevvo_NLP_Internship_Task3.ipynb`](./Elevvo_NLP_Internship_Task3.ipynb)

Includes:
- Custom combined dataset from `Fake.csv` and `True.csv` (Kaggle)
- Text preprocessing: lowercasing, punctuation removal, stopwords, lemmatization
- TF-IDF vectorization (for LogReg and SVM)
- Sequence tokenization & padding (for LSTM)
- LSTM model training and saving
- Classification reports and confusion matrices
- WordClouds and bar charts per class
- Final evaluation and summary of results

---

## ⚠️ Final Remarks

Despite exploring multiple models and tuning their architectures, a consistent number of misclassifications persisted — especially on articles with neutral or technical tone. Even the LSTM model, built using a top-rated Kaggle architecture, struggled to generalize on certain real-world examples. This highlights the need for richer embeddings and better datasets for real production use.
