# 🔍 Fake News Detector

A web app that classifies news articles as **Fake News** or **True News** using an NLP pipeline (TF-IDF + LinearSVC) served with FastAPI.

## ✨ Features

- Paste any news article or headline and get an instant verdict
- Text preprocessing: lowercasing, contraction expansion, special-character removal, stop-word removal, and lemmatization
- Clean, responsive dark-themed UI
- FastAPI backend serving both the API and the frontend

## 🧠 Model

The trained model (`fake_news_model.pkl`) is a scikit-learn pipeline combining TF-IDF vectorization with a LinearSVC classifier. Preprocessing mirrors the training pipeline so live input matches what the model was trained on.

## 📁 Project Structure

```
Falsenewsdetector/
├── main.py                 # FastAPI app + text cleaning + prediction endpoint
├── fake_news_model.pkl     # Trained TF-IDF + LinearSVC model
├── requirements.txt        # Python dependencies
├── .gitignore
├── README.md
└── static/
    └── index.html          # Frontend UI
```

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- [spaCy](https://spacy.io/) English model: `python -m spacy download en_core_web_sm`

### Installation

```bash
git clone https://github.com/<your-username>/Falsenewsdetector.git
cd Falsenewsdetector
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Run the app

```bash
uvicorn main:app --reload
```

Then open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser.

## 📡 API

### `POST /predict`

Classifies the provided text.

**Request body**

```json
{
  "text": "Your news article text here"
}
```

**Response**

```json
{
  "prediction": "Fake News" | "True News"
}
```

## 🛠️ Tech Stack

- **Backend:** FastAPI, scikit-learn, spaCy, NLTK
- **Frontend:** Vanilla HTML / CSS / JavaScript
- **Model:** TF-IDF + LinearSVC
