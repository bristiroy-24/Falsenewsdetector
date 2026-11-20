import re
import joblib
import nltk
import spacy
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

# ── Download required NLTK data ──────────────────────────────────────────────
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ── Load spaCy model for its stop-word list ───────────────────────────────────
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

# ── Build combined stop-word set ──────────────────────────────────────────────
STOP_WORDS: set[str] = set(stopwords.words("english")) | nlp.Defaults.stop_words

# ── Lemmatizer ────────────────────────────────────────────────────────────────
lemmatizer = WordNetLemmatizer()

# ── Contraction map ───────────────────────────────────────────────────────────
CONTRACTIONS: dict[str, str] = {
    "i'm":       "i am",
    "i've":      "i have",
    "i'll":      "i will",
    "i'd":       "i would",
    "you're":    "you are",
    "you've":    "you have",
    "you'll":    "you will",
    "you'd":     "you would",
    "he's":      "he is",
    "he'll":     "he will",
    "he'd":      "he would",
    "she's":     "she is",
    "she'll":    "she will",
    "she'd":     "she would",
    "it's":      "it is",
    "it'll":     "it will",
    "we're":     "we are",
    "we've":     "we have",
    "we'll":     "we will",
    "we'd":      "we would",
    "they're":   "they are",
    "they've":   "they have",
    "they'll":   "they will",
    "they'd":    "they would",
    "that's":    "that is",
    "that'll":   "that will",
    "that'd":    "that would",
    "who's":     "who is",
    "who'll":    "who will",
    "who'd":     "who would",
    "what's":    "what is",
    "what'll":   "what will",
    "what'd":    "what would",
    "where's":   "where is",
    "when's":    "when is",
    "why's":     "why is",
    "how's":     "how is",
    "isn't":     "is not",
    "aren't":    "are not",
    "wasn't":    "was not",
    "weren't":   "were not",
    "haven't":   "have not",
    "hasn't":    "has not",
    "hadn't":    "had not",
    "won't":     "will not",
    "wouldn't":  "would not",
    "don't":     "do not",
    "doesn't":   "does not",
    "didn't":    "did not",
    "can't":     "cannot",
    "couldn't":  "could not",
    "shouldn't": "should not",
    "mightn't":  "might not",
    "mustn't":   "must not",
    "let's":     "let us",
    "there's":   "there is",
    "there're":  "there are",
    "here's":    "here is",
}

# Pre-compile a single regex that matches any contraction key
_CONTRACTION_RE = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in CONTRACTIONS) + r")\b"
)

# ── Text cleaning pipeline ─────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    # 1. Lowercase
    text = text.lower()

    # 2. Expand contractions
    text = _CONTRACTION_RE.sub(lambda m: CONTRACTIONS[m.group(0)], text)

    # 3. Remove special characters (pass 1 – punctuation / symbols)
    text = re.sub(r"[-()!?@#$%&*()?\"*\'\s]+", " ", text)

    # 4. Remove anything that isn't alphanumeric or whitespace (pass 2)
    text = re.sub(r"[^A-Za-z0-9\s]", "", text)

    # 5. Remove stop words & lemmatize
    tokens = [
        lemmatizer.lemmatize(word)
        for word in text.split()
        if word not in STOP_WORDS
    ]

    return " ".join(tokens)


# ── Load trained model ────────────────────────────────────────────────────────
model = joblib.load("fake_news_model.pkl")

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(title="Fake News Detector")

# Serve the frontend
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", include_in_schema=False)
async def root():
    return FileResponse("static/index.html")


# ── Request / response schemas ────────────────────────────────────────────────
class PredictRequest(BaseModel):
    text: str


class PredictResponse(BaseModel):
    prediction: str


# ── Prediction endpoint ───────────────────────────────────────────────────────
@app.post("/predict", response_model=PredictResponse)
async def predict(payload: PredictRequest):
    cleaned = clean_text(payload.text)
    label = int(model.predict([cleaned])[0])
    result = "Fake News" if label == 1 else "True News"
    return PredictResponse(prediction=result)
