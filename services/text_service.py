"""
Text model: fake-news detector (TF-IDF + Logistic Regression).

The cleaning function below is a faithful copy of the one used during training
(see Text_Model.ipynb). If you retrain the model with a different clean_text(),
change it here too, otherwise predictions will silently degrade.

NLTK's English stopword list is bundled in services/data/english_stopwords.txt
so the app works offline and needs no `nltk.download()` step.
"""
import json
import re
from pathlib import Path

import joblib

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "text" / "fake_news_pipeline.pkl"
META_PATH = BASE_DIR / "models" / "text" / "model_metadata.json"
STOPWORDS_PATH = Path(__file__).resolve().parent / "data" / "english_stopwords.txt"

MIN_TEXT_LEN = 30        # characters; shorter input still runs but gets a warning
MAX_TEXT_CHARS = 100_000  # guard against absurdly large pastes


class TextService:
    key = "text"

    def __init__(self):
        self.pipeline = None
        self.metadata = {}
        self.stopwords = set()
        self.error = None

    # ------------------------------------------------------------------ load
    def load(self):
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
        self.stopwords = set(STOPWORDS_PATH.read_text(encoding="utf-8").split())
        self.pipeline = joblib.load(MODEL_PATH)
        if META_PATH.exists():
            self.metadata = json.loads(META_PATH.read_text(encoding="utf-8"))

    @property
    def ready(self):
        return self.pipeline is not None

    @property
    def summary(self):
        m = self.metadata
        parts = []
        if "test_accuracy" in m:
            parts.append(f"{m['test_accuracy'] * 100:.1f}% test accuracy")
        if "train_samples" in m and "test_samples" in m:
            parts.append(f"trained on {m['train_samples'] + m['test_samples']:,} articles")
        return " \u00b7 ".join(parts) or "TF-IDF + Logistic Regression"

    # -------------------------------------------------------------- cleaning
    def clean_text(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r"http\S+|www\.\S+", "", text)           # URLs
        text = re.sub(r"\S+@\S+", "", text)                    # e-mail addresses
        text = re.sub(r"<[^>]+>", "", text)                    # HTML tags
        text = re.sub(r"^.*?\(reuters\)\s*[-\u2013]?\s*", "", text)  # wire-service header
        text = re.sub(r"[^a-z\s]", " ", text)                  # letters only
        text = re.sub(r"\s+", " ", text).strip()
        tokens = [t for t in text.split() if t not in self.stopwords and len(t) > 2]
        return " ".join(tokens)

    # -------------------------------------------------------------- predict
    def predict(self, text: str) -> dict:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Please enter some text to check.")
        if len(text) > MAX_TEXT_CHARS:
            raise ValueError(f"That text is too long. The limit is {MAX_TEXT_CHARS:,} characters.")

        stripped = text.strip()
        warning = None
        if len(stripped) < MIN_TEXT_LEN:
            warning = (f"This text is very short ({len(stripped)} characters), "
                       "so the result may be less reliable.")

        cleaned = self.clean_text(text)
        if not cleaned:
            raise ValueError("There was no readable text left after cleaning. "
                             "Try pasting more of the article.")

        proba = self.pipeline.predict_proba([cleaned])[0]
        fake_pct = round(float(proba[0]) * 100, 2)   # class 0 = FAKE
        real_pct = round(float(proba[1]) * 100, 2)   # class 1 = REAL
        verdict = "REAL" if real_pct >= fake_pct else "FAKE"

        return {
            "verdict": verdict,
            "confidence": max(fake_pct, real_pct),
            "fake_probability": fake_pct,
            "real_probability": real_pct,
            "word_count": len(text.split()),
            "warning": warning,
            "cleaned_preview": cleaned[:200] + ("..." if len(cleaned) > 200 else ""),
        }
