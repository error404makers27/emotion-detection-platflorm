"""
Text preprocessing utilities shared by:
 - Kaggle training scripts (BiLSTM, BERT fine-tuning)
 - The live Streamlit emotion detection pipeline
"""

import re
import string

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

_NLTK_READY = False


def ensure_nltk_data():
    """Download required NLTK corpora once (safe to call repeatedly)."""
    global _NLTK_READY
    if _NLTK_READY:
        return
    for pkg in ["stopwords", "wordnet", "omw-1.4", "punkt"]:
        try:
            nltk.data.find(f"corpora/{pkg}")
        except LookupError:
            nltk.download(pkg, quiet=True)
    _NLTK_READY = True


def clean_text(text: str) -> str:
    """Lowercase, strip URLs/mentions/punctuation/extra whitespace."""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"[^a-z\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_for_model(text: str, remove_stopwords: bool = False) -> str:
    """
    Full preprocessing pipeline: clean -> (optional stopword removal) -> lemmatize.
    Stopwords are kept by default because words like 'not', 'no', 'never' carry
    strong emotional/negation signal that hurts emotion classification if removed.
    """
    ensure_nltk_data()
    text = clean_text(text)
    tokens = text.split()

    if remove_stopwords:
        stop_words = set(stopwords.words("english")) - {
            "not", "no", "nor", "never", "very", "too"
        }
        tokens = [t for t in tokens if t not in stop_words]

    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens]

    return " ".join(tokens)


def strip_punct(text: str) -> str:
    return text.translate(str.maketrans("", "", string.punctuation))
