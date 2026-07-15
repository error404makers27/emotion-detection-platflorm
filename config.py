"""
Learn with Me — Configuration
--------------------------------
All secrets are loaded from environment variables (via a local .env file)
or from Streamlit's secrets manager. NEVER hardcode API keys in source code.

To run locally:
    1. Copy .env.example to .env
    2. Put your Gemini API key inside .env as GEMINI_API_KEY=your_key_here
    3. (Optional) On Streamlit Community Cloud, add GEMINI_API_KEY under
       Settings -> Secrets instead of using a .env file.
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Load .env file if python-dotenv is available (used for local dev).
# Loaded from BASE_DIR explicitly so it works no matter which folder
# `streamlit run` is launched from.
try:
    from dotenv import load_dotenv
    load_dotenv(BASE_DIR / ".env")
except ImportError:
    pass
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
LOG_FILE = DATA_DIR / "interaction_logs.csv"

DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)


def _secrets_file_exists() -> bool:
    """Check common secrets.toml locations before touching st.secrets,
    so Streamlit doesn't render its own 'no secrets file' banner when
    we're only using a .env file (which is the normal local setup)."""
    candidates = [
        BASE_DIR / ".streamlit" / "secrets.toml",
        Path.home() / ".streamlit" / "secrets.toml",
    ]
    return any(p.exists() for p in candidates)


def get_gemini_api_key() -> str:
    """
    Fetch Gemini API key from environment (.env) first, then Streamlit
    secrets only if a secrets.toml file is actually present. Returns an
    empty string if not configured (app shows a warning instead of
    crashing, and falls back to template-based guidance).
    """
    key = os.environ.get("GEMINI_API_KEY", "")
    if not key and _secrets_file_exists():
        try:
            import streamlit as st
            key = st.secrets.get("GEMINI_API_KEY", "")
        except Exception:
            pass
    return key


APP_NAME = "LearnSense AI"
APP_TAGLINE = "Your Emotion-Aware AI Learning Companion"

EMOTION_LABELS = [
    "joy", "sadness", "anger", "fear",
    "surprise", "disgust", "neutral", "confusion"
]

# Pretrained transformer used out-of-the-box for BERT-based emotion detection
# (works immediately, no training needed). Your Kaggle-trained BiLSTM model
# (see kaggle_training/) can be dropped into /models to enable comparison.
HF_EMOTION_MODEL = "j-hartmann/emotion-english-distilroberta-base"

BILSTM_MODEL_PATH = MODELS_DIR / "bilstm_emotion_model.h5"
BILSTM_TOKENIZER_PATH = MODELS_DIR / "bilstm_tokenizer.pkl"
BILSTM_LABEL_ENCODER_PATH = MODELS_DIR / "bilstm_label_encoder.pkl"

MIXED_EMOTION_GAP_THRESHOLD = 0.15  # if top-2 probs differ by less than this -> "mixed emotion"
MAX_SEQUENCE_LENGTH = 100
