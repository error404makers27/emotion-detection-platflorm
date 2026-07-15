"""
CSV Logging — Learn with Me
------------------------------
Stores every interaction: timestamp, user input, detected emotions from
each model, confidence scores, mixed-emotion flag, and the AI response.
Used by the analytics dashboard (Plotly) to visualize trends over time.
"""

import csv
from datetime import datetime

import config

FIELDNAMES = [
    "timestamp", "user_input", "bert_emotion", "bert_confidence",
    "bilstm_emotion", "bilstm_confidence", "is_mixed", "secondary_emotion",
    "ai_response",
]


def _ensure_log_file():
    if not config.LOG_FILE.exists():
        with open(config.LOG_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()


def log_interaction(user_input: str, bert_result, bilstm_result, ai_response: str):
    _ensure_log_file()
    primary = bilstm_result if bilstm_result else bert_result

    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "user_input": user_input,
        "bert_emotion": bert_result.label,
        "bert_confidence": round(bert_result.confidence, 4),
        "bilstm_emotion": bilstm_result.label if bilstm_result else "",
        "bilstm_confidence": round(bilstm_result.confidence, 4) if bilstm_result else "",
        "is_mixed": primary.is_mixed,
        "secondary_emotion": primary.secondary_label or "",
        "ai_response": ai_response,
    }

    with open(config.LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writerow(row)


def load_logs():
    """Returns a pandas DataFrame of all logged interactions (empty if none)."""
    import pandas as pd
    _ensure_log_file()
    try:
        df = pd.read_csv(config.LOG_FILE)
    except pd.errors.EmptyDataError:
        df = pd.DataFrame(columns=FIELDNAMES)
    return df
