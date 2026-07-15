"""
Emotion Detection Pipeline — Learn with Me
--------------------------------------------
Combines two models so the app can show a "model comparison" view:

  1. BERT (transformer)  -> loaded via HuggingFace `transformers`, works out
     of the box with no training required (pretrained on emotion data).
  2. BiLSTM (custom)     -> optional. Train it yourself on Kaggle GPU using
     kaggle_training/train_bilstm.py, then drop the exported files into
     /models. If those files aren't present, the app gracefully runs on
     BERT alone and tells the user BiLSTM isn't loaded yet.

Both models produce:
  - predicted emotion label
  - confidence score (softmax probability of top class)
  - full probability distribution (for the analytics dashboard)
  - mixed-emotion flag (true when the top-2 emotions are close in probability)
"""

from dataclasses import dataclass, field
from typing import Optional
import pickle

import numpy as np

import config
from preprocessing import preprocess_for_model


@dataclass
class EmotionResult:
    model_name: str
    label: str
    confidence: float
    probabilities: dict
    is_mixed: bool
    secondary_label: Optional[str] = None
    secondary_confidence: Optional[float] = None


class BertEmotionDetector:
    """Wraps a pretrained HuggingFace transformer for emotion classification."""

    def __init__(self, model_name: str = config.HF_EMOTION_MODEL):
        self.model_name = model_name
        self._pipeline = None

    def _load(self):
        if self._pipeline is None:
            from transformers import pipeline as hf_pipeline
            self._pipeline = hf_pipeline(
                "text-classification",
                model=self.model_name,
                top_k=None,  # return scores for all labels
                framework="pt",  # force PyTorch; avoids TF/Keras-3 incompatibility
            )
        return self._pipeline

    def predict(self, text: str) -> EmotionResult:
        pipe = self._load()
        raw = pipe(text[:512])[0]  # list of {label, score}
        raw = sorted(raw, key=lambda x: x["score"], reverse=True)

        probs = {r["label"].lower(): float(r["score"]) for r in raw}
        top, second = raw[0], raw[1] if len(raw) > 1 else raw[0]

        is_mixed = (top["score"] - second["score"]) < config.MIXED_EMOTION_GAP_THRESHOLD

        return EmotionResult(
            model_name="BERT (transformer)",
            label=top["label"].lower(),
            confidence=float(top["score"]),
            probabilities=probs,
            is_mixed=is_mixed,
            secondary_label=second["label"].lower() if is_mixed else None,
            secondary_confidence=float(second["score"]) if is_mixed else None,
        )


class BiLSTMEmotionDetector:
    """
    Wraps a custom-trained BiLSTM model (Keras/TensorFlow).
    Expects three artifacts exported from kaggle_training/train_bilstm.py:
        - bilstm_emotion_model.h5   (Keras model)
        - bilstm_tokenizer.pkl      (fitted Keras Tokenizer)
        - bilstm_label_encoder.pkl  (fitted sklearn LabelEncoder)
    """

    def __init__(self):
        self._model = None
        self._tokenizer = None
        self._label_encoder = None

    @property
    def is_available(self) -> bool:
        return (
            config.BILSTM_MODEL_PATH.exists()
            and config.BILSTM_TOKENIZER_PATH.exists()
            and config.BILSTM_LABEL_ENCODER_PATH.exists()
        )

    def _load(self):
        if self._model is None:
            from tensorflow.keras.models import load_model
            self._model = load_model(config.BILSTM_MODEL_PATH)
            with open(config.BILSTM_TOKENIZER_PATH, "rb") as f:
                self._tokenizer = pickle.load(f)
            with open(config.BILSTM_LABEL_ENCODER_PATH, "rb") as f:
                self._label_encoder = pickle.load(f)

    def predict(self, text: str) -> EmotionResult:
        from tensorflow.keras.preprocessing.sequence import pad_sequences

        self._load()
        cleaned = preprocess_for_model(text)
        seq = self._tokenizer.texts_to_sequences([cleaned])
        padded = pad_sequences(seq, maxlen=config.MAX_SEQUENCE_LENGTH, padding="post")

        probs_array = self._model.predict(padded, verbose=0)[0]
        labels = self._label_encoder.classes_

        order = np.argsort(probs_array)[::-1]
        top_idx, second_idx = order[0], order[1] if len(order) > 1 else order[0]

        probs = {labels[i]: float(probs_array[i]) for i in range(len(labels))}
        is_mixed = (probs_array[top_idx] - probs_array[second_idx]) < config.MIXED_EMOTION_GAP_THRESHOLD

        return EmotionResult(
            model_name="BiLSTM (custom-trained)",
            label=labels[top_idx],
            confidence=float(probs_array[top_idx]),
            probabilities=probs,
            is_mixed=is_mixed,
            secondary_label=labels[second_idx] if is_mixed else None,
            secondary_confidence=float(probs_array[second_idx]) if is_mixed else None,
        )


class EmotionPipeline:
    """High-level façade used by the Streamlit app."""

    def __init__(self):
        self.bert = BertEmotionDetector()
        self.bilstm = BiLSTMEmotionDetector()

    def analyze(self, text: str, use_bilstm: bool = True):
        """
        Returns a dict: {"bert": EmotionResult, "bilstm": EmotionResult or None}
        """
        results = {"bert": self.bert.predict(text), "bilstm": None}
        if use_bilstm and self.bilstm.is_available:
            results["bilstm"] = self.bilstm.predict(text)
        return results

    def primary_result(self, results: dict) -> EmotionResult:
        """Prefer the custom BiLSTM result if available, else BERT."""
        return results["bilstm"] if results.get("bilstm") else results["bert"]
