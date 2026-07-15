# 🧭 LearnSense AI
*Your Emotion-Aware AI Learning Companion*

An AI-powered, emotion-aware learning assistant. It reads how a student is
feeling from what they type, then responds with personalized, empathetic
guidance — combining a fine-tunable **BiLSTM**, a pretrained **BERT**
transformer, and **Gemini AI** for natural-language responses, all wrapped
in a **Streamlit** UI with a **Plotly** analytics dashboard.

---

## ⚠️ About API keys (read this first)

Never hardcode API keys in source files or paste them into chats/prompts —
once shared, treat a key as compromised and rotate it immediately.
This project reads your Gemini key from an environment variable at runtime.

**Setup:**
1. Get a fresh key at https://aistudio.google.com/app/apikey
2. Copy `.env.example` → `.env`
3. Put your key in `.env`:
   ```
   GEMINI_API_KEY=your_key_here
   ```
4. `.env` is already in `.gitignore` — never commit it.

If deploying on Streamlit Community Cloud, add the key under
**App settings → Secrets** instead of using `.env`.

---

## 📁 Project Structure

```
learn_with_me/
├── app.py                     # Main Streamlit app (UI, chat, dashboard)
├── config.py                  # Config + secure key loading
├── preprocessing.py           # NLTK-based text cleaning/lemmatization
├── emotion_pipeline.py        # BiLSTM + BERT emotion detection, confidence, mixed-emotion logic
├── gemini_client.py           # Gemini AI guidance generation (with safe fallback)
├── logger.py                  # CSV interaction logging
├── requirements.txt
├── .env.example
├── data/
│   └── interaction_logs.csv   # created automatically at runtime
├── models/                    # put your Kaggle-trained BiLSTM files here
└── kaggle_training/
    ├── train_bilstm.py        # Train BiLSTM on Kaggle GPU
    └── train_bert.py          # Optional: fine-tune BERT further
```

---

## 🚀 Quickstart (runs immediately, no training required)

```bash
pip install -r requirements.txt
cp .env.example .env   # then edit .env with your key
streamlit run app.py
```

The app works out of the box using a pretrained BERT emotion model
(`j-hartmann/emotion-english-distilroberta-base`). The custom BiLSTM is
optional — train it yourself for the "model comparison" feature.

---

## 🏋️ Step 1: Train your BiLSTM on Kaggle GPU (optional but recommended)

1. Create a new Kaggle Notebook, enable **GPU** under Settings → Accelerator.
2. Attach an emotion-labeled dataset (e.g. search Kaggle for
   "Emotion Dataset for NLP" / "dair-ai emotion dataset") with `text` and
   `emotion` columns.
3. Copy the contents of `kaggle_training/train_bilstm.py` into the notebook,
   update `DATA_PATH` to your dataset's CSV path.
4. Run all cells. It will preprocess text, tokenize, train a
   Bidirectional-LSTM, and export:
   - `bilstm_emotion_model.h5`
   - `bilstm_tokenizer.pkl`
   - `bilstm_label_encoder.pkl`
5. Download these 3 files from the Kaggle output panel and place them in
   `learn_with_me/models/`.
6. Restart the Streamlit app — it will auto-detect the BiLSTM model and
   enable side-by-side model comparison.

(Optional) `kaggle_training/train_bert.py` lets you fine-tune BERT further
on your own dataset if you want higher domain accuracy than the pretrained
model.

---

## 🎭 How emotion detection works

- **Preprocessing**: lowercasing, URL/mention stripping, lemmatization
  (negation words like "not"/"never" are preserved — they matter for emotion).
- **Classification**: BERT (always available) + BiLSTM (once trained) each
  output a full probability distribution over emotions.
- **Confidence scoring**: softmax probability of the top predicted emotion.
- **Mixed-emotion detection**: if the top two emotions' probabilities are
  within `MIXED_EMOTION_GAP_THRESHOLD` (default 0.15) of each other, the
  result is flagged as "mixed" and both emotions are surfaced.

## 💬 How AI guidance works

`gemini_client.py` sends the student's message + detected emotion(s) to
Gemini (`gemini-1.5-flash`) with a system-style prompt that keeps responses
warm, encouraging, and practical — like a supportive friend/tutor. If no API
key is configured, or the API call fails, the app falls back to
emotion-specific template responses so the demo never breaks.

## 📊 Analytics dashboard

Every interaction is appended to `data/interaction_logs.csv` (timestamp,
input text, both models' predicted emotions + confidences, mixed-emotion
flag, AI response). The **Analytics Dashboard** tab visualizes this with
Plotly: emotion distribution (pie), confidence trend over time (line),
emotion timeline (scatter), plus a raw log table with CSV export.

---

## ✅ Testing & optimization checklist

- [ ] Verify NLTK downloads succeed on first run (`preprocessing.ensure_nltk_data`)
- [ ] Confirm BERT model loads and predicts on sample inputs of varying length
- [ ] If using BiLSTM: check validation accuracy/F1 from training logs before deploying
- [ ] Test Gemini fallback path by temporarily removing `GEMINI_API_KEY`
- [ ] Load-test CSV logging with many rapid submissions (CSV append is safe for single-user/local use; for multi-user production, migrate `logger.py` to a real database)
- [ ] Review `data/interaction_logs.csv` periodically — it may contain sensitive student text; treat as private data

## 🚢 Deployment notes

- **Streamlit Community Cloud**: push this repo (with `.env` excluded), add
  `GEMINI_API_KEY` under App Settings → Secrets.
- **Model size**: BERT + TensorFlow can be memory-heavy; on constrained
  hosts consider a smaller distilled model or CPU-only inference.
- **Privacy**: interaction logs may contain personal/emotional student data
  — don't commit `data/interaction_logs.csv` to a public repo (already in
  `.gitignore`).
