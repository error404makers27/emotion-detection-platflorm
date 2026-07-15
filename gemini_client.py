"""
Gemini AI Guidance Generator — Learn with Me
------------------------------------------------
This is the "adaptive tutor brain": it takes the student's detected emotion
and turns it into concrete teaching-style instructions for Gemini —
adjusting explanation difficulty, pacing, and tone automatically. The
student never sees emotion labels or confidence scores; they just
experience a tutor that seems to "get" how they're feeling.

Adaptation logic (mirrors how a good human tutor behaves):
  - frustrated / angry / fear / confusion  -> SIMPLIFY: shorter steps,
      more encouragement, break the problem down, slow the pace down.
  - sadness                                -> SUPPORT: gentle, patient,
      reassuring, low-pressure, small wins.
  - joy / confident / neutral (high streak) -> CHALLENGE: faster pace,
      deeper explanations, follow-up questions, extension ideas.
  - surprise / disgust                     -> CLARIFY: re-explain from a
      different angle, check assumptions.

SECURITY NOTE: The API key is never hardcoded here. It's read at runtime
from the GEMINI_API_KEY environment variable (see config.py). If no key is
configured, the app falls back to supportive template responses so it still
works end-to-end for demos.
"""

from config import get_gemini_api_key

_BASE_PERSONA = (
    "You are 'Learn with Me' — a warm, emotionally intelligent AI study "
    "companion. Your tagline is 'learn like with your friend'. You are "
    "talking directly to a student in an ongoing chat conversation. "
    "NEVER mention that you are detecting emotions, never say things like "
    "'I sense you are frustrated' or name a confidence score — just adapt "
    "your teaching naturally, the way a perceptive friend would, without "
    "announcing that you're doing it. Keep responses focused and not overly "
    "long unless the student's question needs depth."
)

# Maps a detected emotion to concrete teaching-style instructions.
_ADAPTATION_STRATEGY = {
    "anger": (
        "The student seems frustrated. Simplify your explanation. Break it into "
        "small, very manageable steps. Be extra patient and encouraging. Avoid "
        "jargon. Validate that this part is genuinely tricky before helping."
    ),
    "fear": (
        "The student seems anxious or unsure of themselves. Go slow and be "
        "reassuring. Use a simple, low-pressure example first before anything "
        "harder. Remind them mistakes are a normal part of learning."
    ),
    "confusion": (
        "The student is confused. Re-explain the core idea in the simplest "
        "possible terms, ideally with a concrete everyday analogy, before any "
        "technical detail. Check understanding with a quick, easy question."
    ),
    "sadness": (
        "The student seems discouraged or low. Be gentle, patient, and warm. "
        "Focus on a small, achievable next step rather than the whole problem. "
        "Prioritize emotional support alongside the explanation."
    ),
    "disgust": (
        "The student seems put off or disengaged by this topic. Try a fresh "
        "angle or a more relatable, interesting example to re-spark curiosity, "
        "rather than repeating the same style of explanation."
    ),
    "surprise": (
        "The student encountered something unexpected. Clarify calmly, address "
        "the surprising part directly, and check what assumption might have "
        "led to the confusion."
    ),
    "joy": (
        "The student is in a good, confident mood. Match their energy. Don't "
        "over-simplify — give a fuller explanation, and add an interesting "
        "follow-up challenge or extension question to push their thinking."
    ),
    "neutral": (
        "The student seems calm and steady. Give a clear, well-paced "
        "explanation at a normal difficulty level, with room to go deeper if "
        "they engage further."
    ),
}

_FALLBACK_TEMPLATES = {
    "joy": "Love the energy! Let's build on that — here's a bit more depth, and a challenge to stretch you further.",
    "sadness": "That's okay, this feels tough right now — let's take it one small step at a time. You've got this.",
    "anger": "Totally fair to feel frustrated by this. Let's break it into smaller, easier pieces.",
    "fear": "No need to worry — we'll go slow, one small step at a time.",
    "surprise": "Let's slow down and unpack what just happened.",
    "disgust": "Let's try a different angle on this one — might click better.",
    "neutral": "Let's work through this step by step.",
    "confusion": "Let's untangle this piece by piece until it clicks.",
}


class GeminiGuidance:
    def __init__(self):
        self.api_key = get_gemini_api_key()
        self._model = None

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key)

    def _load_model(self):
        if self._model is None:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            # "gemini-flash-latest" is an auto-updating alias that always
            # points at Google's current recommended Flash model, so this
            # won't break again the next time Google retires a model version.
            self._model = genai.GenerativeModel("gemini-flash-latest")
        return self._model

    def generate_response(self, student_text: str, emotion: str,
                           chat_history: list | None = None) -> str:
        """
        chat_history: list of {"role": "user"/"assistant", "content": str}
        from earlier turns in the conversation, so Gemini keeps context and
        can notice mood shifts (e.g. "was frustrated, now sounds relieved").
        """
        if not self.is_configured:
            return self._fallback(student_text, emotion)

        strategy = _ADAPTATION_STRATEGY.get(emotion, _ADAPTATION_STRATEGY["neutral"])

        system_prompt = (
            f"{_BASE_PERSONA}\n\n"
            f"TEACHING ADAPTATION FOR THIS REPLY (internal guidance, do not "
            f"repeat this to the student): {strategy}"
        )

        try:
            model = self._load_model()
            convo = model.start_chat(history=self._to_gemini_history(chat_history or []))
            result = convo.send_message(f"{system_prompt}\n\nStudent just said: \"{student_text}\"")
            return result.text.strip()
        except Exception as e:
            return (
                f"{self._fallback(student_text, emotion)}\n\n"
                f"_(Gemini API call failed, showing a fallback response. Error: {e})_"
            )

    @staticmethod
    def _to_gemini_history(chat_history: list):
        history = []
        for turn in chat_history:
            role = "user" if turn["role"] == "user" else "model"
            history.append({"role": role, "parts": [turn["content"]]})
        return history

    def _fallback(self, student_text: str, emotion: str) -> str:
        return _FALLBACK_TEMPLATES.get(emotion, _FALLBACK_TEMPLATES["neutral"])
