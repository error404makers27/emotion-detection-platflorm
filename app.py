"""
LearnSense AI — Your Emotion-Aware AI Learning Companion
=============================================================
Frontend-only redesign. Backend (emotion_pipeline.py, gemini_client.py,
logger.py, preprocessing.py, config.py constants) is untouched — this file
only changes presentation, layout, and interaction flow.

Run with:  streamlit run app.py
"""

import time
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import config
from emotion_pipeline import EmotionPipeline
from gemini_client import GeminiGuidance
from logger import log_interaction, load_logs

# =============================================================================
# PAGE CONFIG + CACHED BACKEND HANDLES (unchanged backend, just wired in)
# =============================================================================
st.set_page_config(
    page_title="ZenLearn — Your Emotion-Aware AI Learning Companion",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

@st.cache_resource
def get_pipeline():
    return EmotionPipeline()

pipeline = get_pipeline()
# NOT cached: GeminiGuidance is cheap to create and must always read the
# current API key. Caching it here previously caused a bug where an old
# (or missing) key from the very first app start would be reused forever,
# even after fixing .env, since editing .env doesn't invalidate
# st.cache_resource.
gemini = GeminiGuidance()

# =============================================================================
# SESSION STATE
# =============================================================================
defaults = {
    "theme": "dark",
    "nav_page": "Home",
    "chat_history": [],          # [{"role","content","emotion","confidence","secondary","model","ts"}]
    "prefill_text": "",
    "domain": "Python",
    "topic": "",
    "difficulty": "Beginner",
    "session_start": time.time(),
    "last_context": None,        # holds most recent detection result for the right panel
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# =============================================================================
# DESIGN TOKENS + CSS
# =============================================================================
DOMAIN_OPTIONS = [
    "Python", "Java", "C", "C++", "Data Structures", "Algorithms",
    "Operating Systems", "Computer Networks", "DBMS", "Machine Learning",
    "Artificial Intelligence", "Deep Learning", "Cloud Computing",
    "Google Cloud", "Generative AI", "Prompt Engineering", "Web Development",
    "React", "Node.js", "JavaScript", "HTML", "CSS", "Cyber Security",
    "DevOps", "Placement Preparation", "Interview Preparation",
    "Communication Skills", "Other",
]

EMOTION_COLOR = {
    "joy": "#22D3A6", "neutral": "#5B8DEF", "sadness": "#6D7BFF",
    "anger": "#FF8A5B", "fear": "#C084FC", "surprise": "#FFC857",
    "disgust": "#F87171", "confusion": "#F0B429",
}

MOOD_ACK = {
    "anger": "I can see you've been putting in effort. Let's solve this one step at a time.",
    "fear": "You're not alone — many students feel this way before exams. Let's make a simple plan together.",
    "confusion": "I understand why this feels confusing. Let's simplify it together.",
    "sadness": "This is genuinely hard right now, and that's okay. Let's take one small step.",
    "disgust": "Let's try a completely different angle on this — might click better.",
    "surprise": "Let's slow down and unpack what just happened.",
    "joy": "That's fantastic! Let's build on your success with a slightly more challenging concept.",
    "neutral": "Let's dig into this together, step by step.",
}


def inject_css(theme: str):
    if theme == "dark":
        bg, bg2 = "#0B0F19", "#141227"
        glass, glass_border = "rgba(255,255,255,0.06)", "rgba(255,255,255,0.10)"
        text, text_muted = "#F1F0FA", "#A8A6C0"
        input_bg = "rgba(255,255,255,0.07)"
    else:
        bg, bg2 = "#FBF7FA", "#F1F4FC"
        glass, glass_border = "rgba(255,255,255,0.72)", "rgba(20,20,40,0.08)"
        text, text_muted = "#241F33", "#645E7A"
        input_bg = "rgba(20,20,40,0.045)"

    # Signature palette: light blue -> pink gradient
    accent1, accent2, accent3 = "#7EC8F2", "#FF9FCF", "#B694FF"

    # Light mode: give buttons a visible pink background so their text/icons
    # never get lost against the light page background (previously only
    # `color` was set, so buttons could render as near-invisible white-on-white).
    light_button_css = f"""
    .stButton > button {{
        background: linear-gradient(135deg, rgba(255,159,207,0.55), rgba(182,148,255,0.35));
        border: 1px solid rgba(255,159,207,0.85);
        color: {text} !important;
        font-weight: 600;
    }}
    .stButton > button:hover {{
        background: linear-gradient(135deg, rgba(255,159,207,0.75), rgba(182,148,255,0.5));
        border-color: {accent2};
        color: {text} !important;
    }}
    """ if theme == "light" else ""
    st.session_state["_accent1"], st.session_state["_accent2"] = accent1, accent2
    st.session_state["_text"], st.session_state["_text_muted"] = text, text_muted
    st.session_state["_glass_border"] = glass_border

    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Sora:wght@600;700;800&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}
    h1,h2,h3, .ls-display {{ font-family: 'Sora', sans-serif; }}
    .ls-mono {{ font-family: 'JetBrains Mono', monospace; }}

    .stApp {{
        background: radial-gradient(circle at 15% 0%, {bg2} 0%, {bg} 55%);
        color: {text};
    }}
    section[data-testid="stSidebar"] {{
        background: {bg2};
        border-right: 1px solid {glass_border};
    }}
    .ls-card {{
        background: {glass};
        border: 1px solid {glass_border};
        backdrop-filter: blur(14px);
        border-radius: 18px;
        padding: 1.1rem 1.3rem;
        box-shadow: 0 8px 30px rgba(0,0,0,0.12);
    }}
    .ls-muted {{ color: {text_muted}; }}

    /* Hero */
    .ls-hero-title {{
        font-size: 2.6rem; font-weight: 800; text-align:center;
        background: linear-gradient(90deg, {accent1}, {accent3} 55%, {accent2});
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0.15rem;
    }}
    .ls-tagline {{ text-align:center; font-size:1.05rem; color:{text_muted}; margin-bottom:1.6rem; }}

    /* Suggestion cards */
    .ls-suggestion {{
        background: {glass}; border:1px solid {glass_border}; border-radius:16px;
        padding: 1rem 1.1rem; transition: transform .15s ease, border-color .15s ease;
        backdrop-filter: blur(10px); cursor:pointer; color:{text};
    }}
    .ls-suggestion:hover {{ transform: translateY(-3px); border-color:{accent1}88; }}

    /* Chat bubbles */
    .ls-bubble-user {{
        background: linear-gradient(135deg,{accent1},{accent3});
        color:#151328; font-weight:500; padding:0.85rem 1.1rem; border-radius:18px 18px 4px 18px;
        max-width:78%; margin-left:auto; box-shadow:0 4px 18px rgba(126,200,242,0.25);
    }}
    .ls-bubble-ai {{
        background:{glass}; border:1px solid {glass_border}; backdrop-filter: blur(12px);
        color:{text}; padding:0.85rem 1.1rem; border-radius:18px 18px 18px 4px;
        max-width:78%; margin-right:auto;
    }}
    .ls-timestamp {{ font-size:0.72rem; color:{text_muted}; margin-top:0.2rem; }}

    /* Pulsing avatar ring (signature element) */
    .ls-avatar-ring {{
        width:38px; height:38px; border-radius:50%;
        display:flex; align-items:center; justify-content:center;
        font-size:1.1rem;
        box-shadow: 0 0 0 3px var(--ring-color, {accent1}88);
        animation: ls-pulse 2.4s ease-in-out infinite;
        background: {bg2};
    }}
    @keyframes ls-pulse {{
        0%   {{ box-shadow: 0 0 0 3px var(--ring-color, {accent1}88); }}
        50%  {{ box-shadow: 0 0 0 8px var(--ring-color, {accent1}33); }}
        100% {{ box-shadow: 0 0 0 3px var(--ring-color, {accent1}88); }}
    }}

    /* Sense panel readout */
    .ls-readout-row {{
        display:flex; justify-content:space-between; padding:0.4rem 0;
        border-bottom:1px solid {glass_border}; font-size:0.86rem; color:{text};
    }}
    .ls-readout-row:last-child {{ border-bottom:none; }}
    .ls-readout-label {{ color:{text_muted}; }}
    .ls-readout-value {{ font-family:'JetBrains Mono',monospace; font-weight:500; color:{text}; }}

    .ls-chip {{
        display:inline-block; background:{input_bg}; border:1px solid {glass_border};
        border-radius:999px; padding:0.4rem 0.9rem; margin:0.25rem 0.3rem 0.25rem 0;
        font-size:0.85rem; cursor:pointer; color:{text};
    }}

    /* Streamlit primary buttons -> accent gradient, readable text */
    .stButton>button[kind="primary"] {{
        background: linear-gradient(135deg,{accent1},{accent3});
        color:#151328; border:none; font-weight:600;
    }}
    .stButton>button {{ color:{text}; }}

    div[data-testid="stChatInput"] textarea {{ font-size: 0.95rem; }}

    {light_button_css}
    </style>
    """, unsafe_allow_html=True)


inject_css(st.session_state.theme)

# =============================================================================
# HELPERS
# =============================================================================

def run_turn(user_text: str, domain: str, topic: str, difficulty: str):
    """Runs emotion detection + Gemini generation, appends to chat_history."""
    contextual_note = f"[Domain: {domain}" + (f", Topic: {topic}" if topic else "") + f", Difficulty: {difficulty}] "
    full_input = contextual_note + user_text

    results = pipeline.analyze(full_input, use_bilstm=True)
    primary = pipeline.primary_result(results)

    ai_response = gemini.generate_response(
        student_text=user_text,
        emotion=primary.label,
        chat_history=[
            {"role": t["role"], "content": t["content"]} for t in st.session_state.chat_history
        ],
    )

    now = datetime.now().strftime("%H:%M")
    st.session_state.chat_history.append({
        "role": "user", "content": user_text, "ts": now,
    })
    st.session_state.chat_history.append({
        "role": "assistant", "content": ai_response, "ts": now,
        "emotion": primary.label, "confidence": primary.confidence,
        "secondary": primary.secondary_label, "model": primary.model_name,
        "domain": domain, "topic": topic or "—",
    })

    st.session_state.last_context = {
        "emotion": primary.label, "confidence": primary.confidence,
        "secondary": primary.secondary_label, "model": primary.model_name,
        "domain": domain, "topic": topic or "—", "difficulty": difficulty,
    }

    log_interaction(full_input, results["bert"], results["bilstm"], ai_response)


def compute_sidebar_stats():
    df = load_logs()
    if df.empty:
        return {"streak": 0, "mood": "—", "conversations": 0}
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    dates = sorted(df["timestamp"].dt.date.unique(), reverse=True)
    streak = 0
    today = datetime.now().date()
    expected = today
    for d in dates:
        if d == expected:
            streak += 1
            expected = expected.fromordinal(expected.toordinal() - 1)
        else:
            break
    today_rows = df[df["timestamp"].dt.date == today]
    mood = today_rows["bert_emotion"].mode()[0].capitalize() if not today_rows.empty else "New session"
    conversations = len(df)
    return {"streak": streak, "mood": mood, "conversations": conversations}


NAV_OPTIONS = ["🏠 Home", "💬 AI Chat", "📊 Analytics", "📚 Learning History", "⚙️ Settings", "ℹ️ About"]
if "nav_radio" not in st.session_state:
    st.session_state.nav_radio = NAV_OPTIONS[0]
if "pending_nav" not in st.session_state:
    st.session_state.pending_nav = None

# Apply any requested navigation BEFORE the radio widget is instantiated
# below. Streamlit forbids setting a widget's state after it has already
# been drawn in the same run, so navigation requests are queued via
# `go_to()` and applied here, at the top of the next run, instead.
if st.session_state.pending_nav is not None:
    st.session_state.nav_radio = st.session_state.pending_nav
    st.session_state.pending_nav = None


def go_to(page_label: str):
    """Queue a navigation request and rerun; applied safely on the next run."""
    st.session_state.pending_nav = page_label


# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.markdown("### 🧭 ZenLearn")
    st.caption("Your Emotion-Aware AI Learning Companion")
    st.markdown("---")

    nav_choice = st.radio(
        "Navigate", NAV_OPTIONS, key="nav_radio", label_visibility="collapsed",
    )
    st.session_state.nav_page = nav_choice.split(" ", 1)[1]

    st.markdown("---")
    stats = compute_sidebar_stats()
    c1, c2 = st.columns(2)
    c1.metric("🔥 Streak", f"{stats['streak']}d")
    c2.metric("💬 Chats", stats["conversations"])
    st.metric("🎭 Today's mood", stats["mood"])

    st.markdown("---")
    theme_toggle = st.toggle("🌙 Dark mode", value=(st.session_state.theme == "dark"))
    new_theme = "dark" if theme_toggle else "light"
    if new_theme != st.session_state.theme:
        st.session_state.theme = new_theme
        st.rerun()

    if not gemini.is_configured:
        st.warning("No Gemini key detected — using fallback responses.", icon="⚠️")

# =============================================================================
# PAGE: HOME
# =============================================================================
if st.session_state.nav_page == "Home":
    st.markdown('<div class="ls-hero-title">Welcome to LearnSense AI</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="ls-tagline">I don\'t just answer your questions — I sense how you\'re '
        'feeling while learning, and adapt my guidance to match.</div>',
        unsafe_allow_html=True,
    )

    cols = st.columns(3)
    suggestions = [
        ("📘", "Explain a difficult programming concept", "I've watched several tutorials but still don't fully understand recursion."),
        ("💼", "Help me prepare for placements", "I have placement interviews coming up and I'm not sure where to start."),
        ("🧠", "Create a personalized study plan", "Can you help me build a study plan for Data Structures and Algorithms?"),
    ]
    for col, (icon, title, prefill) in zip(cols, suggestions):
        with col:
            st.markdown(f'<div class="ls-suggestion"><div style="font-size:1.6rem">{icon}</div>'
                        f'<b>{title}</b></div>', unsafe_allow_html=True)
            if st.button("Start", key=f"sugg_{title}", use_container_width=True):
                st.session_state.prefill_text = prefill
                go_to("💬 AI Chat")
                st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("##### Or jump straight into a topic")
    chip_row = ["Explain Python", "Learn Recursion", "Help with Placements", "Understand DBMS",
                "Cloud Computing", "Google Gemini", "Machine Learning", "Prompt Engineering"]
    chip_cols = st.columns(4)
    for i, chip in enumerate(chip_row):
        with chip_cols[i % 4]:
            if st.button(chip, key=f"chip_{chip}", use_container_width=True):
                st.session_state.prefill_text = chip
                go_to("💬 AI Chat")
                st.rerun()

# =============================================================================
# PAGE: AI CHAT
# =============================================================================
elif st.session_state.nav_page == "AI Chat":
    st.markdown('<div class="ls-display" style="font-size:1.4rem; font-weight:700;">💬 AI Chat</div>',
                unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    if not st.session_state.chat_history:
        st.markdown("#### What would you like to learn today?")
        chips = ["Explain Python", "Learn Recursion", "Help with Placements",
                 "Understand DBMS", "Cloud Computing", "Machine Learning"]
        cc = st.columns(3)
        for i, chip in enumerate(chips):
            if cc[i % 3].button(chip, key=f"empty_chip_{chip}", use_container_width=True):
                st.session_state.prefill_text = chip

    # Render conversation — newest exchange on top, oldest at the bottom.
    # Messages are stored as [user, assistant, user, assistant, ...] pairs,
    # so we group them into pairs first and reverse the *pairs*, keeping the
    # user message before its own assistant reply within each pair.
    indexed_history = list(enumerate(st.session_state.chat_history))
    paired_history = [indexed_history[j:j + 2] for j in range(0, len(indexed_history), 2)]

    for pair in reversed(paired_history):
        for i, turn in pair:
            if turn["role"] == "user":
                st.markdown(
                    f'<div style="display:flex; justify-content:flex-end; margin:0.4rem 0;">'
                    f'<div><div class="ls-bubble-user">{turn["content"]}</div>'
                    f'<div class="ls-timestamp" style="text-align:right;">{turn["ts"]}</div></div></div>',
                    unsafe_allow_html=True,
                )
            else:
                ring = EMOTION_COLOR.get(turn.get("emotion", "neutral"), "#6D5BFF")
                st.markdown(
                    f'<div style="display:flex; gap:0.6rem; margin:0.4rem 0; align-items:flex-start;">'
                    f'<div class="ls-avatar-ring" style="--ring-color:{ring}88;">🧭</div>'
                    f'<div style="flex:1;"><div class="ls-bubble-ai">{turn["content"]}</div>'
                    f'<div class="ls-timestamp">{turn["ts"]}</div></div></div>',
                    unsafe_allow_html=True,
                )

                is_last = (i == len(st.session_state.chat_history) - 1)

                bcol1, bcol2, bcol3, bcol4 = st.columns([1, 1, 1, 4])
                with bcol1:
                    if st.button("🔁 Regenerate", key=f"regen_{i}"):
                        prev_user = st.session_state.chat_history[i - 1]["content"]
                        st.session_state.chat_history = st.session_state.chat_history[:i - 1]
                        run_turn(prev_user, st.session_state.domain, st.session_state.topic, st.session_state.difficulty)
                        st.rerun()
                with bcol2:
                    if st.button("👍", key=f"up_{i}"):
                        st.toast("Thanks for the feedback!")
                with bcol3:
                    if st.button("👎", key=f"down_{i}"):
                        st.toast("Thanks — I'll try a different approach next time.")

                # Native, reliable copy — Streamlit's code blocks have a built-in copy icon
                with st.expander("📋 Copy this response"):
                    st.code(turn["content"], language=None)

                # Interactive follow-up chips, ChatGPT-style, only on the latest reply
                if is_last:
                    fcol1, fcol2, fcol3 = st.columns(3)
                    if fcol1.button("💡 Give another example", key=f"more_ex_{i}", use_container_width=True):
                        with st.spinner("Thinking of another example..."):
                            run_turn("Can you give me another example?", st.session_state.domain,
                                     st.session_state.topic, st.session_state.difficulty)
                        st.rerun()
                    if fcol2.button("🔍 Explain more simply", key=f"simpler_{i}", use_container_width=True):
                        with st.spinner("Simplifying..."):
                            run_turn("Can you explain that more simply?", st.session_state.domain,
                                     st.session_state.topic, st.session_state.difficulty)
                        st.rerun()
                    if fcol3.button("✅ I get it now!", key=f"gotit_{i}", use_container_width=True):
                        with st.spinner("Great — leveling up..."):
                            run_turn("I understand now, can we try something a bit more challenging?",
                                     st.session_state.domain, st.session_state.topic, st.session_state.difficulty)
                        st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="ls-card">', unsafe_allow_html=True)
    st.markdown("**Ask LearnSense AI**")

    fc1, fc2, fc3 = st.columns(3)
    st.session_state.domain = fc1.selectbox("Learning Domain", DOMAIN_OPTIONS,
                                             index=DOMAIN_OPTIONS.index(st.session_state.domain))
    st.session_state.topic = fc2.text_input("Topic (optional)", value=st.session_state.topic,
                                             placeholder="e.g. Recursion, Pointers, Trees")
    st.session_state.difficulty = fc3.selectbox("Difficulty", ["Beginner", "Intermediate", "Advanced"],
                                                 index=["Beginner", "Intermediate", "Advanced"].index(st.session_state.difficulty))

    user_text = st.text_area(
        "Main Learning Challenge", value=st.session_state.prefill_text, height=110,
        placeholder='Describe your learning challenge in detail.\n\nExample: "I\'ve watched several '
                    'videos about recursion, but I still don\'t understand how it works. I\'m feeling '
                    'frustrated because my placement interviews are approaching."',
        label_visibility="collapsed",
    )

    b1, b2, b3 = st.columns([1, 1, 4])
    send = b1.button("📨 Send", type="primary", use_container_width=True)
    b2.button("🎙️ Voice", use_container_width=True, disabled=True, help="Voice input coming soon")
    if b3.button("🗑️ Clear conversation", use_container_width=False):
        st.session_state.chat_history = []
        st.session_state.last_context = None
        st.session_state.prefill_text = ""
        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

    if send and user_text.strip():
        st.session_state.prefill_text = ""
        with st.spinner("Thinking..."):
            run_turn(user_text.strip(), st.session_state.domain, st.session_state.topic, st.session_state.difficulty)
        st.rerun()

# =============================================================================
# PAGE: ANALYTICS
# =============================================================================
elif st.session_state.nav_page == "Analytics":
    st.markdown("## 📊 Analytics")
    df = load_logs()

    if df.empty:
        st.info("No interactions logged yet — start a chat to generate data!")
    else:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Messages", len(df))
        c2.metric("Avg Confidence", f"{df['bert_confidence'].mean():.0%}")
        c3.metric("Mixed Emotion Rate", f"{df['is_mixed'].mean():.0%}")
        most_common = df["bert_emotion"].mode()[0] if not df["bert_emotion"].empty else "-"
        c4.metric("Most Common Emotion", most_common.capitalize())

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Emotion Distribution**")
            emo_counts = df["bert_emotion"].value_counts().reset_index()
            emo_counts.columns = ["Emotion", "Count"]
            st.plotly_chart(px.pie(emo_counts, names="Emotion", values="Count", hole=0.45),
                             use_container_width=True)
        with col_b:
            st.markdown("**Confidence Over Time**")
            st.plotly_chart(
                px.line(df.sort_values("timestamp"), x="timestamp", y="bert_confidence", markers=True),
                use_container_width=True,
            )

        st.markdown("**Emotion Timeline**")
        st.plotly_chart(
            px.scatter(df.sort_values("timestamp"), x="timestamp", y="bert_emotion",
                       color="bert_emotion", size="bert_confidence"),
            use_container_width=True,
        )

        with st.expander("📄 Raw Interaction Log"):
            st.dataframe(df, use_container_width=True)
            st.download_button("Download CSV", data=df.to_csv(index=False).encode("utf-8"),
                                file_name="learnsense_logs.csv", mime="text/csv")

# =============================================================================
# PAGE: LEARNING HISTORY
# =============================================================================
elif st.session_state.nav_page == "Learning History":
    st.markdown("## 📚 Learning History")
    if not st.session_state.chat_history:
        st.info("No conversation yet in this session. Head to AI Chat to get started.")
    else:
        for turn in st.session_state.chat_history:
            role_label = "🧑‍🎓 You" if turn["role"] == "user" else "🧭 LearnSense AI"
            with st.expander(f"{role_label} · {turn['ts']}"):
                st.markdown(turn["content"])
                if turn["role"] == "assistant":
                    st.caption(f"Emotion: {turn.get('emotion','—')} · Domain: {turn.get('domain','—')} "
                               f"· Topic: {turn.get('topic','—')}")

# =============================================================================
# PAGE: SETTINGS
# =============================================================================
elif st.session_state.nav_page == "Settings":
    st.markdown("## ⚙️ Settings")
    st.markdown('<div class="ls-card">', unsafe_allow_html=True)
    st.markdown("**Appearance**")
    st.caption("Use the Dark mode toggle in the sidebar to switch themes.")
    st.markdown("---")
    st.markdown("**Gemini AI Status**")
    if gemini.is_configured:
        st.success("Gemini API key detected — full adaptive AI responses are active.")
    else:
        st.warning("No Gemini API key detected. Add `GEMINI_API_KEY` to your `.env` file "
                   "(see `.env.example`) to enable full AI-generated guidance.")
    st.markdown("---")
    st.markdown("**Custom BiLSTM Model**")
    if pipeline.bilstm.is_available:
        st.success("Custom BiLSTM model loaded — model comparison is active.")
    else:
        st.info("No custom BiLSTM model found in `/models`. Running on pretrained BERT only. "
                "Train one via `kaggle_training/train_bilstm.py`.")
    st.markdown('</div>', unsafe_allow_html=True)

# =============================================================================
# PAGE: ABOUT
# =============================================================================
elif st.session_state.nav_page == "About":
    st.markdown("## ℹ️ About LearnSense AI")
    st.markdown('<div class="ls-card">', unsafe_allow_html=True)
    st.markdown("""
**LearnSense AI** is an emotion-aware learning companion. It reads how you're
feeling from what you type — frustrated, confused, confident, anxious — and
adapts its teaching style accordingly: simpler and gentler when you're
stuck, more challenging when you're in flow.

**Under the hood:**
- **Emotion detection** — a pretrained BERT transformer (optionally compared against a custom-trained BiLSTM)
- **Adaptive guidance** — Google Gemini, prompted with teaching-style instructions based on your detected emotion
- **Analytics** — every interaction is logged locally so you (or your instructor) can track learning patterns over time

*AI responses may be inaccurate — always verify important information.*
    """)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<br><hr style='opacity:0.15'>", unsafe_allow_html=True)
st.caption("LearnSense AI · Your Emotion-Aware AI Learning Companion")
