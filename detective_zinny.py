"""
Detective Zinny in India: The Cultural Hunt
A Streamlit educational detective game powered by Google Gemini.

Setup:
    pip install streamlit google-generativeai
    export GEMINI_API_KEY="your-key"        # or put it in .streamlit/secrets.toml
    streamlit run detective_zinny.py

Features:
  * Clues unlock one at a time (a wrong guess or a hint unlocks the next clue).
  * The next case is prepared in the background while you play, so
    "Next Case / Restart" starts almost instantly.
"""

import json
import os
import random
import re
from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher

import google.generativeai as genai
import streamlit as st

# ----------------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Detective Zinny in India", page_icon="🕵️", layout="centered")

# ----------------------------------------------------------------------------
# Game data
# ----------------------------------------------------------------------------
# name, kind, accepted aliases, one signature monument (used in Expert mode)
LOCATIONS = [
    {"name": "Jaipur", "kind": "city", "aliases": ["pink city"], "monument": "Hawa Mahal"},
    {"name": "Varanasi", "kind": "city", "aliases": ["benares", "banaras", "kashi"], "monument": "Dashashwamedh Ghat"},
    {"name": "Kolkata", "kind": "city", "aliases": ["calcutta"], "monument": "Victoria Memorial"},
    {"name": "Hyderabad", "kind": "city", "aliases": [], "monument": "Charminar"},
    {"name": "Chennai", "kind": "city", "aliases": ["madras"], "monument": "Kapaleeshwarar Temple"},
    {"name": "Mumbai", "kind": "city", "aliases": ["bombay"], "monument": "Gateway of India"},
    {"name": "Agra", "kind": "city", "aliases": [], "monument": "Taj Mahal"},
    {"name": "Amritsar", "kind": "city", "aliases": [], "monument": "Golden Temple"},
    {"name": "Mysuru", "kind": "city", "aliases": ["mysore"], "monument": "Mysore Palace"},
    {"name": "Udaipur", "kind": "city", "aliases": ["city of lakes"], "monument": "City Palace"},
    {"name": "Kochi", "kind": "city", "aliases": ["cochin"], "monument": "Chinese Fishing Nets"},
    {"name": "Madurai", "kind": "city", "aliases": [], "monument": "Meenakshi Amman Temple"},
    {"name": "Delhi", "kind": "city", "aliases": ["new delhi"], "monument": "Red Fort"},
    {"name": "Gujarat", "kind": "state", "aliases": [], "monument": "Rani ki Vav"},
    {"name": "Rajasthan", "kind": "state", "aliases": [], "monument": "Amer Fort"},
    {"name": "Kerala", "kind": "state", "aliases": [], "monument": "Padmanabhaswamy Temple"},
    {"name": "Odisha", "kind": "state", "aliases": ["orissa"], "monument": "Konark Sun Temple"},
    {"name": "Assam", "kind": "state", "aliases": [], "monument": "Kamakhya Temple"},
    {"name": "Punjab", "kind": "state", "aliases": [], "monument": "Jallianwala Bagh"},
    {"name": "Goa", "kind": "state", "aliases": [], "monument": "Basilica of Bom Jesus"},
    {"name": "West Bengal", "kind": "state", "aliases": ["bengal"], "monument": "Victoria Memorial"},
    {"name": "Tamil Nadu", "kind": "state", "aliases": [], "monument": "Brihadeeswarar Temple"},
]

# Attempts are 3 everywhere so a player who keeps guessing wrong sees all 3 clues.
DIFFICULTY = {
    "Easy": {"attempts": 3, "hints": 1, "mcq": True, "monument": False},
    "Hard": {"attempts": 3, "hints": 1, "mcq": False, "monument": False},
    "Expert": {"attempts": 3, "hints": 0, "mcq": False, "monument": True},
}

DIFFICULTY_STYLE = {
    "Easy": (
        "Use famous, widely known facts and landmarks. Keep the language friendly and clear. "
        "A well-read school student should be able to solve these."
    ),
    "Hard": (
        "Use deeper cultural and historical details (lesser-known dishes, historical events, "
        "local language features, folk arts). Do NOT name the location or any monument that "
        "gives it away outright."
    ),
    "Expert": (
        "Make the clues cryptic. Use linguistic wordplay, etymology, riddles, and one clue that "
        "involves a typical-climate or regional temperature comparison (for example, how its "
        "seasonal temperatures compare with another Indian region). One clue must point toward "
        "a specific monument without naming it. Never name the location or the monument."
    ),
}

CLUE_CATEGORIES = [
    "Clue 1 - Traditional Cuisine or Attire",
    "Clue 2 - Historical Events or Local Language",
    "Clue 3 - Regional Dance, Music, or Geography",
]

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
@st.cache_resource
def get_executor() -> ThreadPoolExecutor:
    """Small shared thread pool used to prepare the next case in the background."""
    return ThreadPoolExecutor(max_workers=2)


def normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\b(the|city of|state of)\b", "", text)
    return re.sub(r"\s+", " ", text).strip()


def fuzzy_match(guess: str, targets, threshold: float = 0.85) -> bool:
    g = normalize(guess)
    if not g:
        return False
    for t in targets:
        t = normalize(t)
        if g == t or SequenceMatcher(None, g, t).ratio() >= threshold:
            return True
    return False


def get_api_key() -> str:
    key = st.session_state.get("api_key_input", "").strip()
    if key:
        return key
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass
    return os.environ.get("GEMINI_API_KEY", "")


def parse_json(raw: str) -> dict:
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()
    return json.loads(raw)


def generate_case(location: dict, difficulty: str, api_key: str, model_name: str) -> dict:
    """Ask Gemini for three cultural clues plus a hint for the chosen location.

    NOTE: this runs in a background thread too, so it must not call any st.* function.
    """
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        model_name,
        generation_config={"temperature": 0.9, "response_mime_type": "application/json"},
    )

    prompt = f"""
You are the clue writer for an educational detective game called "Detective Zinny in India".
The criminal is hiding in this Indian {location['kind']}: {location['name']}.
Its signature monument is: {location['monument']}.
Difficulty: {difficulty}.
Style guide: {DIFFICULTY_STYLE[difficulty]}

Using accurate, up-to-date cultural knowledge, write exactly THREE clues, one per category:
1. {CLUE_CATEGORIES[0]}
2. {CLUE_CATEGORIES[1]}
3. {CLUE_CATEGORIES[2]}

Rules:
- Everything must be in English and factually accurate. Do not invent facts.
- NEVER write the name "{location['name']}" or its aliases in any clue or hint.
- Each clue is 1-3 sentences, written in the voice of a detective's case file.
- The clues are revealed to the player one at a time: make Clue 1 the most subtle, Clue 2 more specific, and Clue 3 the most revealing, but never name the answer.
- Also write ONE helpful hint that narrows things down, still without naming the answer.
- Also write a one-sentence "fun_fact" about the location, to show after the game.

Return ONLY valid JSON with this exact shape:
{{
  "clues": [
    {{"category": "Cuisine/Attire", "text": "..."}},
    {{"category": "History/Language", "text": "..."}},
    {{"category": "Dance/Music/Geography", "text": "..."}}
  ],
  "hints": ["..."],
  "fun_fact": "..."
}}
"""
    response = model.generate_content(prompt, request_options={"timeout": 60})
    data = parse_json(response.text)

    clues = data.get("clues", [])
    if len(clues) != 3:
        raise ValueError("Gemini did not return exactly three clues.")
    return {
        "clues": clues,
        "hints": data.get("hints", [])[:1],
        "fun_fact": data.get("fun_fact", ""),
    }


def start_prefetch(difficulty: str, api_key: str, model_name: str, avoid: str):
    """Begin generating the next case in the background."""
    pool = [l for l in LOCATIONS if l["name"] != avoid]
    location = random.choice(pool)
    future = get_executor().submit(generate_case, location, difficulty, api_key, model_name)
    st.session_state.prefetch = {
        "future": future,
        "location": location,
        "difficulty": difficulty,
        "model": model_name,
    }


def start_new_game(difficulty: str, model_name: str):
    st.session_state.pop("menu_error", None)
    api_key = get_api_key()
    if not api_key:
        st.session_state.menu_error = "Please provide your Gemini API key in the sidebar (or set GEMINI_API_KEY)."
        return

    location, case = None, None

    # 1) Use the case that was prepared in the background, if it matches.
    pf = st.session_state.get("prefetch")
    if pf and pf["difficulty"] == difficulty and pf["model"] == model_name:
        st.session_state.prefetch = None
        try:
            with st.spinner("Detective Zinny is finishing the case file..."):
                case = pf["future"].result(timeout=90)
            location = pf["location"]
        except Exception:
            location, case = None, None

    # 2) Otherwise generate a fresh case now.
    if case is None:
        pool = [l for l in LOCATIONS if l["name"] != st.session_state.get("last_location")]
        location = random.choice(pool)
        with st.spinner("Detective Zinny is gathering evidence from across India..."):
            try:
                case = generate_case(location, difficulty, api_key, model_name)
            except Exception as e:  # network, quota, bad key, bad JSON...
                st.session_state.menu_error = f"Could not generate the case file: {e}"
                return

    cfg = DIFFICULTY[difficulty]
    options = None
    if cfg["mcq"]:
        same_kind = [l["name"] for l in LOCATIONS if l["kind"] == location["kind"] and l["name"] != location["name"]]
        options = random.sample(same_kind, 3) + [location["name"]]
        random.shuffle(options)

    st.session_state.update(
        phase="playing",
        difficulty=difficulty,
        location=location,
        case=case,
        options=options,
        attempts_left=cfg["attempts"],
        hints_used=0,
        clues_shown=1,
        feedback="",
        last_location=location["name"],
    )

    # Prepare the NEXT case while the player works on this one.
    start_prefetch(difficulty, api_key, model_name, avoid=location["name"])


def check_guess(guess: str, monument_guess: str = ""):
    loc = st.session_state.location
    cfg = DIFFICULTY[st.session_state.difficulty]

    targets = [loc["name"]] + loc["aliases"]
    location_ok = fuzzy_match(guess, targets)
    monument_ok = True
    if cfg["monument"]:
        monument_ok = fuzzy_match(monument_guess, [loc["monument"]], threshold=0.8)

    if location_ok and monument_ok:
        st.session_state.phase = "won"
        return

    st.session_state.attempts_left -= 1
    if st.session_state.attempts_left <= 0:
        st.session_state.phase = "lost"
        return

    if cfg["monument"] and location_ok and not monument_ok:
        msg = "You have the right place, but the monument is wrong."
    else:
        msg = "That is not where the criminal is hiding."

    if st.session_state.clues_shown < 3:
        st.session_state.clues_shown += 1
        msg += " A new clue has been unlocked!"
    else:
        msg += " Re-read the clues, Watson."
    st.session_state.feedback = msg


# ----------------------------------------------------------------------------
# Session state defaults
# ----------------------------------------------------------------------------
st.session_state.setdefault("phase", "menu")

# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("Case settings")
    st.text_input(
        "Gemini API Key",
        type="password",
        key="api_key_input",
        help="Stored only in this session. You can also set GEMINI_API_KEY or use st.secrets.",
    )
    model_name = st.text_input("Gemini model", value="gemini-3.5-flash-lite")
    st.caption("Get a key at https://aistudio.google.com/apikey")

    st.divider()
    st.markdown("**Difficulty guide**")
    st.markdown(
        "- **Easy:** 4 choices, 1 hint, 3 attempts\n"
        "- **Hard:** type the location, 1 hint, 3 attempts\n"
        "- **Expert:** type location and monument, no hints, 3 attempts"
    )
    st.caption("You start with 1 clue. A wrong guess or a hint unlocks the next one.")

# ----------------------------------------------------------------------------
# Main UI
# ----------------------------------------------------------------------------
st.title("🕵️ Detective Zinny in India")
st.subheader("The Cultural Hunt")

phase = st.session_state.phase

if st.session_state.get("menu_error"):
    st.error(st.session_state.menu_error)

# ---- MENU ------------------------------------------------------------------
if phase == "menu":
    st.markdown(
        "A criminal is hiding somewhere in India. Decode **three cultural clues** about food, "
        "history, language, music, dance, and geography to find out where, before the trail goes cold!"
    )
    difficulty = st.radio("Choose your difficulty", list(DIFFICULTY.keys()), horizontal=True)
    if st.button("Open the case", type="primary"):
        start_new_game(difficulty, model_name)
        st.rerun()

# ---- PLAYING ---------------------------------------------------------------
elif phase == "playing":
    diff = st.session_state.difficulty
    cfg = DIFFICULTY[diff]
    case = st.session_state.case
    shown = st.session_state.clues_shown

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Difficulty", diff)
    col2.metric("Attempts left", st.session_state.attempts_left)
    col3.metric("Hints left", cfg["hints"] - st.session_state.hints_used)
    col4.metric("Clues unlocked", f"{shown}/3")

    st.markdown("### 📂 Case file")
    for i, clue in enumerate(case["clues"]):
        with st.container(border=True):
            st.markdown(f"**{CLUE_CATEGORIES[i]}**")
            if i < shown:
                st.write(clue["text"])
            elif cfg["hints"] > 0:
                st.caption("🔒 Locked: unlocks after a wrong guess or by using a hint.")
            else:
                st.caption("🔒 Locked: unlocks after a wrong guess.")

    # Hints
    if cfg["hints"] > 0:
        for i in range(st.session_state.hints_used):
            if i < len(case["hints"]):
                st.info(f"💡 Hint: {case['hints'][i]}")
        hints_left = cfg["hints"] - st.session_state.hints_used
        if hints_left > 0:
            if st.button(f"Use a hint ({hints_left} left): shows a hint and unlocks the next clue"):
                st.session_state.hints_used += 1
                st.session_state.clues_shown = min(3, st.session_state.clues_shown + 1)
                st.rerun()
    else:
        st.warning("Expert mode: no hints allowed.")

    if st.session_state.feedback:
        st.error(st.session_state.feedback)

    st.markdown("### 🔎 Where is the criminal hiding?")
    with st.form("guess_form", clear_on_submit=True):
        monument_guess = ""
        if cfg["mcq"]:
            guess = st.radio("Choose a location", st.session_state.options, index=None)
        else:
            guess = st.text_input("Type the exact city or state name")
            if cfg["monument"]:
                monument_guess = st.text_input("Name the specific monument")
        submitted = st.form_submit_button("Make the arrest", type="primary")

    if submitted:
        if not guess or (cfg["monument"] and not monument_guess.strip()):
            st.session_state.feedback = "Please fill in all required fields before making an arrest."
        else:
            check_guess(guess, monument_guess)
        st.rerun()

# ---- WON / LOST ------------------------------------------------------------
elif phase in ("won", "lost"):
    loc = st.session_state.location
    case = st.session_state.case

    if phase == "won":
        st.balloons()
        st.success("🎉 **Congratulations, Watson! You successfully tracked down the criminal!**")
    else:
        st.error("🚨 **The criminal escaped!**")
        st.markdown("# **GAME OVER**")

    with st.container(border=True):
        st.markdown(f"**The hideout was:** {loc['name']}")
        if st.session_state.difficulty == "Expert":
            st.markdown(f"**The monument was:** {loc['monument']}")
        if case.get("fun_fact"):
            st.markdown(f"**Case notes:** {case['fun_fact']}")

    with st.expander("Review all three clues"):
        for i, clue in enumerate(case["clues"]):
            st.markdown(f"**{CLUE_CATEGORIES[i]}**")
            st.write(clue["text"])

    st.markdown("### Do you want to play the next game?")
    c1, c2 = st.columns(2)
    if c1.button("Next Case / Restart", type="primary"):
        start_new_game(st.session_state.difficulty, model_name)
        st.rerun()
    if c2.button("Change difficulty"):
        st.session_state.phase = "menu"
        st.rerun()
