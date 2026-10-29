"""
Utility functions for Marathi text handling, data loading, font resolution, and UI themes.
"""
import os
import json
import pandas as pd
from typing import List, Dict, Any, Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

DEVANAGARI_DIGITS = "०१२३४५६७८९"
ARABIC_DIGITS = "0123456789"
DEV_TO_ARABIC_MAP = {d: a for d, a in zip(DEVANAGARI_DIGITS, ARABIC_DIGITS)}
ARABIC_TO_DEV_MAP = {a: d for a, d in zip(ARABIC_DIGITS, DEVANAGARI_DIGITS)}

def devanagari_to_arabic_num(text: str) -> str:
    """Converts Devanagari numeral characters (०-९) to Arabic digits (0-9)."""
    return "".join(DEV_TO_ARABIC_MAP.get(ch, ch) for ch in text)

def arabic_to_devanagari_num(text: str) -> str:
    """Converts Arabic digits (0-9) to Devanagari numeral characters (०-९)."""
    return "".join(ARABIC_TO_DEV_MAP.get(ch, ch) for ch in text)

def is_devanagari_char(ch: str) -> bool:
    """Returns True if the character is in the Devanagari Unicode block (0x0900 to 0x097F)."""
    return 0x0900 <= ord(ch) <= 0x097F

def get_devanagari_font_path() -> Optional[str]:
    """Finds an available Devanagari-supporting TrueType font on Windows/System."""
    candidate_paths = [
        r"C:\Windows\Fonts\Nirmala.ttc",
        r"C:\Windows\Fonts\NirmalaB.ttc",
        r"C:\Windows\Fonts\mangal.ttf",
        r"C:\Windows\Fonts\aparaj.ttf",
        r"C:\Windows\Fonts\gargi.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    ]
    for p in candidate_paths:
        if os.path.exists(p):
            return p
    return None

def load_stopwords() -> List[str]:
    """Loads Marathi stopwords from data/stopwords_mr.txt."""
    path = os.path.join(DATA_DIR, "stopwords_mr.txt")
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        words = [line.strip() for line in f if line.strip()]
    return list(set(words))

def load_ingredients() -> List[Dict[str, Any]]:
    """Loads ingredients dictionary from data/ingredients.json."""
    path = os.path.join(DATA_DIR, "ingredients.json")
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("ingredients", [])

def load_cooking_actions() -> List[Dict[str, Any]]:
    """Loads cooking actions dictionary from data/cooking_actions.json."""
    path = os.path.join(DATA_DIR, "cooking_actions.json")
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("actions", [])

def load_recipes() -> List[Dict[str, Any]]:
    """Loads recipes list from data/recipes.json."""
    path = os.path.join(DATA_DIR, "recipes.json")
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_recipes_df() -> pd.DataFrame:
    """Loads recipes DataFrame from data/recipes.csv."""
    path = os.path.join(DATA_DIR, "recipes.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8")

def load_pos_tagged_corpus() -> pd.DataFrame:
    """Loads POS tagged training corpus from data/pos_tagged_corpus.csv."""
    path = os.path.join(DATA_DIR, "pos_tagged_corpus.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8")

def load_wsd_lexicon() -> Dict[str, Any]:
    """Loads WSD lexicon from data/wsd_lexicon.json."""
    path = os.path.join(DATA_DIR, "wsd_lexicon.json")
    if not os.path.exists(path):
        return {"target_words": []}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def apply_custom_theme(theme_mode: str = "dark") -> str:
    """
    Returns custom CSS for Dark / Light mode toggle with rich modern aesthetics,
    vibrant gradients, card glassmorphism, responsive tables, and typography.
    """
    if theme_mode.lower() == "dark":
        css = """
        <style>
        /* Dark Mode Theme */
        .stApp {
            background-color: #0b1120;
            color: #f1f5f9;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }
        .main-header {
            background: linear-gradient(135deg, #FF6B35 0%, #D9381E 50%, #9A031E 100%);
            padding: 1.8rem 2rem;
            border-radius: 16px;
            color: white;
            box-shadow: 0 10px 25px rgba(217, 56, 30, 0.25);
            margin-bottom: 1.5rem;
        }
        .main-header h1 {
            color: #ffffff !important;
            font-weight: 800 !important;
            margin: 0;
            font-size: 2.2rem;
        }
        .main-header p {
            color: #ffedd5 !important;
            margin-top: 0.5rem;
            font-size: 1.05rem;
        }
        .metric-card {
            background: #1e293b;
            border: 1px solid #334155;
            padding: 1.2rem;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
            text-align: center;
        }
        .metric-card h3 {
            margin: 0;
            color: #94a3b8;
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        .metric-card p {
            margin: 0.5rem 0 0 0;
            font-size: 1.8rem;
            font-weight: bold;
            color: #f97316;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background-color: #1e293b;
            padding: 6px;
            border-radius: 10px;
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 8px;
            color: #94a3b8;
            font-weight: 600;
        }
        .stTabs [aria-selected="true"] {
            background-color: #ff6b35 !important;
            color: white !important;
        }
        .info-pill {
            display: inline-block;
            background: #1e293b;
            border: 1px solid #ff6b35;
            color: #ffedd5;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.85rem;
            margin: 2px 4px;
        }
        .status-badge-marathi {
            background: #064e3b;
            color: #6ee7b7;
            padding: 3px 10px;
            border-radius: 12px;
            font-size: 0.8rem;
            font-weight: bold;
        }
        div[data-testid="stSidebar"] {
            background-color: #0f172a;
            border-right: 1px solid #1e293b;
        }
        </style>
        """
    else:
        css = """
        <style>
        /* Light Mode Theme */
        .stApp {
            background-color: #f8fafc;
            color: #0f172a;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }
        .main-header {
            background: linear-gradient(135deg, #ffedd5 0%, #fed7aa 50%, #fbd5c5 100%);
            padding: 1.8rem 2rem;
            border-radius: 16px;
            color: #7c2d12;
            border: 1px solid #fed7aa;
            box-shadow: 0 8px 20px rgba(249, 115, 22, 0.12);
            margin-bottom: 1.5rem;
        }
        .main-header h1 {
            color: #9a3412 !important;
            font-weight: 800 !important;
            margin: 0;
            font-size: 2.2rem;
        }
        .main-header p {
            color: #c2410c !important;
            margin-top: 0.5rem;
            font-size: 1.05rem;
        }
        .metric-card {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            padding: 1.2rem;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
            text-align: center;
        }
        .metric-card h3 {
            margin: 0;
            color: #64748b;
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        .metric-card p {
            margin: 0.5rem 0 0 0;
            font-size: 1.8rem;
            font-weight: bold;
            color: #ea580c;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background-color: #f1f5f9;
            padding: 6px;
            border-radius: 10px;
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 8px;
            color: #475569;
            font-weight: 600;
        }
        .stTabs [aria-selected="true"] {
            background-color: #f97316 !important;
            color: white !important;
        }
        .info-pill {
            display: inline-block;
            background: #fff7ed;
            border: 1px solid #f97316;
            color: #9a3412;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.85rem;
            margin: 2px 4px;
        }
        .status-badge-marathi {
            background: #dcfce7;
            color: #166534;
            padding: 3px 10px;
            border-radius: 12px;
            font-size: 0.8rem;
            font-weight: bold;
        }
        div[data-testid="stSidebar"] {
            background-color: #f1f5f9;
            border-right: 1px solid #e2e8f0;
        }
        </style>
        """
    return css

def init_app_page(page_title: str, page_icon: str = "🍲") -> str:
    """
    Initializes standard Streamlit page configuration, persists light/dark mode in session state,
    renders sidebar toggle, and applies theme stylesheet.
    """
    import streamlit as st
    try:
        st.set_page_config(
            page_title=f"{page_title} - RecipeParse",
            page_icon=page_icon,
            layout="wide",
            initial_sidebar_state="expanded"
        )
    except Exception:
        # Ignore if set_page_config was already called
        pass

    if "app_theme" not in st.session_state:
        st.session_state["app_theme"] = "dark"

    # Theme toggle in sidebar
    st.sidebar.markdown("### 🎨 थीम निवडा (Theme)")
    theme_choice = st.sidebar.radio(
        "Theme Mode",
        ["🌙 Dark Mode", "☀️ Light Mode"],
        index=0 if st.session_state["app_theme"] == "dark" else 1,
        key="_theme_toggle_radio",
        label_visibility="collapsed"
    )
    current_mode = "dark" if "Dark" in theme_choice else "light"
    st.session_state["app_theme"] = current_mode

    # Inject theme CSS
    st.markdown(apply_custom_theme(current_mode), unsafe_allow_html=True)

    # Sidebar project branding badge
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        <div style="padding: 10px; border-radius: 10px; background: rgba(255, 107, 53, 0.1); border: 1px solid #ff6b35; text-align: center;">
            <strong style="color: #ff6b35;">RecipeParse NLP</strong><br>
            <span style="font-size: 0.8em; opacity: 0.85;">मराठी पाककृती माहिती निष्कासन प्रणाली</span>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.sidebar.markdown("---")

    return current_mode

