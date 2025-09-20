# app.py
# Fetii Austin Data Copilot

import io
import os
import re
import datetime as dt
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

try:
    import requests
    from bs4 import BeautifulSoup
    SCRAPE_READY = True
except Exception:
    SCRAPE_READY = False

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

OPENAI_READY = bool(os.getenv("OPENAI_API_KEY") and OpenAI is not None)

PRIMARY_COLOR = "#0F766E"
ACCENT_COLOR = "#0EA5E9"
DOWNTOWN_LAT = 30.2672
DOWNTOWN_LON = -97.7431

DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTH_LOOKUP = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}

st.set_page_config(page_title="Fetii Austin Data Copilot", page_icon="FA", layout="wide")

st.markdown(
    f"""
    <style>
    .stApp {{
        background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
        color: #0f172a;
    }}
    .big-title {{
        font-size: 2.3rem;
        font-weight: 800;
        color: {PRIMARY_COLOR};
        margin-bottom: 0.2rem;
    }}
    .subtitle {{
        color: #475569;
        font-size: 1rem;
        margin-bottom: 1rem;
    }}
    .pill {{
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 999px;
        background: {ACCENT_COLOR}20;
        color: {ACCENT_COLOR};
        font-weight: 600;
        font-size: 0.8rem;
        margin-right: 0.4rem;
    }}
    .footer-note {{
        color: #64748b;
        font-size: 0.8rem;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

@dataclass
class QueryResponse:
    text: str
    kind: str
    table: Optional[pd.DataFrame] = None
    sample: Optional[pd.DataFrame] = None
    chart_type: Optional[str] = None
    chart_x: Optional[str] = None
    chart_y: Optional[str] = None
    chart_title: Optional[str] = None
    chart_sort: Optional[List[str]] = None


def standardize_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]
    return df


def rename_first(df: pd.DataFrame, candidates: List[str], target: str) -> None:
    for cand in candidates:
        if cand in df.columns:
            df.rename(columns={cand: target}, inplace=True)
            break


def make_age_bucket(age: Optional[float]) -> Optional[str]:
    if pd.isna(age):
        return None
    try:
        age_val = int(age)
    except Exception:
        return None
    bins = [(13, 17), (18, 24), (25, 34), (35, 44), (45, 54), (55, 64)]
    for lo, hi in bins:
        if lo <= age_val <= hi:
            return f"{lo}-{hi}"
    if age_val >= 65:
        return "65+"
    return None


def haversine_miles(lat1, lon1, lat2, lon2) -> float:
    if any(pd.isna(val) for val in [lat1, lon1, lat2, lon2]):
        return np.nan
    radius = 3958.8
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return float(2 * radius * np.arctan2(np.sqrt(a), np.sqrt(1 - a)))


def find_default_excel() -> str:
    candidates = [
        "FetiiAI_Data_Austin.xlsx",
        os.path.join("data", "FetiiAI_Data_Austin.xlsx"),
        "/mnt/data/FetiiAI_Data_Austin.xlsx",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError("FetiiAI_Data_Austin.xlsx")


@st.cache_data(show_spinner=False)
def load_excel(upload_bytes: Optional[bytes]) -> Dict[str, pd.DataFrame]:
    source: Any
    if upload_bytes:
        source = io.BytesIO(upload_bytes)
    else:
        source = find_default_excel()
    try:
        xls = pd.ExcelFile(source)
    except ImportError as exc:
        raise RuntimeError("Reading Excel files requires openpyxl. Install it with pip install openpyxl.") from exc
    except FileNotFoundError as exc:
        raise RuntimeError("Dataset not found. Upload FetiiAI_Data_Austin.xlsx via the sidebar.") from exc
    sheets: Dict[str, pd.DataFrame] = {}
    for name in xls.sheet_names:
        sheets[name.strip().lower()] = xls.parse(name)
    return sheets

# ... trimmed ...
