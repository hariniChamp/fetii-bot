# app.py
# Fetii Austin Data Copilot — with AI (GPT->SQL) chat over your dataset.

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

# Optional web enrichment
try:
    import requests
    from bs4 import BeautifulSoup
    SCRAPE_READY = True
except Exception:
    SCRAPE_READY = False

# OpenAI (optional for AI mode)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

OPENAI_READY = bool(os.getenv("OPENAI_API_KEY") and OpenAI is not None)

# SQL engine (DuckDB) for free-form AI queries
try:
    import duckdb
    DUCK_READY = True
except Exception:
    duckdb = None
    DUCK_READY = False

# Branding / constants
PRIMARY_COLOR = "#0F766E"
ACCENT_COLOR = "#0EA5E9"
DOWNTOWN_LAT = 30.2672
DOWNTOWN_LON = -97.7431

DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTH_LOOKUP = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
}

st.set_page_config(page_title="Fetii Austin Data Copilot", page_icon="🚐", layout="wide")
st.markdown(
    f"""
    <style>
    .stApp {{ background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%); color:#0f172a; }}
    .big-title {{ font-size:2.3rem; font-weight:800; color:{PRIMARY_COLOR}; margin-bottom:0.2rem; }}
    .subtitle {{ color:#475569; font-size:1rem; margin-bottom:1rem; }}
    .footer-note {{ color:#64748b; font-size:.8rem; }}
    code.sql {{ background:#F1F5F9; padding:.25rem .35rem; border-radius:.35rem; }}
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
    debug_sql: Optional[str] = None

# -------------------- utilities & loading --------------------

def standardize_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df

def rename_first(df: pd.DataFrame, candidates: List[str], target: str) -> None:
    lower_map = {str(col).lower(): col for col in df.columns}
    for cand in candidates:
        col = lower_map.get(str(cand).lower())
        if col:
            df.rename(columns={col: target}, inplace=True)
            break

def make_age_bucket(age: Optional[float]) -> Optional[str]:
    if pd.isna(age): return None
    try: age = int(age)
    except Exception: return None
    for lo, hi in [(13,17),(18,24),(25,34),(35,44),(45,54),(55,64)]:
        if lo <= age <= hi: return f"{lo}-{hi}"
    return "65+" if age >= 65 else None

def haversine_miles(lat1, lon1, lat2, lon2) -> float:
    if any(pd.isna(v) for v in [lat1, lon1, lat2, lon2]): return np.nan
    R = 3958.8
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1); dl = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(p1)*np.cos(p2)*np.sin(dl/2)**2
    return float(2*R*np.arctan2(np.sqrt(a), np.sqrt(1-a)))

def find_default_excel() -> str:
    for p in ["FetiiAI_Data_Austin.xlsx", os.path.join("data","FetiiAI_Data_Austin.xlsx"), "/mnt/data/FetiiAI_Data_Austin.xlsx"]:
        if os.path.exists(p): return p
    raise FileNotFoundError("FetiiAI_Data_Austin.xlsx")

@st.cache_data(show_spinner=False)
def load_excel(upload_bytes: Optional[bytes]) -> Dict[str, pd.DataFrame]:
    src: Any = io.BytesIO(upload_bytes) if upload_bytes else find_default_excel()
    try:
        xls = pd.ExcelFile(src)
    except ImportError as exc:
        raise RuntimeError("Install openpyxl:  pip install openpyxl") from exc
    sheets: Dict[str, pd.DataFrame] = {}
    for name in xls.sheet_names:
        sheets[name.strip().lower()] = xls.parse(name)
    return sheets

def build_master_frames(sheets: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    def find_sheet(cands: List[str]) -> Optional[str]:
        for c in cands:
            for n in sheets.keys():
                if c.lower() == n or c.lower() in n:
                    return n
        return None

    trip_key = find_sheet(["trip data","trips","trip"])
    if not trip_key: raise ValueError("Trip sheet missing")
    trips = standardize_frame(sheets[trip_key])
    rider_key = find_sheet(["rider data","riders","passengers"])
    riders = standardize_frame(sheets[rider_key]) if rider_key else pd.DataFrame()
    demo_key = find_sheet(["ride demo","demographics","user demo","demo"])
    demo = standardize_frame(sheets[demo_key]) if demo_key else pd.DataFrame()

    for f in [trips, riders]:
        rename_first(f, ["Trip ID","trip_id","TripId","TripID"], "TripID")
    rename_first(trips, ["User ID","user_id","UserId","Booking User ID"], "UserID")
    rename_first(riders, ["User ID","user_id","UserId"], "UserID")
    rename_first(demo, ["User ID","user_id","UserId"], "UserID")

    rename_first(trips, ["party_size","Party Size","num_riders","Riders","how many users rode in the fetii","users_count","Total Passengers"], "PartySize")
    rename_first(trips, ["pickup address","Pickup address","pickup_address","Pick Up Address"], "PickupAddress")
    rename_first(trips, ["drop off address","Drop off address","dropoff address","dropoff Address","drop_off_address","Drop Off Address"], "DropoffAddress")
    rename_first(trips, ["pickup latitude","Pickup latitude","pickup_lat","Pick Up Latitude"], "PickupLat")
    rename_first(trips, ["pickup longitude","Pickup longitude","pickup_lon","Pick Up Longitude"], "PickupLon")
    rename_first(trips, ["drop off latitude","Drop off latitude","dropoff latitude","dropoff_lat","Drop Off Latitude"], "DropoffLat")
    rename_first(trips, ["drop off longitude","Drop off longitude","dropoff longitude","dropoff_lon","Drop Off Longitude"], "DropoffLon")
    rename_first(trips, ["timestamp","Timestamp","pickup time","Pickup Time","pickup_timestamp","start_time","Start Time","Trip Date and Time"], "DateTime")

    if "DateTime" in trips.columns:
        trips["DateTime"] = pd.to_datetime(trips["DateTime"], errors="coerce")
    if "DropoffTime" in trips.columns:
        trips["DropoffTime"] = pd.to_datetime(trips["DropoffTime"], errors="coerce")

    # enrich time parts
    if "DateTime" not in trips.columns:
        trips["DateTime"] = pd.to_datetime(trips.get("PickupTime"), errors="coerce")
        if trips["DateTime"].isna().all() and "DropoffTime" in trips.columns:
            trips["DateTime"] = pd.to_datetime(trips["DropoffTime"], errors="coerce")
    if "DateTime" not in trips.columns or trips["DateTime"].isna().all():
        trips["DateTime"] = pd.to_datetime(trips.get("DateTime"), errors="coerce")
    trips["Year"] = trips["DateTime"].dt.year
    trips["Month"] = trips["DateTime"].dt.month
    trips["MonthName"] = trips["DateTime"].dt.strftime("%B")
    trips["Day"] = trips["DateTime"].dt.day
    trips["Hour"] = trips["DateTime"].dt.hour
    trips["DOW"] = trips["DateTime"].dt.day_name()

    # venues + distances
    if "DropoffAddress" in trips.columns:
        dropoff_series = trips["DropoffAddress"].fillna("").astype(str).str.strip()
    else:
        dropoff_series = pd.Series("", index=trips.index, dtype="object")
    if "PickupAddress" in trips.columns:
        pickup_series = trips["PickupAddress"].fillna("").astype(str).str.strip()
    else:
        pickup_series = pd.Series("", index=trips.index, dtype="object")
    trips["DropoffAddress"] = dropoff_series
    trips["PickupAddress"] = pickup_series
    trips["DropoffSimple"] = trips["DropoffAddress"].apply(lambda v: v.split(",")[0].strip() if v else "(unknown)")
    trips["PickupSimple"] = trips["PickupAddress"].apply(lambda v: v.split(",")[0].strip() if v else "(unknown)")
    trips["IsWeekend"] = trips["DOW"].isin(["Friday","Saturday","Sunday"])
    for c in ["PickupLat","PickupLon","DropoffLat","DropoffLon"]:
        if c in trips.columns: trips[c] = pd.to_numeric(trips[c], errors="coerce")
    if {"DropoffLat","DropoffLon"}.issubset(trips.columns):
        trips["DropoffToDowntownMi"] = trips.apply(
            lambda r: haversine_miles(r["DropoffLat"], r["DropoffLon"], DOWNTOWN_LAT, DOWNTOWN_LON), axis=1
        )
    else:
        trips["DropoffToDowntownMi"] = np.nan
    if {"PickupLat","PickupLon"}.issubset(trips.columns):
        trips["PickupToDowntownMi"] = trips.apply(
            lambda r: haversine_miles(r["PickupLat"], r["PickupLon"], DOWNTOWN_LAT, DOWNTOWN_LON), axis=1
        )
    else:
        trips["PickupToDowntownMi"] = np.nan

    # age enrichment
    demo_small = pd.DataFrame(columns=["UserID","Age","AgeBucket"])
    if not demo.empty and "UserID" in demo.columns:
        rename_first(demo, ["age","Age","user_age"], "Age")
        demo["Age"] = pd.to_numeric(demo.get("Age"), errors="coerce")
        demo["AgeBucket"] = demo["Age"].map(make_age_bucket)
        demo_small = demo[["UserID","Age","AgeBucket"]].drop_duplicates()

    if not riders.empty and {"TripID","UserID"}.issubset(riders.columns):
        riders_enriched = riders.merge(demo_small, on="UserID", how="left")
    else:
        riders_enriched = pd.DataFrame(columns=["TripID","UserID","Age","AgeBucket"])

    if not riders_enriched.empty:
        agg = (riders_enriched.groupby("TripID")
               .agg(PassengerCount=("UserID","count"),
                    AvgAge=("Age","mean"),
                    TripAgeMin=("Age","min"),
                    TripAgeMax=("Age","max"),
                    DominantAgeBucket=("AgeBucket", lambda s: s.dropna().mode().iat[0] if not s.dropna().empty else None))
               .reset_index())
    else:
        agg = pd.DataFrame(columns=["TripID","PassengerCount","AvgAge","TripAgeMin","TripAgeMax","DominantAgeBucket"])

    trips = trips.merge(agg, on="TripID", how="left")
    if "PartySize" in trips.columns:
        trips["PartySize"] = pd.to_numeric(trips["PartySize"], errors="coerce")
    else:
        trips["PartySize"] = pd.Series(np.nan, index=trips.index, dtype="float")
    if "PassengerCount" in trips.columns:
        trips["PartySize"] = trips["PartySize"].fillna(trips["PassengerCount"])
    trips["IsLargeGroup"] = trips["PartySize"] >= 6

    return {"trips": trips, "riders": riders_enriched, "demo": demo_small}

# -------------------- parsing helpers (for AI filter hints) --------------------

def parse_age_range(text: str) -> Optional[Tuple[int,int]]:
    pats = [r"ages?\s*(\d{1,2})\s*(?:-|to|through)\s*(\d{1,2})",
            r"(\d{1,2})\s*(?:-|to|through)\s*(\d{1,2})\s*(?:year|yr|yo)",
            r"(\d{1,2})\s*(?:-|to|through)\s*(\d{1,2})"]
    t = text.lower()
    for p in pats:
        m = re.search(p, t)
        if m: 
            a,b = int(m.group(1)), int(m.group(2))
            return (min(a,b), max(a,b))
    return None

def parse_day_token(text: str) -> Optional[str]:
    t = text.lower()
    if "weekend" in t: return "weekend"
    if "weekday" in t: return "weekday"
    for d in DAY_NAMES:
        if d.lower() in t or d[:3].lower() in t: return d
    return None

def parse_daypart(text: str) -> Optional[str]:
    t = text.lower()
    if any(k in t for k in ["night","late","after dark"]): return "night"
    if "morning" in t: return "morning"
    if "afternoon" in t: return "afternoon"
    if any(k in t for k in ["evening","happy hour"]): return "evening"
    return None

def parse_group_threshold(text: str) -> Optional[int]:
    t = text.lower()
    m = re.search(r"(\d+)\s*\+", t) or re.search(r"(\d+)\s*(?:or more|and up|plus)", t) \
        or re.search(r"(?:groups?|parties|riders)\s*(?:of|over|above|greater than)\s*(\d+)", t)
    if m: return int(m.group(1))
    if "large group" in t or "large groups" in t: return 6
    return None

def month_bounds(y: int, m: int) -> Tuple[dt.datetime, dt.datetime]:
    start = dt.datetime(y, m, 1)
    end = dt.datetime(y + (m // 12), (m % 12) + 1, 1)
    return start, end

def parse_timeframe(text: str, now: dt.datetime) -> Optional[Tuple[dt.datetime, dt.datetime, str]]:
    t = text.lower()
    ref = now.replace(hour=0, minute=0, second=0, microsecond=0)
    if "last month" in t:
        cur_start = dt.datetime(ref.year, ref.month, 1)
        prev_last = cur_start - dt.timedelta(days=1)
        s,e = month_bounds(prev_last.year, prev_last.month)
        return s,e,f"{prev_last.strftime('%B')} {prev_last.year}"
    if "this month" in t or "current month" in t:
        s,e = month_bounds(ref.year, ref.month)
        return s,e,f"{ref.strftime('%B')} {ref.year}"
    m = re.search(r"(january|february|march|april|may|june|july|august|september|october|november|december)\s*(\d{4})?", t)
    if m:
        mon = MONTH_LOOKUP[m.group(1)]; year = int(m.group(2)) if m.group(2) else ref.year
        s,e = month_bounds(year, mon); return s,e,f"{m.group(1).title()} {year}"
    if "last week" in t or "past week" in t:
        sow = ref - dt.timedelta(days=ref.weekday()); end = sow; start = end - dt.timedelta(days=7)
        return start, end, "last week"
    if "this week" in t or "current week" in t:
        start = ref - dt.timedelta(days=ref.weekday()); end = start + dt.timedelta(days=7)
        return start, end, "this week"
    m = re.search(r"last\s+(\d{1,2})\s+days", t) or re.search(r"past\s+(\d{1,2})\s+days", t)
    if m:
        days = int(m.group(1)); end = ref + dt.timedelta(days=1); start = end - dt.timedelta(days=days)
        return start, end, f"last {days} days"
    return None



def apply_timeframe_filter(df: pd.DataFrame, timeframe: Optional[Tuple[dt.datetime, dt.datetime, str]]) -> Tuple[pd.DataFrame, Optional[str]]:
    """Filter trips by timeframe tuple (start, end, label)."""
    if timeframe is None or "DateTime" not in df.columns:
        return df, None
    start, end, label = timeframe
    mask = df["DateTime"].notna()
    if start is not None:
        mask &= df["DateTime"] >= start
    if end is not None:
        mask &= df["DateTime"] < end
    return df[mask], label


def filter_by_age_range(df: pd.DataFrame, rng: Optional[Tuple[int, int]]) -> pd.DataFrame:
    if not rng:
        return df
    lo, hi = rng
    mask = pd.Series(False, index=df.index)
    if {"TripAgeMin", "TripAgeMax"}.issubset(df.columns):
        mask |= (
            df["TripAgeMin"].notna()
            & df["TripAgeMax"].notna()
            & (df["TripAgeMin"] <= hi)
            & (df["TripAgeMax"] >= lo)
        )
    if "AvgAge" in df.columns:
        mask |= df["AvgAge"].between(lo, hi, inclusive="both")
    if "DominantAgeBucket" in df.columns:
        mask |= df["DominantAgeBucket"].fillna("").str.contains(f"{lo}-{hi}")
    return df[mask]


def apply_day_filters(df: pd.DataFrame, day_token: Optional[str], daypart: Optional[str]) -> pd.DataFrame:
    result = df
    if day_token == "weekend":
        result = result[result["IsWeekend"]]
    elif day_token == "weekday":
        result = result[~result["IsWeekend"]]
    elif day_token in DAY_NAMES:
        result = result[result["DOW"] == day_token]
    if daypart and "Hour" in result.columns:
        hours = result["Hour"].astype('Int64')
        if daypart == "night":
            result = result[(hours >= 20) | (hours <= 2)]
        elif daypart == "morning":
            result = result[(hours >= 5) & (hours <= 11)]
        elif daypart == "afternoon":
            result = result[(hours >= 12) & (hours <= 16)]
        elif daypart == "evening":
            result = result[(hours >= 17) & (hours <= 21)]
    return result


def filter_by_group_threshold(df: pd.DataFrame, threshold: Optional[int]) -> pd.DataFrame:
    if not threshold:
        return df
    size = pd.to_numeric(df.get("PartySize"), errors="coerce")
    if "PassengerCount" in df.columns:
        size = size.fillna(df["PassengerCount"])
    return df[size >= threshold]


def filter_destination_keyword(df: pd.DataFrame, keyword: str, column: str) -> pd.DataFrame:
    keyword_lower = keyword.lower()
    mask = df[column].astype(str).str.contains(keyword_lower, case=False, na=False)
    if keyword_lower == "downtown":
        dist_col = "DropoffToDowntownMi" if column.startswith("Dropoff") else "PickupToDowntownMi"
        if dist_col in df.columns:
            mask = mask | (df[dist_col] <= 1.5)
    return df[mask]


def format_sample(df: pd.DataFrame, columns: List[str], limit: int = 25) -> Optional[pd.DataFrame]:
    cols = [c for c in columns if c in df.columns]
    if not cols or df.empty:
        return None
    sort_col = "DateTime" if "DateTime" in df.columns else None
    ordered = df.sort_values(sort_col, ascending=False) if sort_col else df
    return ordered.head(limit)[cols]

def _df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

# -------------------- AI SQL utilities --------------------

def df_schema(df: pd.DataFrame) -> List[str]:
    out = []
    for c, d in df.dtypes.items():
        t = str(d)
        if "datetime" in t: t = "timestamp"
        elif "int" in t: t = "integer"
        elif "float" in t or "double" in t: t = "float"
        elif "bool" in t: t = "boolean"
        else: t = "text"
        out.append(f"{c} {t}")
    return out

def schema_prompt(frames: Dict[str,pd.DataFrame]) -> str:
    trips, riders, demo = frames["trips"], frames["riders"], frames["demo"]
    s = []
    s.append("Tables and columns (DuckDB SQL):")
    s.append("trips(" + ", ".join(df_schema(trips)) + ")")
    s.append("riders(" + ", ".join(df_schema(riders)) + ")")
    s.append("demo("   + ", ".join(df_schema(demo))   + ")")
    s.append("\nSemantics:")
    s.append("- DateTime is the primary timestamp for a trip; Hour, DOW, Month, Year are derived.")
    s.append("- PartySize is riders in the booking; PassengerCount is riders joined via riders table; use COALESCE(PartySize, PassengerCount).")
    s.append("- Use DropoffAddress/DropoffSimple and PickupAddress/PickupSimple for place matching (case-insensitive LIKE '%keyword%').")
    s.append("- Downtown filter: DropoffToDowntownMi or PickupToDowntownMi <= 1.5 approximately matches 'downtown'.")
    s.append("- Age filters: prefer TripAgeMin/TripAgeMax overlap OR AvgAge BETWEEN lo AND hi; DominantAgeBucket like '18-24' as backup.")
    return "\n".join(s)

def hint_filters(question: str, now: dt.datetime) -> List[str]:
    hints = []
    tf = parse_timeframe(question, now)
    if tf:
        s,e,_ = tf
        hints.append(f"(DateTime >= TIMESTAMP '{s:%Y-%m-%d} 00:00:00' AND DateTime < TIMESTAMP '{e:%Y-%m-%d} 00:00:00')")
    th = parse_group_threshold(question)
    if th:
        hints.append(f"(COALESCE(PartySize, PassengerCount) >= {th})")
    day = parse_day_token(question)
    if day == "weekend":
        hints.append("(DOW IN ('Friday','Saturday','Sunday'))")
    elif day == "weekday":
        hints.append("(DOW NOT IN ('Friday','Saturday','Sunday'))")
    elif day in DAY_NAMES:
        hints.append(f"(DOW = '{day}')")
    dp = parse_daypart(question)
    if dp:
        if dp == "night": hints.append("(Hour >= 20 OR Hour <= 2)")
        if dp == "morning": hints.append("(Hour BETWEEN 5 AND 11)")
        if dp == "afternoon": hints.append("(Hour BETWEEN 12 AND 16)")
        if dp == "evening": hints.append("(Hour BETWEEN 17 AND 21)")  # <-- FIXED LINE
    age = parse_age_range(question)
    if age:
        lo, hi = age
        hints.append(f"((TripAgeMin <= {hi} AND TripAgeMax >= {lo}) OR (AvgAge BETWEEN {lo} AND {hi}) OR (DominantAgeBucket LIKE '%{lo}-{hi}%'))")
    return hints

def build_llm_sql(question: str, frames: Dict[str,pd.DataFrame], now: dt.datetime) -> Optional[str]:
    if not OPENAI_READY: return None
    client = OpenAI()

    schema = schema_prompt(frames)
    filters = hint_filters(question, now)
    must_filter = " AND ".join(filters) if filters else ""

    guidance = (
        "Write a single DuckDB SQL SELECT query that answers the user's question using the tables.\n"
        "- Prefer trips as the main table; join riders/demo only if necessary.\n"
        "- For venue keywords, use case-insensitive LIKE on DropoffAddress or DropoffSimple by default, "
        "unless the user says 'pickup' then use PickupAddress/PickupSimple.\n"
        "- If the user says 'downtown', treat it as DropoffToDowntownMi <= 1.5 (or PickupToDowntownMi for pickup).\n"
        "- If grouping, return two columns: label and value (name them appropriately). "
        "If counting, return one row with a single numeric column named count.\n"
        "- Only return the SQL. No prose, no code fences."
    )
    hint_block = f"\nMANDATORY EXTRA FILTERS (append with AND in WHERE if WHERE exists, or use WHERE if none):\n{must_filter}\n" if must_filter else ""

    user = f"QUESTION:\n{question}\n\n{schema}\n\n{guidance}{hint_block}"
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.1,
            messages=[
                {"role":"system","content":"You are a precise SQL generator for DuckDB. Output only SQL."},
                {"role":"user","content":user},
            ],
        )
        sql = resp.choices[0].message.content.strip()
        sql = re.sub(r"^```sql|^```|```$", "", sql, flags=re.IGNORECASE|re.MULTILINE).strip()
        if not sql.lower().startswith("select"):
            return None
        if must_filter and " where " in sql.lower():
            sql = re.sub(r"(?i)\bwhere\b", f"WHERE {must_filter} AND ", sql, count=1)
        elif must_filter:
            sql += f" WHERE {must_filter}"
        return sql
    except Exception:
        return None

def run_duckdb_query(frames: Dict[str,pd.DataFrame], sql: str) -> pd.DataFrame:
    con = duckdb.connect()
    con.register("trips", frames["trips"])
    con.register("riders", frames["riders"])
    con.register("demo",  frames["demo"])
    return con.execute(sql).df()

def infer_chart(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[List[str]]]:
    if df is None or df.empty: return None, None, None, None
    if df.shape[1] == 2:
        cols = list(df.columns)
        num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
        cat_cols = [c for c in cols if c not in num_cols]
        if len(num_cols)==1 and len(cat_cols)==1:
            return "bar", cat_cols[0], num_cols[0], None
    if "Hour" in df.columns and "Trips" in df.columns:
        return "bar", "Hour", "Trips", None
    if "Day" in df.columns and "Trips" in df.columns:
        return "bar", "Day", "Trips", DAY_NAMES
    return None, None, None, None

def polish_text(question: str, draft: str, scraped: str) -> str:
    if not OPENAI_READY or OpenAI is None: return draft
    try:
        client = OpenAI()
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.2,
            messages=[
                {"role":"system","content":"Concise analyst. Keep numbers as-is; do not invent."},
                {"role":"user","content":f"Q:\n{question}\n\nDraft:\n{draft}\n\nContext (optional):\n{scraped[:2000]}"},
            ],
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return draft

# -------------------- Engine --------------------



class QueryEngine:
    def __init__(self, frames: Dict[str, pd.DataFrame], scraped_text: str):
        self.frames = frames
        self.trips = frames["trips"]
        self.scraped = scraped_text
        self.now = dt.datetime.now()
        self.handlers = [
            self.handle_count_to_place,
            self.handle_top_destinations,
            self.handle_large_group_hours,
            self.handle_trips_by_day,
            self.handle_trips_by_hour,
            self.handle_total_trips,
        ]

    def _apply_filters(self, question: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        df = self.trips.copy()
        meta: Dict[str, Any] = {"descriptors": []}

        day_token = parse_day_token(question)
        daypart = parse_daypart(question)
        df = apply_day_filters(df, day_token, daypart)
        meta["day_token"] = day_token
        meta["daypart"] = daypart
        if day_token:
            meta["descriptors"].append(day_token)
        if daypart:
            meta["descriptors"].append(daypart)

        age_range = parse_age_range(question)
        meta["age_range"] = age_range
        if age_range:
            df = filter_by_age_range(df, age_range)
            meta["descriptors"].append(f"ages {age_range[0]}-{age_range[1]}")

        threshold = parse_group_threshold(question)
        meta["threshold"] = threshold
        if threshold:
            df = filter_by_group_threshold(df, threshold)
            meta["descriptors"].append(f"{threshold}+ riders")

        meta["pre_timeframe_df"] = df.copy()

        timeframe = parse_timeframe(question, self.now)
        meta["timeframe"] = timeframe
        df, timeframe_label = apply_timeframe_filter(df, timeframe)
        meta["timeframe_label_requested"] = timeframe_label
        meta["timeframe_label"] = timeframe_label
        if timeframe_label:
            meta["descriptors"].append(timeframe_label)

        return df, meta

    def _detail_suffix(self, meta: Dict[str, Any]) -> str:
        desc = [d for d in meta.get("descriptors", []) if d]
        return f" ({', '.join(desc)})" if desc else ""

    def _fallback_timeframe(self, df: pd.DataFrame, base_df: pd.DataFrame, meta: Dict[str, Any]) -> Tuple[pd.DataFrame, str]:
        timeframe = meta.get("timeframe")
        if not timeframe or not df.empty:
            return df, ""
        if base_df.empty or "DateTime" not in base_df.columns:
            return df, ""
        dates = pd.to_datetime(base_df["DateTime"], errors="coerce").dropna()
        if dates.empty:
            return df, ""
        latest = dates.max()
        start, end = month_bounds(latest.year, latest.month)
        fallback_df = base_df[(base_df["DateTime"] >= start) & (base_df["DateTime"] < end)]
        if fallback_df.empty:
            return df, ""
        original_label = meta.get("timeframe_label")
        month_label = f"{latest.strftime('%B %Y')} (latest available)"
        descriptors = meta.get("descriptors", [])
        if original_label in descriptors:
            descriptors.remove(original_label)
        descriptors.append(month_label)
        meta["descriptors"] = descriptors
        meta["timeframe_label"] = month_label
        note = f" No data for {original_label}; showing {month_label} instead." if original_label else f" Showing {month_label}."
        return fallback_df, note

    def handle_count_to_place(self, question: str) -> Optional[QueryResponse]:
        match = re.search(
            r"how many (?:groups|rides|trips)[^?]*?(?:to|went to|headed to|drop(?:ped)? off at|at)\s+([A-Za-z0-9 '&\.-]+)",
            question,
            re.I,
        )
        if not match:
            return None
        place = match.group(1).strip()
        df, meta = self._apply_filters(question)
        if df.empty:
            return None

        dest_field = "DropoffAddress"
        simple_field = "DropoffSimple"
        label = "drop-off"
        lowered = question.lower()
        if "pickup" in lowered or "pick up" in lowered:
            dest_field = "PickupAddress"
            simple_field = "PickupSimple"
            label = "pickup"

        base_df = meta.get("pre_timeframe_df", df)
        base_mask = base_df[dest_field].astype(str).str.contains(place, case=False, na=False)
        if simple_field in base_df.columns:
            base_mask |= base_df[simple_field].astype(str).str.contains(place, case=False, na=False)
        if place.lower() == "downtown":
            base_filtered = filter_destination_keyword(base_df, "downtown", dest_field)
        else:
            base_filtered = base_df[base_mask]
        if base_filtered.empty:
            return None

        filtered = base_filtered
        if meta.get("timeframe"):
            filtered, note = self._fallback_timeframe(df[base_mask] if place.lower() != "downtown" else filter_destination_keyword(df, "downtown", dest_field), base_filtered, meta)
        else:
            note = ""
        if filtered.empty:
            return None

        count = int(len(filtered))
        text = f"{count:,} trips with {label} at {place}{self._detail_suffix(meta)}.{note}"
        text = polish_text(question, text, self.scraped)
        sample = format_sample(filtered, ["TripID", "DateTime", dest_field, "PickupAddress", "PartySize", "DominantAgeBucket"])
        return QueryResponse(text=text, kind="count", sample=sample)

    def handle_top_destinations(self, question: str) -> Optional[QueryResponse]:
        lowered = question.lower()
        if "top" not in lowered or not any(k in lowered for k in ["drop", "destination", "pickup", "spot"]):
            return None
        df, meta = self._apply_filters(question)
        if df.empty:
            return None

        dest_field = "DropoffSimple"
        label = "Drop-off"
        if "pickup" in lowered or "pick up" in lowered:
            dest_field = "PickupSimple"
            label = "Pickup"

        df_base = df.copy()
        if dest_field not in df_base.columns:
            return None

        counts = df_base[dest_field].fillna("(unknown)").astype(str).value_counts().head(10).reset_index()
        if counts.empty:
            base_df = meta.get("pre_timeframe_df", df_base)
            counts = base_df[dest_field].fillna("(unknown)").astype(str).value_counts().head(10).reset_index()
            note = " Showing latest available data." if not counts.empty else ""
        else:
            note = ""
        if counts.empty:
            return None
        counts.columns = [label, "Trips"]

        text = f"Top {label.lower()} destinations{self._detail_suffix(meta)}.{note}"
        text = polish_text(question, text, self.scraped)
        return QueryResponse(
            text=text,
            kind="table",
            table=counts,
            chart_type="bar",
            chart_x=label,
            chart_y="Trips",
            chart_sort=None,
            chart_title=text,
        )

    def handle_large_group_hours(self, question: str) -> Optional[QueryResponse]:
        lowered = question.lower()
        if not any(k in lowered for k in ["when do", "what time", "typical hours", "peak hours", "usually"]):
            return None
        df, meta = self._apply_filters(question)
        threshold = meta.get("threshold") or 6
        if meta.get("threshold") is None:
            df = filter_by_group_threshold(df, threshold)
            meta["descriptors"].append(f"{threshold}+ riders")
        if "downtown" in lowered:
            df = filter_destination_keyword(df, "downtown", "DropoffAddress")
        if df.empty or "Hour" not in df.columns:
            base_df = meta.get("pre_timeframe_df", df)
            df = filter_destination_keyword(base_df, "downtown", "DropoffAddress") if "downtown" in lowered else base_df
            if meta.get("timeframe"):
                df, note = self._fallback_timeframe(pd.DataFrame(), df, meta)
            else:
                note = ""
        else:
            note = ""
        if df.empty or "Hour" not in df.columns:
            return None

        counts = df["Hour"].dropna().astype(int).value_counts().sort_index()
        if counts.empty:
            return None
        hist = pd.DataFrame({"Hour": counts.index, "Trips": counts.values})
        top_hours = counts.sort_values(ascending=False).head(3).index.tolist()
        hour_label = ", ".join(f"{h:02d}:00" for h in top_hours)
        text = f"Peak hours for groups of {threshold}+ riders{self._detail_suffix(meta)}: {hour_label}.{note}"
        text = polish_text(question, text, self.scraped)
        return QueryResponse(
            text=text,
            kind="table",
            table=hist,
            chart_type="bar",
            chart_x="Hour",
            chart_y="Trips",
            chart_title="Trips by hour",
        )

    def handle_trips_by_day(self, question: str) -> Optional[QueryResponse]:
        lowered = question.lower()
        if not any(k in lowered for k in ["busiest day", "day of week", "rides by day", "which day"]):
            return None
        df, meta = self._apply_filters(question)
        if df.empty or "DOW" not in df.columns:
            base_df = meta.get("pre_timeframe_df", df)
            df, note = self._fallback_timeframe(pd.DataFrame(), base_df, meta)
            if df.empty or "DOW" not in df.columns:
                return None
        else:
            note = ""
        counts = df["DOW"].value_counts().reindex(DAY_NAMES).fillna(0).astype(int).reset_index()
        counts.columns = ["Day", "Trips"]
        text = f"Trips by day of week{self._detail_suffix(meta)}.{note}"
        text = polish_text(question, text, self.scraped)
        return QueryResponse(
            text=text,
            kind="table",
            table=counts,
            chart_type="bar",
            chart_x="Day",
            chart_y="Trips",
            chart_sort=DAY_NAMES,
            chart_title="Trips by day of week",
        )

    def handle_trips_by_hour(self, question: str) -> Optional[QueryResponse]:
        lowered = question.lower()
        if not any(k in lowered for k in ["by hour", "hourly", "hour of day", "time of day"]):
            return None
        df, meta = self._apply_filters(question)
        if df.empty or "Hour" not in df.columns:
            base_df = meta.get("pre_timeframe_df", df)
            df, note = self._fallback_timeframe(pd.DataFrame(), base_df, meta)
            if df.empty or "Hour" not in df.columns:
                return None
        else:
            note = ""
        counts = df["Hour"].dropna().astype(int).value_counts().sort_index()
        if counts.empty:
            return None
        table = pd.DataFrame({"Hour": counts.index, "Trips": counts.values})
        text = f"Trips by hour{self._detail_suffix(meta)}.{note}"
        text = polish_text(question, text, self.scraped)
        return QueryResponse(
            text=text,
            kind="table",
            table=table,
            chart_type="bar",
            chart_x="Hour",
            chart_y="Trips",
            chart_title="Trips by hour",
        )

    def handle_total_trips(self, question: str) -> Optional[QueryResponse]:
        lowered = question.lower()
        if "how many" not in lowered or not any(token in lowered for token in ["trip", "ride", "group"]):
            return None
        df, meta = self._apply_filters(question)
        if df.empty:
            base_df = meta.get("pre_timeframe_df", df)
            df, note = self._fallback_timeframe(df, base_df, meta)
            if df.empty:
                return None
        else:
            note = ""
        count = int(len(df))
        text = f"{count:,} trips match this question{self._detail_suffix(meta)}.{note}"
        text = polish_text(question, text, self.scraped)
        sample = format_sample(df, ["TripID", "DateTime", "PickupAddress", "DropoffAddress", "PartySize", "DominantAgeBucket"])
        return QueryResponse(text=text, kind="count", sample=sample)

    def ai_sql(self, question: str) -> Optional[QueryResponse]:
        if not (OPENAI_READY and DUCK_READY):
            return None
        sql = build_llm_sql(question, self.frames, self.now)
        if not sql:
            return None
        try:
            df = run_duckdb_query(self.frames, sql)
        except Exception as e:
            return QueryResponse(text=f"SQL error: {e}", kind="message", debug_sql=sql)
        if df.shape == (1, 1) and pd.api.types.is_numeric_dtype(df.iloc[:, 0]):
            val = df.iloc[0, 0]
            if isinstance(val, (int, np.integer)) or (isinstance(val, float) and float(val).is_integer()):
                val = int(val)
            text = f"Answer: {val:,}" if isinstance(val, (int, np.integer)) else f"Answer: {val}"
            text = polish_text(question, text, self.scraped)
            return QueryResponse(text=text, kind="count", debug_sql=sql)
        chart_type, cx, cy, csort = infer_chart(df)
        text = polish_text(question, "Here’s what I found from the dataset.", self.scraped)
        return QueryResponse(text=text, kind="table", table=df, chart_type=chart_type, chart_x=cx, chart_y=cy, chart_sort=csort, debug_sql=sql)

    def fallback_help(self) -> QueryResponse:
        txt = """Try questions like:
- How many groups went to Moody Center last month?
- Top drop-off spots for ages 18-24 on Saturday nights
- When do large groups (6+) typically ride downtown?"""
        return QueryResponse(text=txt, kind="message")

    def answer(self, question: str) -> QueryResponse:
        cleaned = question.strip()
        if not cleaned:
            return self.fallback_help()
        ai_resp = self.ai_sql(cleaned)
        if ai_resp and ai_resp.kind != "message":
            return ai_resp
        for handler in self.handlers:
            resp = handler(cleaned)
            if resp:
                return resp
        return ai_resp if ai_resp else self.fallback_help()
# -------------------- UI --------------------

st.markdown('<div class="big-title">Fetii Austin Data Copilot</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Ask natural-language questions about group rides in Austin.</div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("Setup")
    uploaded_file = st.file_uploader("Upload FetiiAI_Data_Austin.xlsx", type=["xlsx"])
    st.caption("Skip the upload if the dataset sits next to this app.")
    enable_scrape = st.toggle("Add fetii.com context to answers", value=False)
    if enable_scrape and not SCRAPE_READY:
        st.info("Install requests and beautifulsoup4 to enable this option.")
    st.markdown("---")
    st.header("Status")
    st.write(f"**AI mode**: {'ON' if (OPENAI_READY and DUCK_READY) else 'OFF'}")
    if not OPENAI_READY:
        st.caption("Set OPENAI_API_KEY to enable AI chat.")
    if not DUCK_READY:
        st.caption("Install duckdb to enable AI chat: pip install duckdb")

# Load data
try:
    sheets = load_excel(uploaded_file.read() if uploaded_file else None)
    frames = build_master_frames(sheets)
    trips_df = frames["trips"]
    if trips_df.empty:
        st.error("Trips sheet not found or empty.")
        st.stop()
except Exception as exc:
    st.error(f"Failed to load Excel: {exc}")
    st.stop()

# optional scrape
scraped_text = ""
if enable_scrape and SCRAPE_READY:
    try:
        urls = [
            "https://www.fetii.com/",
            "https://www.fetii.com/how-it-works",
            "https://www.fetii.com/faq",
            "https://www.fetii.com/about",
        ]
        headers = {"User-Agent": "FetiiHackathonBot/1.0"}
        blobs: List[str] = []
        for u in urls:
            r = requests.get(u, headers=headers, timeout=10)
            if r.ok:
                soup = BeautifulSoup(r.text, "html.parser")
                for tag in soup(["script","style","noscript"]): tag.extract()
                text = " ".join(soup.get_text(separator=" ").split())
                blobs.append(f"[Source: {u}]\n{text}\n")
        scraped_text = "\n\n".join(blobs)
    except Exception:
        scraped_text = ""

engine = QueryEngine(frames, scraped_text)

# KPIs
date_series = trips_df.get("DateTime")
if date_series is not None:
    date_values = pd.to_datetime(date_series, errors="coerce").dropna()
    date_min = date_values.min() if not date_values.empty else None
    date_max = date_values.max() if not date_values.empty else None
else:
    date_values = pd.Series([], dtype="datetime64[ns]")
    date_min = date_max = None
unique_bookers = trips_df["UserID"].nunique() if "UserID" in trips_df.columns else None
avg_party = trips_df["PartySize"].dropna().mean() if "PartySize" in trips_df.columns else np.nan

mc = st.columns(4)
with mc[0]: st.metric("Total trips", f"{len(trips_df):,}")
with mc[1]: st.metric("Date coverage", f"{date_min:%Y-%m-%d} → {date_max:%Y-%m-%d}" if (date_min and date_max) else "Unknown")
with mc[2]: st.metric("Unique bookers", f"{unique_bookers:,}" if unique_bookers is not None else "—")
with mc[3]: st.metric("Avg party size", f"{avg_party:.2f}" if not np.isnan(avg_party) else "—")

with st.expander("Preview dataset", expanded=False):
    st.dataframe(trips_df.head(25), use_container_width=True)

st.markdown("---")
st.markdown("Ask **anything** about the dataset. Examples: *How many groups went to Moody Center last month?* • *Top drop-off spots for 18–24 on Saturday nights* • *When do large groups (6+) ride downtown?*")

# Chat form
with st.form("chat_form"):
    user_q = st.text_input("Question", placeholder="How many groups went to Moody Center last month?")
    submitted = st.form_submit_button("Ask")

def render_response(res: QueryResponse):
    st.subheader("Answer")
    st.write(res.text)
    if res.table is not None and not res.table.empty:
        st.dataframe(res.table, use_container_width=True)
        st.download_button("Download table (CSV)", _df_to_csv_bytes(res.table), "result.csv", "text/csv")
        if res.chart_type and res.chart_x and res.chart_y:
            chart = alt.Chart(res.table).mark_bar().encode(
                x=alt.X(f"{res.chart_x}:N", sort=res.chart_sort or "-y"),
                y=f"{res.chart_y}:Q",
                tooltip=list(res.table.columns),
            )
            if res.chart_title:
                chart = chart.properties(title=res.chart_title)
            st.altair_chart(chart, use_container_width=True)
    if res.sample is not None and not res.sample.empty:
        st.caption("Sample matching trips")
        st.dataframe(res.sample, use_container_width=True)
    if res.debug_sql:
        with st.expander("Debug: SQL used"):
            st.code(res.debug_sql, language="sql")

if submitted:
    if user_q.strip():
        with st.spinner("Thinking..."):
            result = engine.answer(user_q.strip())
        render_response(result)
    else:
        st.info("Enter a question to run the chatbot.")

st.markdown("---")
st.markdown('<div class="footer-note">AI mode uses GPT to generate DuckDB SQL over your dataset. '
            'Set OPENAI_API_KEY and install duckdb to enable it. Fallback examples still work without AI.</div>',
            unsafe_allow_html=True)
