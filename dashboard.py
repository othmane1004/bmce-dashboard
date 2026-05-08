import re
import io
import json
import requests
from pathlib import Path
from datetime import date

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import altair as alt

# ── Groq API (LLaMA 3.3 70B) ──
GROQ_API_KEY = "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"

def call_gemini(prompt: str) -> str:
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 400,
    }
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=60)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    except Exception as exc:
        return f"[Erreur Groq : {exc}]"

st.set_page_config(page_title="BMCE Capital Gestion — S&R", layout="wide", initial_sidebar_state="expanded")

# ──────────────────────────────────────────────
# DESIGN TOKENS — BMCE Capital Gestion
# ──────────────────────────────────────────────
PRIMARY    = "#1C1C1C"
SECONDARY  = "#4A4A4A"
MUTED      = "#9A9A9A"
BG         = "#FFFFFF"
SURFACE    = "#F5F5F5"
SIDEBAR_BG = "#FAFAFA"
BORDER     = "#E0E0E0"
ACCENT     = "#C0001A"      # rouge BMCE
ACCENT2    = "#960014"
GREEN      = "#1A7A4A"
RED        = "#C0001A"
CHART_GRID = "#F0F0F0"
PALETTE    = ["#C0001A", "#1C52C8", "#F59E0B", "#10B981", "#8B5CF6",
              "#EC4899", "#06B6D4", "#84CC16", "#F97316", "#6366F1",
              "#14B8A6", "#EF4444", "#A855F7", "#22C55E", "#EAB308"]

_T = dict(P=PRIMARY, S=SECONDARY, M=MUTED, B=BG, SF=SURFACE,
          BD=BORDER, AC=ACCENT, AC2=ACCENT2, GR=GREEN, RD=RED)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

/* ── Masquer entièrement la sidebar ── */
section[data-testid="stSidebar"],
section[data-testid="stSidebar"] + div button[data-testid="stSidebarCollapsedControl"],
button[data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"] {
  display: none !important;
  width: 0 !important;
}
.stApp > div:first-child { margin-left: 0 !important; }
.block-container { max-width: 1280px !important; padding-left: 2rem !important; }

/* ── Override variables CSS Streamlit (file uploader les utilise en interne) ── */
:root {
  --primary-color: #C0001A !important;
  --background-color: #FFFFFF !important;
  --secondary-background-color: #FAFAFA !important;
  --text-color: #1C1C1C !important;
}

html,body,.stApp,[class*="css"]{font-family:'Montserrat',sans-serif!important;}

/* fond blanc partout */
html,body,.stApp,.main,.block-container,
div[data-testid="stAppViewContainer"],div[data-testid="stAppViewContainer"]>.main,
div[data-testid="stMain"],header[data-testid="stHeader"]{background:#fff!important}
.block-container{padding-top:0!important;max-width:1180px!important}

/* sidebar base */
section[data-testid="stSidebar"]{background:#FAFAFA!important;border-right:1px solid #E8E8E8!important}
section[data-testid="stSidebar"] *{color:%(P)s!important;-webkit-text-fill-color:%(P)s!important}

/* ── FILE UPLOADER : casser le fond noir à tous les niveaux ── */
[data-testid="stFileUploader"],
[data-testid="stFileUploader"] *,
[data-testid="stFileUploadDropzone"],
[data-testid="stFileUploadDropzone"] > div,
[data-testid="stFileUploadDropzone"] > div > div,
[data-testid="stFileUploadDropzone"] > div > div > div {
  background:#FAFAFA!important;
  background-color:#FAFAFA!important;
  box-shadow:none!important;
}
[data-testid="stFileUploadDropzone"]{
  border:1.5px dashed %(BD)s!important;
  border-radius:6px!important;
}
[data-testid="stFileUploadDropzone"] span,
[data-testid="stFileUploadDropzone"] p,
[data-testid="stFileUploadDropzone"] small,
[data-testid="stFileUploadDropzone"] div{
  color:%(M)s!important;-webkit-text-fill-color:%(M)s!important;
  background:transparent!important;background-color:transparent!important;
}
[data-testid="stFileUploadDropzone"] button,
[data-testid="stFileUploadDropzone"] button *{
  background:#fff!important;background-color:#fff!important;
  border:1.5px solid %(AC)s!important;border-radius:4px!important;
  color:%(AC)s!important;-webkit-text-fill-color:%(AC)s!important;
  font-weight:600!important;font-size:0.78rem!important;
  padding:6px 16px!important;box-shadow:none!important;
}
[data-testid="stFileUploadDropzone"] button:hover,
[data-testid="stFileUploadDropzone"] button:hover *{
  background:%(AC)s!important;background-color:%(AC)s!important;
  color:#fff!important;-webkit-text-fill-color:#fff!important;
}

/* tags/badges multiselect — fond rouge BMCE */
section[data-testid="stSidebar"] [data-baseweb="tag"],
[data-baseweb="tag"]{
  background:%(AC)s!important;border:none!important;
  border-radius:3px!important;padding:2px 8px!important;
}
section[data-testid="stSidebar"] [data-baseweb="tag"] span,
[data-baseweb="tag"] span{
  color:#fff!important;-webkit-text-fill-color:#fff!important;
  font-size:0.72rem!important;font-weight:600!important;
}
[data-baseweb="tag"] [role="presentation"] svg{fill:#fff!important}

/* ── Éliminer TOUS les bleus Streamlit → rouge BMCE ── */
/* checkbox */
[data-baseweb="checkbox"] [data-checked="true"] > div,
[data-baseweb="checkbox"] input:checked ~ div > div{
  background-color:%(AC)s!important;border-color:%(AC)s!important;
}
/* slider track + thumb */
[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"]{
  background:%(AC)s!important;border-color:%(AC)s!important;
}
[data-testid="stSlider"] [data-baseweb="slider"] div:nth-child(5){
  background:%(AC)s!important;
}
/* select focus ring */
div[data-baseweb="select"] div:focus-within{border-color:%(AC)s!important;box-shadow:0 0 0 2px rgba(192,0,26,.15)!important}
/* generic blue links */
a{color:%(AC)s!important;-webkit-text-fill-color:%(AC)s!important}
/* focus outline global */
*:focus{outline-color:%(AC)s!important}
/* progress/spinner */
[data-testid="stSpinner"] > div > div{border-top-color:%(AC)s!important}
/* radio */
[data-baseweb="radio"] [data-checked="true"] > div{background:%(AC)s!important;border-color:%(AC)s!important}

/* ── Boutons Vega-Embed (menu "..." des graphiques Altair) ── */
.vega-embed summary,
.vega-embed .vega-actions-wrapper,
.vega-embed details summary{
  background:#fff!important;
  background-color:#fff!important;
  border:1px solid %(BD)s!important;
  border-radius:4px!important;
  color:%(P)s!important;
  box-shadow:0 1px 4px rgba(0,0,0,.08)!important;
  opacity:1!important;
}
.vega-embed summary svg path,
.vega-embed summary svg{
  fill:%(P)s!important;stroke:%(P)s!important;
}
/* dropdown actions du menu vega */
.vega-embed .vega-actions,
.vega-embed details .vega-actions{
  background:#fff!important;
  border:1px solid %(BD)s!important;
  border-radius:6px!important;
  box-shadow:0 4px 16px rgba(0,0,0,.10)!important;
}
.vega-embed .vega-actions a,
.vega-embed .vega-actions button{
  background:#fff!important;
  color:%(P)s!important;
  -webkit-text-fill-color:%(P)s!important;
  font-family:'Montserrat',sans-serif!important;
  font-size:0.78rem!important;
}
.vega-embed .vega-actions a:hover,
.vega-embed .vega-actions button:hover{
  background:%(SF)s!important;
}
/* bouton fullscreen Streamlit */
button[title="View fullscreen"],
button[data-testid="StyledFullScreenButton"]{
  background:#fff!important;
  background-color:#fff!important;
  border:1px solid %(BD)s!important;
  border-radius:4px!important;
  box-shadow:0 1px 3px rgba(0,0,0,.08)!important;
}
button[title="View fullscreen"] svg,
button[data-testid="StyledFullScreenButton"] svg{
  fill:%(P)s!important;
}

/* text */
p,span,label,div,.stMarkdown *{color:%(P)s!important;-webkit-text-fill-color:%(P)s!important}

/* selects */
div[data-baseweb="select"]>div{border:1px solid %(BD)s!important;border-radius:4px!important;background:#fff!important}
div[data-baseweb="popover"],div[data-baseweb="menu"],ul[role="listbox"]{background:#fff!important;border:1px solid %(BD)s!important;border-radius:6px!important;box-shadow:0 4px 16px rgba(0,0,0,.10)!important}
div[data-baseweb="menu"] li,li[role="option"]{background:#fff!important;color:%(P)s!important;-webkit-text-fill-color:%(P)s!important}
li[role="option"]:hover{background:%(SF)s!important}

/* metrics */
div[data-testid="metric-container"]{background:#fff!important;border:1px solid %(BD)s!important;border-left:3px solid %(AC)s!important;border-radius:4px!important;padding:16px 18px!important;box-shadow:0 1px 4px rgba(0,0,0,.06)!important}
div[data-testid="metric-container"] [data-testid="stMetricLabel"] p{font-size:0.7rem!important;font-weight:600!important;text-transform:uppercase!important;letter-spacing:.08em!important;color:%(M)s!important;-webkit-text-fill-color:%(M)s!important}
div[data-testid="metric-container"] [data-testid="stMetricValue"] *{font-size:1.45rem!important;font-weight:700!important;color:%(P)s!important;-webkit-text-fill-color:%(P)s!important}
div[data-testid="metric-container"] *{opacity:1!important}
div[data-testid="metric-container"] [data-direction="up"] *{color:%(GR)s!important;-webkit-text-fill-color:%(GR)s!important}
div[data-testid="metric-container"] [data-direction="down"] *{color:%(RD)s!important;-webkit-text-fill-color:%(RD)s!important}

/* tabs */
.stTabs [data-baseweb="tab-list"]{border-bottom:2px solid %(BD)s!important;gap:0!important}
.stTabs [data-baseweb="tab"]{color:#999!important;-webkit-text-fill-color:#999!important;font-weight:600!important;font-size:.8rem!important;text-transform:uppercase!important;letter-spacing:.06em!important;border-bottom:2px solid transparent!important;padding:10px 20px!important;margin-bottom:-2px!important}
.stTabs [aria-selected="true"]{color:%(AC)s!important;-webkit-text-fill-color:%(AC)s!important;border-bottom:2px solid %(AC)s!important}

/* charts */
div[data-testid="stVegaLiteChart"]{background:#fff!important;border:1px solid %(BD)s!important;border-radius:6px!important;padding:16px!important}

/* tooltip */
#vg-tooltip-element,.vg-tooltip{background:%(P)s!important;color:#fff!important;border:none!important;border-radius:4px!important;padding:8px 12px!important;font-family:'Montserrat',sans-serif!important;font-size:.78rem!important;box-shadow:0 4px 16px rgba(0,0,0,.2)!important}
#vg-tooltip-element *,.vg-tooltip *{color:#fff!important;-webkit-text-fill-color:#fff!important;background:transparent!important}
#vg-tooltip-element td.key,.vg-tooltip td.key{color:rgba(255,255,255,.55)!important;-webkit-text-fill-color:rgba(255,255,255,.55)!important}
#vg-tooltip-element table,.vg-tooltip table{border-collapse:collapse!important}
#vg-tooltip-element td,.vg-tooltip td{padding:2px 6px!important;border:none!important}

/* expander */
details{background:#fff!important;border:1px solid %(BD)s!important;border-radius:6px!important}
details summary{color:%(P)s!important;font-weight:600!important}

/* info/warning boxes — override blue */
div[data-testid="stAlert"]{border-radius:4px!important;border-left:3px solid %(AC)s!important}

hr{border:none!important;border-top:1px solid %(BD)s!important;margin:1rem 0!important}

/* boutons download + boutons standard */
[data-testid="stDownloadButton"] button,
[data-testid="stButton"] button{
  background:#fff!important;background-color:#fff!important;
  border:1.5px solid %(BD)s!important;border-radius:4px!important;
  color:%(P)s!important;-webkit-text-fill-color:%(P)s!important;
  font-size:0.78rem!important;font-weight:600!important;
  box-shadow:none!important;padding:6px 14px!important;
}
[data-testid="stDownloadButton"] button:hover,
[data-testid="stButton"] button:hover{
  border-color:%(AC)s!important;color:%(AC)s!important;
  -webkit-text-fill-color:%(AC)s!important;
}
div[data-testid="stSlider"] *{color:%(P)s!important;-webkit-text-fill-color:%(P)s!important}
::-webkit-scrollbar{width:4px;height:4px}
::-webkit-scrollbar-thumb{background:%(BD)s;border-radius:99px}
</style>
""" % _T, unsafe_allow_html=True)

# ──────────────────────────────────────────────
# ALTAIR THEME
# ──────────────────────────────────────────────
def _bmce_theme():
    return {
        "config": {
            "background": BG,
            "view": {"stroke": "transparent"},
            "axis": {
                "domainColor": BORDER, "gridColor": CHART_GRID,
                "tickColor": "transparent", "labelColor": MUTED,
                "titleColor": SECONDARY, "labelFont": "Montserrat",
                "titleFont": "Montserrat", "labelFontSize": 11,
                "titleFontSize": 11, "titleFontWeight": 600,
                "gridDash": [3, 3],
            },
            "legend": {
                "labelColor": SECONDARY, "titleColor": PRIMARY,
                "labelFont": "Inter", "titleFont": "Inter",
                "labelFontSize": 11, "titleFontSize": 11,
                "titleFontWeight": 700,
            },
            "title": {
                "color": PRIMARY, "font": "Montserrat",
                "fontWeight": 700, "fontSize": 13,
                "anchor": "start", "offset": 8,
            },
            "line": {"strokeWidth": 2.5},
            "point": {"filled": True, "size": 50},
            "range": {"category": PALETTE},
        }
    }

alt.themes.register("bmce", _bmce_theme)
alt.themes.enable("bmce")

# ──────────────────────────────────────────────
# LOGO
# ──────────────────────────────────────────────
_LOGO_PATHS = [
    Path("/Users/mac/Desktop/bmce_capital_logo.jpeg"),
    Path("bmce_capital_logo.jpeg"),
    Path("bmce_capital_logo.jpg"),
    Path("bmce_capital_logo.png"),
]

def _find_logo():
    for p in _LOGO_PATHS:
        try:
            if p.exists():
                return p
        except Exception:
            pass
    return None

logo_path = _find_logo()

# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────
_RE_DATE = re.compile(r"(\d{1,2})\s*[\/\-\.]\s*(\d{1,2})")

def norm_ddmm(x):
    if x is None:
        return None
    # Cas datetime/date Excel : openpyxl retourne un objet avec .day et .month
    if hasattr(x, "day") and hasattr(x, "month"):
        dd, mm = x.day, x.month
        return f"{dd:02d}/{mm:02d}" if 1 <= dd <= 31 and 1 <= mm <= 12 else None
    m = _RE_DATE.search(str(x))
    if not m:
        return None
    dd, mm = int(m.group(1)), int(m.group(2))
    return f"{dd:02d}/{mm:02d}" if 1 <= dd <= 31 and 1 <= mm <= 12 else None

def mmdd_sort(ddmm):
    dd, mm = ddmm.split("/")
    return (int(mm), int(dd))

def ddmm_to_dt(ddmm, year=None):
    year = year or date.today().year
    dd, mm = ddmm.split("/")
    return pd.Timestamp(year=year, month=int(mm), day=int(dd))

def to_float(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.number)):
        return float(x)
    s = str(x).strip()
    if s.lower() in {"", "-", "—", "n/a", "na", "nan", "null", "none"}:
        return np.nan
    s = s.replace("\u00A0", " ").replace("\u202F", " ").replace(" ", "")
    pct = s.endswith("%")
    if pct:
        s = s[:-1]
    s = s.replace(",", ".")
    try:
        v = float(s)
    except Exception:
        return np.nan
    return v / 100.0 if pct else v

def fmt_money(x):
    return "—" if pd.isna(x) else f"{x:,.0f}".replace(",", " ")

def fmt_pct(x, nd=1):
    return "—" if pd.isna(x) else f"{x * 100:.{nd}f}%"

def _norm_sg_key(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())

def pick_bmce_sg(sg_values):
    """Best-effort detection of BMCE Capital Gestion label in SG list."""
    vals = [str(v).strip() for v in sg_values if str(v).strip()]
    if not vals:
        return None
    scored = []
    for sg in vals:
        k = _norm_sg_key(sg)
        score = 0
        if "bmce" in k:
            score += 5
        if "capital" in k:
            score += 2
        if "gestion" in k:
            score += 2
        if "bmcecapitalgestion" in k:
            score += 10
        scored.append((score, sg))
    scored.sort(reverse=True)
    return scored[0][1] if scored and scored[0][0] > 0 else None

def flow_series_on_market_days(flow_daily, sg, market_dates):
    s = (
        flow_daily[flow_daily["SG"] == sg]
        .groupby("Date_dt")["SR"]
        .sum(min_count=1)
        .reindex(market_dates, fill_value=0.0)
        .astype(float)
    )
    return pd.DataFrame({"Date_dt": s.index, "SR": s.values})

def compute_flow_kpis(series_df, market_abs_series=None):
    out = {
        "ytd_net_flow": np.nan,
        "mtd_net_flow": np.nan,
        "last_5d_net_flow": np.nan,
        "avg_daily_flow": np.nan,
        "positive_day_ratio": np.nan,
        "flow_volatility": np.nan,
        "max_drawdown": np.nan,
        "best_day": np.nan,
        "worst_day": np.nan,
        "momentum_5d_vs_prev5d": np.nan,
        "abs_market_share": np.nan,
    }
    if series_df.empty:
        return out

    s = series_df["SR"].astype(float)
    d = pd.to_datetime(series_df["Date_dt"])

    out["ytd_net_flow"] = s.sum()
    last_dt = d.max()
    mtd_mask = (d.dt.year == last_dt.year) & (d.dt.month == last_dt.month)
    out["mtd_net_flow"] = s[mtd_mask].sum()
    out["last_5d_net_flow"] = s.tail(5).sum()
    out["avg_daily_flow"] = s.mean()
    out["positive_day_ratio"] = (s > 0).mean() if len(s) else np.nan
    out["flow_volatility"] = s.std(ddof=0) if len(s) else np.nan
    out["best_day"] = s.max() if len(s) else np.nan
    out["worst_day"] = s.min() if len(s) else np.nan

    if len(s) >= 10:
        out["momentum_5d_vs_prev5d"] = s.tail(5).sum() - s.tail(10).head(5).sum()

    cum = s.cumsum()
    drawdown = cum - cum.cummax()
    out["max_drawdown"] = abs(drawdown.min()) if len(drawdown) else np.nan

    if market_abs_series is not None:
        denom = market_abs_series.sum()
        if abs(denom) > 1e-12:
            out["abs_market_share"] = s.abs().sum() / denom
    return out

def is_total(s):
    t = str(s).strip().lower()
    return ("total" in t and ("gén" in t or "gen" in t)) or t == "total"

def unique_cols(cols):
    seen = {}
    out = []
    for c in cols:
        c = "" if c is None else str(c).strip()
        if c not in seen:
            seen[c] = 0
            out.append(c)
        else:
            seen[c] += 1
            out.append(f"{c}__{seen[c]}")
    return out

def find_col(cols, patterns):
    for c in list(cols):
        lc = str(c).strip().lower()
        for p in patterns:
            if p in lc:
                return c
    return None

def apply_weekly_bridge_fix(tidy, eps=1e-9):
    """Fix known bridge-day anomaly where weekly SG appears on Mon/Tue.

    If a SG is active on 02/02 and 03/02, exactly zero on 04/02, and active again
    on 06/02, we neutralize 02/02 and 03/02 for that SG.
    """
    if tidy.empty:
        return tidy

    needed = {"02/02", "03/02", "04/02", "06/02"}
    if not needed.issubset(set(tidy["Date"].dropna().unique().tolist())):
        return tidy

    sg_date = tidy.groupby(["SG", "Date"], as_index=False)["SR"].sum(min_count=1)
    piv = sg_date.pivot(index="SG", columns="Date", values="SR")
    for d in needed:
        if d not in piv.columns:
            return tidy
    piv = piv.fillna(0.0)

    weekly_sg = piv[
        (piv["02/02"].abs() > eps)
        & (piv["03/02"].abs() > eps)
        & (piv["04/02"].abs() <= eps)
        & (piv["06/02"].abs() > eps)
    ].index.tolist()

    if not weekly_sg:
        return tidy

    out = tidy.copy()
    mask = out["SG"].isin(weekly_sg) & out["Date"].isin(["02/02", "03/02"])
    out.loc[mask, "SR"] = 0.0
    return out

# ──────────────────────────────────────────────
# TABLEAU DE PERFORMANCES PDF — PARSER & CACHE
# ──────────────────────────────────────────────
_PERF_CACHE_PATH    = Path(__file__).parent / ".bmce_cache" / "perf_history.json"
_ASFIM_SR_CACHE_PATH = Path(__file__).parent / ".bmce_cache" / "asfim_sr_cache.json"

_MOIS_FR = {
    "janvier": 1, "février": 2, "fevrier": 2, "mars": 3, "avril": 4,
    "mai": 5, "juin": 6, "juillet": 7, "août": 8, "aout": 8,
    "septembre": 9, "octobre": 10, "novembre": 11, "décembre": 12, "decembre": 12,
}

_CLS_NORM_PERF = {
    "MONÉTAIRE": "MONETAIRE", "MONETAIRE": "MONETAIRE",
    "DIVERSIFIÉ": "DIVERSIFIE", "DIVERSIFIE": "DIVERSIFIE", "DIVERSIFIES": "DIVERSIFIE",
    "CONTRACTUEL": "CONTRACTUEL", "ACTIONS": "ACTIONS", "OMLT": "OMLT", "OCT": "OCT",
}

def extract_date_from_perf_filename(filename_or_sheetname):
    """Extract DD/MM from sheet name '22-04-2026' or filename 'Tableau de performance du 22 avril 2026.xlsx'."""
    s = str(filename_or_sheetname).strip()
    # Format feuille Excel : DD-MM-YYYY
    m = re.search(r"(\d{2})-(\d{2})-(\d{4})", s)
    if m:
        return f"{m.group(1)}/{m.group(2)}"
    # Format nom de fichier : "22 avril 2026"
    m = re.search(r"(\d{1,2})\s+(\w+)\s+(\d{4})", s.lower())
    if m:
        month = _MOIS_FR.get(m.group(2))
        if month:
            return f"{int(m.group(1)):02d}/{month:02d}"
    return None

def parse_tableau_performances_excel(xlsx_bytes):
    """Parse ASFIM Tableau de performances Excel file.
    Structure : 1 feuille nommée 'DD-MM-YYYY', header en ligne 1,
    colonnes : OPCVM, Société de Gestion, Classification, AN, VL, YTD, 1 jour, 1 semaine, ...
    Returns (DataFrame [OPCVM, SG, Classification, AN, VL, return_1j, return_1s, YTD_perf], date_str).
    """
    try:
        xls = pd.ExcelFile(io.BytesIO(xlsx_bytes), engine="openpyxl")
        sheet_name = xls.sheet_names[0]
        df = pd.read_excel(io.BytesIO(xlsx_bytes), sheet_name=sheet_name, header=1, engine="openpyxl")
    except Exception:
        return pd.DataFrame(), None

    # Date depuis le nom de la feuille (fallback: dernière date détectée dans les colonnes)
    date_str = extract_date_from_perf_filename(sheet_name)

    if df.empty:
        return pd.DataFrame(), date_str

    # Mapping des colonnes
    rename = {
        "OPCVM": "OPCVM",
        "Société de Gestion": "SG",
        "Classification": "Classification",
        "AN": "AN",
        "VL": "VL",
        "YTD": "YTD_perf",
        "1 jour": "return_1j",
        "1 semaine": "return_1s",
        "1 mois": "return_1m",
    }
    df = df.rename(columns={c: rename[c] for c in df.columns if c in rename})

    required = ["OPCVM", "SG", "Classification", "AN", "VL", "return_1j"]
    if not all(c in df.columns for c in required):
        return pd.DataFrame(), date_str

    keep = [c for c in ["OPCVM", "SG", "Classification", "AN", "VL",
                         "return_1j", "return_1s", "YTD_perf", "return_1m"] if c in df.columns]
    df = df[keep].copy()

    # Nettoyage
    df["OPCVM"] = df["OPCVM"].astype(str).str.strip()
    df["SG"]    = df["SG"].astype(str).str.strip()
    df["Classification"] = (
        df["Classification"].astype(str).str.strip()
        .map(lambda x: _CLS_NORM_PERF.get(x, x))
    )
    for col in ["AN", "VL", "return_1j", "return_1s", "YTD_perf", "return_1m"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[df["OPCVM"].notna() & df["OPCVM"].ne("nan") & df["OPCVM"].ne("") & df["AN"].notna()]

    if not date_str:
        _date_candidates = []
        for c in df.columns:
            d = norm_ddmm(c)
            if d:
                _date_candidates.append(d)
        if _date_candidates:
            date_str = sorted(set(_date_candidates), key=mmdd_sort)[-1]

    return df.reset_index(drop=True), date_str


def save_perf_snapshot(perf_df, date_str, overwrite=True):
    """Persist AN+VL+return_1j per OPCVM for a given date into JSON cache.
    If overwrite=False, skip funds that already have a snapshot for this date.
    """
    if perf_df.empty or not date_str:
        return
    history = {}
    if _PERF_CACHE_PATH.exists():
        try:
            history = json.loads(_PERF_CACHE_PATH.read_text(encoding="utf-8"))
        except Exception:
            history = {}

    for _, row in perf_df.iterrows():
        opcvm = row["OPCVM"]
        if opcvm not in history:
            history[opcvm] = {"meta": {}, "snapshots": {}}
        history[opcvm]["meta"] = {
            "SG": str(row.get("SG", "")),
            "Classification": str(row.get("Classification", "")),
        }
        if not overwrite and date_str in history[opcvm]["snapshots"]:
            continue
        an_v   = float(row["AN"])       if pd.notna(row.get("AN"))       else None
        vl_v   = float(row["VL"])       if pd.notna(row.get("VL"))       else None
        r1j_v  = float(row["return_1j"]) if pd.notna(row.get("return_1j")) else None
        history[opcvm]["snapshots"][date_str] = {"AN": an_v, "VL": vl_v, "return_1j": r1j_v}

    _PERF_CACHE_PATH.write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def load_perf_history():
    """Load persisted AN/VL history. Returns dict or {}."""
    if not _PERF_CACHE_PATH.exists():
        return {}
    try:
        return json.loads(_PERF_CACHE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def compute_sr_from_perf_history(history):
    """Compute weekly S&R from AN history: S&R_t = AN_t − AN_{t−1} × (VL_t / VL_{t−1}).
    Returns DataFrame equivalent to asfim_sr: [OPCVM, SG, Classification, Date, SR, AN, YTD].
    """
    if not history:
        return pd.DataFrame(columns=["OPCVM", "SG", "Classification", "Date", "SR", "AN", "YTD"])

    rows = []
    for opcvm, data in history.items():
        meta = data.get("meta", {})
        sg  = meta.get("SG", "")
        cls = meta.get("Classification", "")
        snaps = data.get("snapshots", {})
        sorted_dates = sorted(snaps.keys(), key=mmdd_sort)
        cumul_ytd = 0.0

        for i, d in enumerate(sorted_dates):
            snap = snaps[d]
            an_curr = snap.get("AN")
            vl_curr = snap.get("VL")

            if i == 0 or an_curr is None:
                sr = np.nan
            else:
                d_prev  = sorted_dates[i - 1]
                an_prev = snaps[d_prev].get("AN")
                vl_prev = snaps[d_prev].get("VL")
                r_1j    = snap.get("return_1j")
                r_1s    = snap.get("return_1s")

                # Calculer l'écart en jours entre les deux snapshots
                _year = date.today().year
                try:
                    _dd, _mm = d.split("/")
                    _dd_p, _mm_p = d_prev.split("/")
                    _dt_curr = date(_year, int(_mm), int(_dd))
                    _dt_prev = date(_year, int(_mm_p), int(_dd_p))
                    _gap = (_dt_curr - _dt_prev).days
                except Exception:
                    _gap = 1

                if an_prev is None:
                    sr = np.nan
                elif vl_prev and vl_curr and vl_prev > 0:
                    # Cas idéal : deux VL disponibles → ratio exact
                    sr = an_curr - an_prev * (vl_curr / vl_prev)
                elif _gap >= 5 and r_1s is not None:
                    # Écart hebdomadaire (5+ jours) : utiliser return_1s
                    # return_1s ≈ VL(D)/VL(D-7) → bien adapté pour une semaine
                    sr = an_curr - an_prev * (1 + r_1s)
                elif r_1j is not None:
                    # Écart journalier : utiliser return_1j
                    sr = an_curr - an_prev * (1 + r_1j)
                else:
                    sr = an_curr - an_prev  # fallback brut

            if pd.notna(sr):
                cumul_ytd += sr

            rows.append({
                "OPCVM": opcvm, "SG": sg, "Classification": cls, "Date": d,
                "SR": sr, "AN": an_curr if an_curr is not None else np.nan,
                "YTD": cumul_ytd,
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    ytd_map = df.groupby("OPCVM")["YTD"].last()
    df["YTD"] = df["OPCVM"].map(ytd_map)
    return df


def save_asfim_sr_cache(df):
    """Persist the full asfim_sr DataFrame to JSON cache."""
    if df.empty:
        return
    try:
        _ASFIM_SR_CACHE_PATH.write_text(
            df.to_json(orient="records", force_ascii=False, default_handler=str),
            encoding="utf-8",
        )
    except Exception:
        pass


def load_asfim_sr_cache():
    """Load the persisted asfim_sr DataFrame from JSON cache."""
    if not _ASFIM_SR_CACHE_PATH.exists():
        return pd.DataFrame()
    try:
        df = pd.read_json(_ASFIM_SR_CACHE_PATH, orient="records")
        for col in ("SR", "AN", "YTD"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df
    except Exception:
        return pd.DataFrame()


def extract_an_from_asfim(asfim_sr_df):
    """Extract the last available AN per fund from the loaded asfim_sr DataFrame.
    Used to seed the perf cache so the first PDF upload can compute S&R immediately.
    Returns DataFrame [OPCVM, SG, Classification, AN, VL, return_1j, Date].
    """
    if asfim_sr_df.empty or "AN" not in asfim_sr_df.columns:
        return pd.DataFrame()
    df = asfim_sr_df[asfim_sr_df["AN"].notna()].copy()
    if df.empty:
        return pd.DataFrame()
    df["_sort"] = df["Date"].apply(mmdd_sort)
    df = df.sort_values("_sort")
    # Last AN per fund
    last = df.groupby("OPCVM", as_index=False).last()
    seed = last[["OPCVM", "SG", "Classification", "Date", "AN"]].copy()
    seed["VL"] = np.nan
    seed["return_1j"] = np.nan
    return seed


def build_tidy_from_asfim_sr(asfim_df):
    """Build tidy-equivalent DataFrame from per-fund asfim_sr."""
    if asfim_df.empty:
        return pd.DataFrame(columns=["Bloc", "Fonds", "SG", "Date", "SR", "YTD_row"])
    df = asfim_df.copy()
    df = df[df["SR"].notna()]
    agg = (
        df.groupby(["SG", "Classification", "Date"], as_index=False)
        .agg(SR=("SR", "sum"), YTD_row=("YTD", "sum"))
        .rename(columns={"Classification": "Bloc"})
    )
    agg["Fonds"] = None
    return agg


def read_tcd_daily_totals(xlsx_bytes):
    """Read daily totals directly from pivot-table sheet (Grand Total row)."""
    try:
        xls = pd.ExcelFile(io.BytesIO(xlsx_bytes), engine="openpyxl")
    except Exception:
        return {}

    tcd_name = None
    for sh in xls.sheet_names:
        low = sh.lower()
        if ("tableau" in low and "crois" in low) or ("pivot" in low):
            tcd_name = sh
            break
    if tcd_name is None:
        return {}

    try:
        raw = pd.read_excel(io.BytesIO(xlsx_bytes), sheet_name=tcd_name, header=None, engine="openpyxl")
    except Exception:
        return {}
    if raw.empty:
        return {}

    # Find the row that carries date-like headers.
    hdr_idx = None
    best = -1
    for i in range(len(raw)):
        row = raw.iloc[i]
        sc = sum(1 for v in row.values if norm_ddmm(v))
        if sc > best:
            best = sc
            hdr_idx = i
    if hdr_idx is None or best <= 0:
        return {}

    hdr = raw.iloc[hdr_idx]
    col_to_date = {}
    for j, v in enumerate(hdr.values):
        d = norm_ddmm(v)
        if d:
            col_to_date[j] = d
    if not col_to_date:
        return {}

    # Find Grand Total / Total row.
    total_idx = None
    for i in range(hdr_idx + 1, len(raw)):
        first = raw.iat[i, 0] if raw.shape[1] > 0 else None
        s = str(first).strip().lower() if pd.notna(first) else ""
        if any(k in s for k in ["grand total", "total général", "total general", "total"]):
            total_idx = i
            if "grand total" in s or "général" in s or "general" in s:
                break
    if total_idx is None:
        return {}

    out = {}
    for j, d in col_to_date.items():
        out[d] = to_float(raw.iat[total_idx, j])
    return out

def read_tcd_sg_daily(xlsx_bytes):
    """Read SG/day values directly from pivot-table sheet."""
    try:
        xls = pd.ExcelFile(io.BytesIO(xlsx_bytes), engine="openpyxl")
    except Exception:
        return pd.DataFrame(columns=["SG", "Date", "SR"])

    tcd_name = None
    for sh in xls.sheet_names:
        low = sh.lower()
        if ("tableau" in low and "crois" in low) or ("pivot" in low):
            tcd_name = sh
            break
    if tcd_name is None:
        return pd.DataFrame(columns=["SG", "Date", "SR"])

    try:
        raw = pd.read_excel(io.BytesIO(xlsx_bytes), sheet_name=tcd_name, header=None, engine="openpyxl")
    except Exception:
        return pd.DataFrame(columns=["SG", "Date", "SR"])
    if raw.empty:
        return pd.DataFrame(columns=["SG", "Date", "SR"])

    hdr_idx = None
    best = -1
    for i in range(len(raw)):
        row = raw.iloc[i]
        sc = sum(1 for v in row.values if norm_ddmm(v))
        if sc > best:
            best = sc
            hdr_idx = i
    if hdr_idx is None or best <= 0:
        return pd.DataFrame(columns=["SG", "Date", "SR"])

    hdr = raw.iloc[hdr_idx]
    col_to_date = {}
    for j, v in enumerate(hdr.values):
        d = norm_ddmm(v)
        if d:
            col_to_date[j] = d
    if not col_to_date:
        return pd.DataFrame(columns=["SG", "Date", "SR"])

    rows = []
    for i in range(hdr_idx + 1, len(raw)):
        sg = raw.iat[i, 0] if raw.shape[1] > 0 else None
        if pd.isna(sg):
            continue
        sg = str(sg).strip()
        if not sg:
            continue
        sgl = sg.lower()
        if any(k in sgl for k in ["grand total", "total général", "total general", "total"]):
            continue
        for j, d in col_to_date.items():
            rows.append({
                "SG": sg,
                "Date": d,
                "SR": to_float(raw.iat[i, j]),
            })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.groupby(["SG", "Date"], as_index=False)["SR"].sum(min_count=1)
    return out

def read_asfim_sr(xlsx_bytes):
    """Read per-fund (OPCVM) S&R data from ASFIM sheet.

    Priority: if a S&R JJ/MM column exists for a date, read it directly.
    Fallback: recalculate from consecutive AN/VL — SR_t = AN_t - AN_{t-1} * (VL_t / VL_{t-1}).
    This correctly handles both daily files (recalc) and weekly files where
    the precomputed S&R column already accounts for the right reference date.
    """
    try:
        df = pd.read_excel(io.BytesIO(xlsx_bytes), sheet_name="ASFIM", header=1, engine="openpyxl")
    except Exception:
        return pd.DataFrame(columns=["OPCVM", "SG", "Classification", "Date", "SR"])
    if df.empty:
        return pd.DataFrame(columns=["OPCVM", "SG", "Classification", "Date", "SR"])

    opcvm_col = find_col(df.columns, ["opcvm"])
    sg_col    = find_col(df.columns, ["société de gestion", "societe de gestion"])
    cls_col   = find_col(df.columns, ["classification"])
    if opcvm_col is None or sg_col is None:
        return pd.DataFrame(columns=["OPCVM", "SG", "Classification", "Date", "SR"])

    # Colonnes S&R directes (priorité)
    sr_by_date = {}
    for c in df.columns:
        cs = str(c).strip().lower()
        if cs.startswith("s&r") and norm_ddmm(str(c)):
            d = norm_ddmm(str(c))
            if d and d not in sr_by_date:
                sr_by_date[d] = c

    # Colonnes AN et VL (fallback recalcul)
    an_by_date = {}
    for c in df.columns:
        if re.match(r"^(an\b|actif.?net)", str(c).strip().lower()) and norm_ddmm(str(c)):
            d = norm_ddmm(str(c))
            if d and d not in an_by_date:
                an_by_date[d] = c

    vl_by_date = {}
    for c in df.columns:
        if re.match(r"^(vl\b|valeur.?liquidative)", str(c).strip().lower()) and norm_ddmm(str(c)):
            d = norm_ddmm(str(c))
            if d and d not in vl_by_date:
                vl_by_date[d] = c

    # Dates avec colonne S&R directe — base de référence
    sr_dates = sorted(sr_by_date.keys(), key=mmdd_sort)
    last_sr_sort = mmdd_sort(sr_dates[-1]) if sr_dates else (0, 0)

    # Inclure uniquement les dates AN qui sont :
    #   (a) couvertes par une colonne S&R directe, OU
    #   (b) dans un écart ≤ 7 jours calendaires après la dernière date S&R connue
    #   → exclut les colonnes AN orphelines (ex: 26/12 à 8 mois du dernier S&R)
    def _within_range(d):
        s = mmdd_sort(d)
        # différence en "mois×31 + jours" — suffisant pour détecter les gros écarts
        delta = (s[0] - last_sr_sort[0]) * 31 + (s[1] - last_sr_sort[1])
        return delta <= 7

    dates = sorted(
        [d for d in an_by_date.keys() if d in sr_by_date or _within_range(d)],
        key=mmdd_sort,
    )
    if not dates:
        return pd.DataFrame(columns=["OPCVM", "SG", "Classification", "Date", "SR"])

    rows = []
    for _, row in df.iterrows():
        opcvm = str(row[opcvm_col]).strip() if pd.notna(row[opcvm_col]) else ""
        sg    = str(row[sg_col]).strip()    if pd.notna(row[sg_col])    else ""
        cls   = str(row[cls_col]).strip()   if cls_col and pd.notna(row.get(cls_col)) else ""
        if not opcvm or opcvm.lower() in {"nan", "none"}:
            continue
        if not sg or sg.lower() in {"nan", "none"}:
            continue

        prev_an, prev_vl, has_prev = np.nan, np.nan, False

        for date in dates:
            an_val = to_float(row[an_by_date[date]]) if date in an_by_date else np.nan
            vl_val = to_float(row[vl_by_date[date]]) if date in vl_by_date else np.nan

            # Priorité : lire S&R directement si la colonne existe
            if date in sr_by_date:
                sr_val = to_float(row[sr_by_date[date]])
            elif has_prev and pd.notna(an_val) and pd.notna(vl_val) and pd.notna(prev_an) and pd.notna(prev_vl) and abs(prev_vl) > 1e-12:
                sr_val = an_val - prev_an * (vl_val / prev_vl)
            else:
                sr_val = np.nan

            rows.append({
                "OPCVM": opcvm,
                "SG": sg,
                "Classification": cls,
                "Date": date,
                "SR": sr_val,
                "AN": an_val,
                "VL": vl_val,
            })

            if pd.notna(an_val) and pd.notna(vl_val) and abs(vl_val) > 1e-12:
                prev_an, prev_vl, has_prev = an_val, vl_val, True

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    ytd = out.groupby(["OPCVM", "SG", "Classification"], as_index=False)["SR"].sum(min_count=1).rename(columns={"SR": "YTD"})
    out = out.merge(ytd, on=["OPCVM", "SG", "Classification"], how="left")
    return out

def read_asfim_sg_classification(xlsx_bytes):
    """Read SG -> Classification mapping from ASFIM sheet."""
    try:
        df = pd.read_excel(io.BytesIO(xlsx_bytes), sheet_name="ASFIM", header=1, engine="openpyxl")
    except Exception:
        return pd.DataFrame(columns=["SG", "Classification"])
    if df.empty:
        return pd.DataFrame(columns=["SG", "Classification"])

    sg_col = find_col(df.columns, ["société de gestion", "societe de gestion"])
    cls_col = find_col(df.columns, ["classification"])
    if sg_col is None or cls_col is None:
        return pd.DataFrame(columns=["SG", "Classification"])

    out = df[[sg_col, cls_col]].copy()
    out.columns = ["SG", "Classification"]
    out["SG"] = out["SG"].astype(str).str.strip()
    out["Classification"] = out["Classification"].astype(str).str.strip()
    out = out[
        out["SG"].ne("")
        & out["Classification"].ne("")
        & out["SG"].str.lower().ne("nan")
        & out["Classification"].str.lower().ne("nan")
    ].drop_duplicates().reset_index(drop=True)
    return out

# ──────────────────────────────────────────────
# EXCEL READER
# ──────────────────────────────────────────────
def read_recap(xlsx_bytes, sheet_name):
    raw = pd.read_excel(io.BytesIO(xlsx_bytes), sheet_name=sheet_name, header=None, engine="openpyxl")
    if raw.empty or raw.shape[0] < 2:
        return raw
    hdr = raw.iloc[0].tolist()
    # Pre-process datetime objects BEFORE unique_cols converts them to strings
    # openpyxl returns date cells as datetime objects; str() gives "2026-04-13 00:00:00"
    # which causes regex to match "26-04" instead of "13-04"
    processed_hdr = []
    for h in hdr:
        if hasattr(h, 'day') and hasattr(h, 'month'):
            processed_hdr.append(f"S&R {h.day:02d}/{h.month:02d}")
        elif hasattr(h, 'strftime') and callable(h.strftime):
            try:
                processed_hdr.append(f"S&R {h.day:02d}/{h.month:02d}")
            except Exception:
                processed_hdr.append(h)
        else:
            processed_hdr.append(h)
    cols = unique_cols(processed_hdr)
    df = raw.iloc[1:].copy()
    df.columns = cols
    first = df.columns[0]
    inject = {c: np.nan for c in df.columns}
    inject[first] = str(hdr[0]).strip() if hdr and hdr[0] is not None else ""
    df = pd.concat([pd.DataFrame([inject]), df], ignore_index=True)
    return df.dropna(how="all").reset_index(drop=True)

# ──────────────────────────────────────────────
# PARSER (KEEP YTD, REMOVE DEC & PART)
# ──────────────────────────────────────────────
def parse_recap_sr(df):
    df = df.copy()
    df.columns = unique_cols(df.columns)
    df = df.reset_index(drop=True)

    date_cols = [c for c in df.columns if norm_ddmm(c) and "s&r" in str(c).lower()]

    ytd_col = None
    for c in df.columns:
        if str(c).strip().upper() in {"TOTAL YTD", "TOTAL"}:
            ytd_col = c
            break

    # REMOVE DEC + PART: we do NOT look for them anymore
    non_date = [c for c in df.columns if c not in date_cols and c != ytd_col]

    col_scores = []
    for c in non_date:
        s = df[c].astype(str).fillna("")
        txt = s.str.strip().str.lower()
        ne = (txt != "") & (txt != "nan") & (txt != "none")
        num = s.apply(lambda v: pd.notna(to_float(v)))
        sc = ne.mean() - 0.6 * num.mean()
        col_scores.append((sc, c))
    col_scores.sort(reverse=True)

    sg_col = col_scores[0][1] if col_scores else df.columns[0]
    # Second text column = fund/OPCVM name (if it exists and is different)
    fonds_col = col_scores[1][1] if len(col_scores) >= 2 else None

    def row_txt(row):
        return " ".join(
            str(v).strip()
            for v in row.values
            if pd.notna(v) and str(v).strip().lower() not in {"nan", "none", ""}
        ).upper()

    rows = []
    bloc = None

    for i in range(len(df)):
        row = df.iloc[i]
        t = row_txt(row)

        if "OPCVM" in t and "ACTION" in t:
            bloc = "ACTIONS"
            continue
        if "OPCVM" in t and "DIVERS" in t:
            bloc = "DIVERSIFIES"
            continue
        if "OPCVM" in t and "CONTRAC" in t:
            bloc = "CONTRACTUEL"
            continue
        if "OPCVM" in t and ("MONETAIRE" in t or "MONÉTAIRE" in t or "MARCHE" in t or "MARCHÉ" in t):
            bloc = "MONETAIRE"
            continue
        if "OPCVM" in t and "OCT" in t:
            bloc = "OCT"
            continue
        if "OPCVM" in t and ("OMLT" in t or "OBLIG" in t):
            bloc = "OMLT"
            continue
        # Section agrégée "ALL OPCVM" ou "ENSEMBLE" → stop parsing
        if ("ALL OPCVM" in t or "ENSEMBLE" in t or
                ("OPCVM" in t and t.strip().startswith("ALL"))):
            bloc = None
            continue
        if bloc is None:
            continue

        sg = row.get(sg_col, row.iloc[0])
        if pd.isna(sg):
            continue
        sg = str(sg).strip()
        if not sg or sg.lower() in {"nan", "none"}:
            continue
        if is_total(sg):
            continue

        has_num = any(pd.notna(to_float(row.get(c))) for c in date_cols)
        if (not has_num) and ytd_col:
            has_num = pd.notna(to_float(row.get(ytd_col)))
        if not has_num:
            continue

        y = to_float(row.get(ytd_col)) if ytd_col else np.nan

        fonds = None
        if fonds_col:
            fonds_val = row.get(fonds_col)
            if pd.notna(fonds_val):
                fonds_str = str(fonds_val).strip()
                if fonds_str and fonds_str.lower() not in {"nan", "none"}:
                    fonds = fonds_str

        for c in date_cols:
            rows.append({
                "Bloc": bloc,
                "Fonds": fonds,
                "SG": sg,
                "Date": norm_ddmm(c),
                "SR": to_float(row.get(c)),
                "YTD_row": y,
            })

    tidy = pd.DataFrame(rows)
    if tidy.empty:
        return tidy
    tidy["k"] = tidy["Date"].apply(mmdd_sort)
    tidy = tidy.sort_values(["k", "Bloc", "SG"]).drop(columns=["k"])
    return tidy

# ──────────────────────────────────────────────
# AGGREGATE YTD (from ALL OPCVM section)
# ──────────────────────────────────────────────
def parse_all_opcvm_ytd(df):
    """Read per-SG total YTD from the 'ALL OPCVM' aggregate section of Recap S&R."""
    df = df.copy()
    df.columns = unique_cols(df.columns)
    df = df.reset_index(drop=True)

    ytd_col = None
    for c in df.columns:
        if str(c).strip().upper() in {"TOTAL YTD", "TOTAL"}:
            ytd_col = c
            break
    if ytd_col is None:
        return {}

    non_date = [c for c in df.columns
                if not (norm_ddmm(c) and "s&r" in str(c).lower()) and c != ytd_col]
    col_scores = []
    for c in non_date:
        s = df[c].astype(str).fillna("")
        txt = s.str.strip().str.lower()
        ne = (txt != "") & (txt != "nan") & (txt != "none")
        num = s.apply(lambda v: pd.notna(to_float(v)))
        sc = ne.mean() - 0.6 * num.mean()
        col_scores.append((sc, c))
    col_scores.sort(reverse=True)
    sg_col = col_scores[0][1] if col_scores else df.columns[0]

    def row_txt(row):
        return " ".join(
            str(v).strip()
            for v in row.values
            if pd.notna(v) and str(v).strip().lower() not in {"nan", "none", ""}
        ).upper()

    in_all = False
    ytd_map = {}

    for i in range(len(df)):
        row = df.iloc[i]
        t = row_txt(row)

        if ("ALL OPCVM" in t or "ENSEMBLE" in t or
                ("OPCVM" in t and t.strip().startswith("ALL"))):
            in_all = True
            continue

        if not in_all:
            continue

        # Stop at any new bloc section header
        if "OPCVM" in t and any(k in t for k in [
                "ACTION", "DIVERS", "CONTRAC", "MONETAIRE", "MONÉTAIRE",
                "MARCHE", "MARCHÉ", "OCT", "OMLT", "OBLIG"]):
            break

        sg = row.get(sg_col, row.iloc[0])
        if pd.isna(sg):
            continue
        sg = str(sg).strip()
        if not sg or sg.lower() in {"nan", "none"}:
            continue
        if is_total(sg):
            continue

        y = to_float(row.get(ytd_col)) if ytd_col else np.nan
        if pd.notna(y):
            ytd_map[sg] = y

    return ytd_map

# ──────────────────────────────────────────────
# ALTAIR LINE CHART
# ──────────────────────────────────────────────
def line_chart(data, x, y, color=None, title="", h=340):
    base = alt.Chart(data).mark_line(strokeWidth=2.5, point=alt.OverlayMarkDef(filled=True, size=40))
    enc = {
        "x": alt.X(f"{x}:T", title="Date", axis=alt.Axis(format="%m/%d")),
        "y": alt.Y(f"{y}:Q", title="S&R"),
    }
    tips = [
        alt.Tooltip(f"{x}:T", title="Date", format="%m/%d/%Y"),
        alt.Tooltip(f"{y}:Q", title="S&R", format=",.0f"),
    ]
    if color:
        enc["color"] = alt.Color(f"{color}:N", title=None, scale=alt.Scale(range=PALETTE))
        tips.append(alt.Tooltip(f"{color}:N", title="Bloc"))
    enc["tooltip"] = tips
    return (
        base.encode(**enc)
        .properties(title=title, height=h, background=BG)
        .configure_view(strokeWidth=0)
        .configure_axis(
            grid=True, gridColor=CHART_GRID,
            domainColor=BORDER, tickColor=BORDER,
            labelColor=SECONDARY, titleColor=PRIMARY,
        )
        .configure_title(color=PRIMARY, fontSize=14, fontWeight=700)
    )

# ──────────────────────────────────────────────
# HTML TABLE (NO DEC, NO PART, NO WOW)
# ──────────────────────────────────────────────
_COL_LABELS = {
    "Classification": "CLASSIFICATION",
    "Bloc": "BLOC",
    "SG": "SOCIÉTÉ DE GESTION",
    "S&R": "S&R",
    "YTD": "YTD",
}
_RIGHT = {"S&R", "YTD"}

def _fmt_cell(col, val):
    if col in _RIGHT:
        return fmt_money(val) if pd.notna(val) else "—"
    return "—" if pd.isna(val) else str(val)

def _cell_color(col, val):
    if col == "S&R" and pd.notna(val):
        return GREEN if val > 0 else RED if val < 0 else PRIMARY
    return PRIMARY

def html_table(df, columns, max_h=520):
    hdr = ""
    for c in columns:
        al = "right" if c in _RIGHT or c == "#" else "left"
        lab = _COL_LABELS.get(c, c)
        hdr += (
            f'<th style="text-align:{al};padding:10px 18px;font-weight:700;font-size:0.66rem;'
            f'letter-spacing:0.10em;text-transform:uppercase;white-space:nowrap;'
            f'border-bottom:2px solid {ACCENT};background:{SIDEBAR_BG};color:#9AAABB;'
            f'position:sticky;top:0;z-index:1;">{lab}</th>'
        )

    body = ""
    for i in range(len(df)):
        r = df.iloc[i]
        cells = ""
        for c in columns:
            val = r[c]
            al = "right" if c in _RIGHT or c == "#" else "left"
            txt = _fmt_cell(c, val)
            clr = _cell_color(c, val)
            # rank column style
            if c == "#":
                cells += (
                    f'<td style="text-align:right;padding:9px 18px;'
                    f'font-family:\'IBM Plex Mono\',monospace;font-size:0.75rem;'
                    f'color:{MUTED};font-weight:500;border-bottom:1px solid {BORDER};">{txt}</td>'
                )
            elif c == "S&R":
                bg_pill = "rgba(11,122,80,.08)" if (pd.notna(val) and isinstance(val,(int,float)) and val > 0) else \
                          "rgba(185,28,28,.08)" if (pd.notna(val) and isinstance(val,(int,float)) and val < 0) else "transparent"
                cells += (
                    f'<td style="text-align:right;padding:9px 18px;border-bottom:1px solid {BORDER};">'
                    f'<span style="display:inline-block;padding:3px 10px;border-radius:6px;'
                    f'background:{bg_pill};font-family:\'IBM Plex Mono\',monospace;'
                    f'font-size:0.83rem;font-weight:700;color:{clr};letter-spacing:-0.02em;">{txt}</span></td>'
                )
            elif c in {"SG", "OPCVM"}:
                cells += (
                    f'<td style="text-align:left;padding:9px 18px;color:{PRIMARY};'
                    f'font-weight:600;font-size:0.85rem;border-bottom:1px solid {BORDER};'
                    f'max-width:260px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">{txt}</td>'
                )
            elif c == "Classification" or c == "Bloc":
                cells += (
                    f'<td style="text-align:left;padding:9px 18px;border-bottom:1px solid {BORDER};">'
                    f'<span style="display:inline-block;padding:2px 8px;border-radius:4px;'
                    f'background:{SURFACE};border:1px solid {BORDER};'
                    f'font-size:0.70rem;font-weight:600;color:{SECONDARY};'
                    f'text-transform:uppercase;letter-spacing:0.06em;">{txt}</span></td>'
                )
            else:
                cells += (
                    f'<td style="text-align:{al};padding:9px 18px;color:{clr};font-weight:500;'
                    f'font-size:0.83rem;border-bottom:1px solid {BORDER};white-space:nowrap;">{txt}</td>'
                )
        hover_bg = "#F7FAFD"
        body += (
            f'<tr style="background:{BG};transition:background .10s;" '
            f'onmouseover="this.style.background=\'{hover_bg}\'" '
            f'onmouseout="this.style.background=\'{BG}\'">{cells}</tr>'
        )

    # colgroup for fixed column widths
    colgroup = "<colgroup>"
    for c in columns:
        if c == "#":            colgroup += '<col style="width:44px">'
        elif c == "S&R":        colgroup += '<col style="width:150px">'
        elif c == "YTD":        colgroup += '<col style="width:150px">'
        elif c in {"Classification","Bloc"}: colgroup += '<col style="width:120px">'
        elif c == "SG":         colgroup += '<col style="width:220px">'
        elif c == "OPCVM":      colgroup += '<col style="min-width:180px">'
        else:                   colgroup += '<col>'
    colgroup += "</colgroup>"

    return (
        f'<div style="border:1px solid {BORDER};border-radius:8px;overflow:hidden;'
        f'box-shadow:0 1px 6px rgba(0,0,0,.06);">'
        f'<div style="max-height:{max_h}px;overflow-y:auto;overflow-x:auto;">'
        f'<table style="width:100%;border-collapse:collapse;background:{BG};'
        f'font-family:\'Montserrat\',sans-serif;">'
        f'{colgroup}'
        f'<thead><tr>{hdr}</tr></thead><tbody>{body}</tbody></table></div></div>'
    )

# ──────────────────────────────────────────────
# TOP/FLOP CARD (NO DEC, NO PART, NO WOW)
# ──────────────────────────────────────────────
def _card(sg, bloc, sr, ytd, accent):
    sr_c = GREEN if (pd.notna(sr) and sr >= 0) else RED
    def _r(label, value, color=PRIMARY):
        return (
            f'<div style="display:flex;justify-content:space-between;padding:3px 0;font-size:0.85rem;">'
            f'<span style="color:{MUTED};font-weight:500;">{label}</span>'
            f'<span style="color:{color};font-weight:600;">{value}</span></div>'
        )
    return (
        f'<div style="background:{BG};border:1px solid {BORDER};border-radius:14px;'
        f'padding:16px 20px;margin-bottom:12px;border-left:4px solid {accent};'
        f'box-shadow:0 1px 4px rgba(15,23,42,.03);">'
        f'<div style="font-weight:700;color:{PRIMARY};font-size:1.05rem;margin-bottom:1px;">{sg}</div>'
        f'<div style="color:{MUTED};font-weight:500;font-size:0.78rem;margin-bottom:12px;'
        f'text-transform:uppercase;letter-spacing:0.04em;">{bloc}</div>'
        + _r("S&R", fmt_money(sr), sr_c)
        + _r("YTD", fmt_money(ytd))
        + '</div>'
    )

def export_button(df, filename, label="⬇ Exporter Excel"):
    filename = filename.replace(".csv", ".xlsx")
    buf = io.BytesIO()
    df.to_excel(buf, index=False, engine="openpyxl")
    st.download_button(label=label, data=buf.getvalue(), file_name=filename,
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                       use_container_width=False)

def section_header(title, subtitle=""):
    """Render a clean BMCE-style section header."""
    sub_html = f'<div style="font-size:0.75rem;color:{MUTED};margin-top:3px;font-weight:400;">{subtitle}</div>' if subtitle else ""
    st.markdown(
        f'<div style="margin:28px 0 16px 0;padding-bottom:10px;border-bottom:1px solid {BORDER};">'
        f'<div style="display:flex;align-items:center;gap:10px;">'
        f'<span style="display:inline-block;width:4px;height:20px;background:{ACCENT};border-radius:2px;flex-shrink:0;"></span>'
        f'<div><div style="font-size:1.0rem;font-weight:700;color:{PRIMARY};">{title}</div>{sub_html}</div>'
        f'</div></div>',
        unsafe_allow_html=True,
    )

# ══════════════════════════════════════════════
#  LAYOUT
# ══════════════════════════════════════════════
# ── HEADER ──
logo_html = ""
logo_b64 = ""
logo_ext = "jpeg"
_logo_fixed = Path("/Users/mac/Desktop/DAHSBOARD BMCE/bmce_capital_logo.jpeg")
_logo_src = _logo_fixed if _logo_fixed.exists() else logo_path
if _logo_src:
    import base64
    try:
        with open(str(_logo_src), "rb") as _f:
            logo_b64 = base64.b64encode(_f.read()).decode()
        logo_ext = str(_logo_src).rsplit(".", 1)[-1].lower().replace("jpg", "jpeg")
        logo_html = f'<img src="data:image/{logo_ext};base64,{logo_b64}" style="height:38px;object-fit:contain;display:block;">'
    except Exception:
        pass

logo_right_html = (
    f'<img src="data:image/{logo_ext};base64,{logo_b64}" '
    f'style="height:100px;object-fit:contain;display:block;">'
) if logo_b64 else ""

st.markdown(
    f'<div style="padding:52px 0 0 0;margin-bottom:8px;">'
    f'  <div style="display:flex;align-items:center;gap:16px;padding-bottom:14px;border-bottom:2px solid {BORDER};">'
    f'    {logo_html}'
    f'    <div style="width:1px;height:32px;background:{BORDER};flex-shrink:0;"></div>'
    f'    <div style="flex:1;">'
    f'      <div style="font-size:1.25rem;font-weight:700;color:{PRIMARY};line-height:1.2;">Dashboard S&amp;R — ASFIM</div>'
    f'      <div style="font-size:.78rem;color:{MUTED};font-weight:500;margin-top:1px;">Pilotage des flux · vue marché, classements et tendances</div>'
    f'    </div>'
    f'    {logo_right_html}'
    f'  </div>'
    f'</div>',
    unsafe_allow_html=True,
)

_CACHE_PATH = Path(__file__).parent / ".bmce_cache" / "Analyse_SR.xlsx"
_CACHE_PATH.parent.mkdir(exist_ok=True)

# ── Bouton mise à jour quotidienne ──
import subprocess
_btn_col, _status_col = st.columns([1, 3])
with _btn_col:
    _maj_clicked = st.button(
        "🔄 Mettre à jour les données du jour",
        use_container_width=True,
        key="btn_maj_quotidienne",
    )
with _status_col:
    _maj_placeholder = st.empty()

if _maj_clicked:
    import time as _time
    _t_start = _time.time()
    with st.spinner("Mise à jour en cours…"):
        _maj_ok = True
        _maj_errors = []

        # Étape 1 : script bash de préparation
        try:
            _proc = subprocess.run(
                ["bash", "/Users/mac/.n8n-files/prepare_daily.sh"],
                capture_output=True, text=True, timeout=120
            )
            if _proc.returncode != 0:
                _maj_errors.append(f"Script : {_proc.stderr.strip() or 'erreur inconnue'}")
                _maj_ok = False
        except Exception as _e:
            _maj_errors.append(f"Script : {_e}")
            _maj_ok = False

        # Étape 2 : webhook n8n
        if _maj_ok:
            try:
                _r = requests.get(
                    "http://localhost:5678/webhook/6f36323d-ac88-412c-8337-3065e273d6ba",
                    timeout=60
                )
                if _r.status_code >= 400:
                    _maj_errors.append(f"Webhook : HTTP {_r.status_code}")
                    _maj_ok = False
            except Exception as _e:
                _maj_errors.append(f"Webhook : {_e}")
                _maj_ok = False

    if _maj_ok:
        # Attendre que n8n ait fini d'écrire le fichier (max 30s)
        import time as _time
        _maj_placeholder.info("⏳ Finalisation…")
        for _i in range(30):
            if _CACHE_PATH.exists() and _CACHE_PATH.stat().st_mtime > _t_start:
                break
            _time.sleep(1)
        _maj_placeholder.success("✅ Données mises à jour !")
        st.rerun()
    else:
        _maj_placeholder.error("❌ " + " | ".join(_maj_errors))

with st.sidebar:
    st.markdown(f"<div style='font-weight:700;color:{PRIMARY};font-size:1rem;'>Paramètres</div>", unsafe_allow_html=True)
    st.markdown("---")

    # ── Analyse SR Excel (optionnel si historique AN disponible) ──
    up = st.file_uploader("Analyse_SR.xlsx (optionnel)", type=["xlsx"])
    if up is not None:
        _CACHE_PATH.write_bytes(up.getvalue())
    if _CACHE_PATH.exists():
        _cache_mtime = date.fromtimestamp(_CACHE_PATH.stat().st_mtime)
        st.caption(f"Excel en mémoire : {_cache_mtime.strftime('%d/%m/%Y')}")
        if st.button("🗑 Effacer l'Excel sauvegardé", key="clear_cache"):
            _CACHE_PATH.unlink(missing_ok=True)
            st.rerun()

    st.markdown("---")

    # ── Tableau de performances ASFIM (PDF) ──
    st.markdown(
        f"<div style='font-weight:600;color:{PRIMARY};font-size:0.85rem;margin-bottom:6px;'>"
        f"📄 Tableau de performances ASFIM</div>",
        unsafe_allow_html=True,
    )
    up_perf = st.file_uploader("Uploader le Tableau de performances (.xlsx)", type=["xlsx"], key="perf_xlsx")
    if up_perf is not None:
        with st.spinner("Lecture du fichier…"):
            _perf_df_new, _perf_date_auto = parse_tableau_performances_excel(up_perf.getvalue())
        if _perf_df_new.empty:
            st.error("Fichier non reconnu — vérifiez le format.")
        else:
            _perf_date_input = st.text_input(
                "Date (JJ/MM)", value=_perf_date_auto or "", key="perf_date_input",
                help="Détectée automatiquement depuis le nom de la feuille (ex: 22-04-2026 → 22/04)"
            )
            if st.button("💾 Enregistrer ce snapshot", key="save_perf"):
                d_ok = norm_ddmm(_perf_date_input) if _perf_date_input else None
                if d_ok:
                    save_perf_snapshot(_perf_df_new, d_ok)
                    st.success(f"✓ {len(_perf_df_new)} fonds enregistrés pour le {d_ok}")
                    st.rerun()
                else:
                    st.error("Date invalide — format attendu : JJ/MM")

    # Statut du cache AN (perf_history)
    _perf_hist_sidebar = load_perf_history()
    _perf_dates_sidebar = sorted(
        set(d for v in _perf_hist_sidebar.values() for d in v.get("snapshots", {}).keys()),
        key=mmdd_sort,
    ) if _perf_hist_sidebar else []
    if _perf_dates_sidebar:
        st.caption(
            f"📦 AN mémorisés : {', '.join(_perf_dates_sidebar[-4:])}"
            + (" …" if len(_perf_dates_sidebar) > 4 else "")
        )

    # Statut du cache asfim_sr
    if _ASFIM_SR_CACHE_PATH.exists():
        try:
            _sr_cache_dates = sorted(
                pd.read_json(_ASFIM_SR_CACHE_PATH, orient="records")["Date"].dropna().unique().tolist(),
                key=mmdd_sort,
            )
            st.caption(
                f"📊 S&R en cache : {len(_sr_cache_dates)} dates · "
                f"jusqu'au {_sr_cache_dates[-1] if _sr_cache_dates else '—'}"
            )
        except Exception:
            pass

    st.markdown("---")
    # Bouton reset complet
    if st.button("🗑 Réinitialiser tout le cache", key="clear_all"):
        for _p in [_CACHE_PATH, _PERF_CACHE_PATH, _ASFIM_SR_CACHE_PATH]:
            _p.unlink(missing_ok=True)
        st.rerun()

    st.markdown("---")
    show_audit = st.checkbox("Afficher le fichier en entier", value=False)

# ── Injection JS : corrige file uploader + boutons fullscreen/vega ──
components.html(f"""
<script>
(function(){{
  var BG = '#FAFAFA';
  var ACCENT = '{ACCENT}';
  var PRIMARY = '{PRIMARY}';
  var BORDER = '{BORDER}';

  function patch(){{
    try{{
      var doc = window.parent.document;

      // ── File uploader ──
      var sel = '[data-testid="stFileUploader"] *,' +
                '[data-testid="stFileUploadDropzone"],' +
                '[data-testid="stFileUploadDropzone"] *';
      doc.querySelectorAll(sel).forEach(function(el){{
        var tag = el.tagName && el.tagName.toLowerCase();
        if(tag === 'button'){{
          el.style.setProperty('background','#ffffff','important');
          el.style.setProperty('background-color','#ffffff','important');
          el.style.setProperty('border','1.5px solid '+ACCENT,'important');
          el.style.setProperty('border-radius','4px','important');
          el.style.setProperty('color',ACCENT,'important');
          el.style.setProperty('-webkit-text-fill-color',ACCENT,'important');
          el.style.setProperty('box-shadow','none','important');
        }} else {{
          el.style.setProperty('background',BG,'important');
          el.style.setProperty('background-color',BG,'important');
          el.style.setProperty('box-shadow','none','important');
        }}
      }});
      doc.querySelectorAll('[data-testid="stFileUploadDropzone"]').forEach(function(el){{
        el.style.setProperty('border','1.5px dashed #E0E0E0','important');
        el.style.setProperty('border-radius','6px','important');
      }});

      // ── Boutons fullscreen Streamlit (tous contextes) ──
      doc.querySelectorAll('button[data-testid="StyledFullScreenButton"], button[title="View fullscreen"], button[title="Exit fullscreen"], [data-testid="stVegaLiteChart"] button, [data-testid="element-container"] button').forEach(function(el){{
        el.style.setProperty('background','#ffffff','important');
        el.style.setProperty('background-color','#ffffff','important');
        el.style.setProperty('border','1px solid '+BORDER,'important');
        el.style.setProperty('border-radius','4px','important');
        el.style.setProperty('box-shadow','0 1px 4px rgba(0,0,0,.08)','important');
        el.style.setProperty('color',PRIMARY,'important');
        // SVG inside
        el.querySelectorAll('svg, path').forEach(function(s){{
          s.style.setProperty('fill',PRIMARY,'important');
          s.style.setProperty('stroke',PRIMARY,'important');
          s.style.setProperty('color',PRIMARY,'important');
        }});
      }});

      // ── Vega-Embed summary (bouton "...") ──
      doc.querySelectorAll('.vega-embed summary, .vega-embed details summary').forEach(function(el){{
        el.style.setProperty('background','#ffffff','important');
        el.style.setProperty('background-color','#ffffff','important');
        el.style.setProperty('border','1px solid '+BORDER,'important');
        el.style.setProperty('border-radius','4px','important');
        el.style.setProperty('box-shadow','0 1px 4px rgba(0,0,0,.08)','important');
        el.querySelectorAll('svg, path').forEach(function(s){{
          s.style.setProperty('fill',PRIMARY,'important');
          s.style.setProperty('stroke',PRIMARY,'important');
        }});
      }});

      // ── Vega actions dropdown ──
      doc.querySelectorAll('.vega-actions, .vega-actions a').forEach(function(el){{
        el.style.setProperty('background','#ffffff','important');
        el.style.setProperty('background-color','#ffffff','important');
        el.style.setProperty('color',PRIMARY,'important');
        el.style.setProperty('-webkit-text-fill-color',PRIMARY,'important');
      }});

    }}catch(e){{}}
  }}

  // Appliquer immédiatement puis en cascade
  patch();
  [100,300,600,1000,2000,4000].forEach(function(t){{ setTimeout(patch, t); }});

  // MutationObserver sans filtre d'attributs pour attraper tout ajout DOM
  var obs = new MutationObserver(function(){{ patch(); }});
  try{{
    obs.observe(window.parent.document.body,
      {{childList:true,subtree:true}});
  }}catch(e){{}}

  // Injecter aussi une feuille de style dans le parent pour les cas CSS purs
  try{{
    var style = window.parent.document.createElement('style');
    style.id = 'bmce-fix-btns';
    style.textContent = [
      'button[data-testid="StyledFullScreenButton"],button[title="View fullscreen"],button[title="Exit fullscreen"],[data-testid="stVegaLiteChart"] button,[data-testid="element-container"] button{{background:#fff!important;background-color:#fff!important;border:1px solid #E0E0E0!important;border-radius:4px!important;box-shadow:none!important;color:#1C1C1C!important;}}',
      'button[data-testid="StyledFullScreenButton"] svg,button[title="View fullscreen"] svg,[data-testid="stVegaLiteChart"] button svg{{fill:#1C1C1C!important;stroke:none!important;}}',
      'button[data-testid="StyledFullScreenButton"] path{{fill:#1C1C1C!important;}}',
      '.vega-embed summary{{background:#fff!important;background-color:#fff!important;border:1px solid #E0E0E0!important;border-radius:4px!important;box-shadow:none!important;}}',
      '.vega-embed summary svg,.vega-embed summary path{{fill:#1C1C1C!important;}}'
    ].join('');
    if(!window.parent.document.getElementById('bmce-fix-btns')){{
      window.parent.document.head.appendChild(style);
    }}
  }}catch(e){{}}
}})();
</script>
""", height=0)

# ── Disponibilité des sources de données ──
_perf_history       = load_perf_history()
_perf_dates_all     = sorted(
    set(d for v in _perf_history.values() for d in v.get("snapshots", {}).keys()),
    key=mmdd_sort,
) if _perf_history else []
_perf_has_new       = len(_perf_dates_all) >= 2   # assez pour calculer un S&R
_xlsx_available     = up is not None or _CACHE_PATH.exists()
_asfim_cache_avail  = _ASFIM_SR_CACHE_PATH.exists()

# Premier lancement : aucune source disponible
if not _xlsx_available and not _asfim_cache_avail:
    st.markdown(
        f'<div style="margin:40px auto;max-width:480px;text-align:center;padding:48px 32px;'
        f'border:1px solid {BORDER};border-radius:8px;background:{SURFACE};">'
        f'<div style="font-size:2.5rem;margin-bottom:16px;">🔄</div>'
        f'<div style="font-size:1.1rem;font-weight:700;color:{PRIMARY};margin-bottom:12px;">Aucune donnée disponible</div>'
        f'<div style="font-size:.9rem;color:{MUTED};line-height:1.8;">'
        f'Cliquez le bouton <strong>🔄 Mettre à jour les données du jour</strong><br>'
        f'en haut de page pour charger automatiquement les données.'
        f'</div></div>',
        unsafe_allow_html=True,
    )
    st.stop()

YEAR = date.today().year

# ── Étape 1 : charger les données de base ──
if _xlsx_available:
    # Source principale : Analyse SR Excel
    xlsx_bytes = up.getvalue() if up is not None else _CACHE_PATH.read_bytes()
    xls           = pd.ExcelFile(io.BytesIO(xlsx_bytes), engine="openpyxl")
    tcd_totals    = read_tcd_daily_totals(xlsx_bytes)
    tcd_sg_daily  = read_tcd_sg_daily(xlsx_bytes)
    asfim_cls_map = read_asfim_sg_classification(xlsx_bytes)
    asfim_sr      = read_asfim_sr(xlsx_bytes)

    recap_name = None
    for sh in xls.sheet_names:
        low = sh.lower()
        if "recap" in low and ("s&r" in low or "sr" in low):
            recap_name = sh
            break
    if recap_name is None:
        for sh in xls.sheet_names:
            if "recap" in sh.lower():
                recap_name = sh
                break
    if recap_name is None:
        st.error("Feuille 'Recap S&R' introuvable.")
        st.stop()

    df_recap      = read_recap(xlsx_bytes, recap_name)
    tidy          = parse_recap_sr(df_recap)
    tidy          = apply_weekly_bridge_fix(tidy)
    all_opcvm_ytd = parse_all_opcvm_ytd(df_recap)

    # Auto-seed perf_history avec AN de la dernière date de l'ASFIM sheet
    _seed_df = extract_an_from_asfim(asfim_sr)
    if not _seed_df.empty:
        for _sd in _seed_df["Date"].dropna().unique():
            _sd_norm = norm_ddmm(_sd)
            if _sd_norm:
                save_perf_snapshot(
                    _seed_df[_seed_df["Date"] == _sd].copy(),
                    _sd_norm,
                    overwrite=False,
                )

    # Persister asfim_sr complet pour les prochaines sessions sans Excel
    save_asfim_sr_cache(asfim_sr)

else:
    # Source : cache persisté depuis le dernier chargement Excel
    asfim_sr      = load_asfim_sr_cache()
    tcd_sg_daily  = pd.DataFrame(columns=["SG", "Date", "SR"])
    if not asfim_sr.empty:
        tidy          = build_tidy_from_asfim_sr(asfim_sr[asfim_sr["SR"].notna()])
        tcd_totals    = asfim_sr[asfim_sr["SR"].notna()].groupby("Date")["SR"].sum().to_dict()
        asfim_cls_map = asfim_sr[["SG", "Classification"]].drop_duplicates().reset_index(drop=True)
        all_opcvm_ytd = asfim_sr.groupby("SG")["YTD"].max().to_dict()
    else:
        tidy          = pd.DataFrame(columns=["Bloc", "Fonds", "SG", "Date", "SR", "YTD_row"])
        tcd_totals    = {}
        asfim_cls_map = pd.DataFrame(columns=["SG", "Classification"])
        all_opcvm_ytd = {}

    # Bannière source cache
    _cache_dates = sorted(asfim_sr["Date"].dropna().unique().tolist(), key=mmdd_sort) if not asfim_sr.empty else []
    st.markdown(
        f'<div style="background:{SURFACE};border:1px solid {BORDER};border-left:4px solid {ACCENT};'
        f'border-radius:4px;padding:10px 16px;margin-bottom:12px;font-size:0.82rem;color:{SECONDARY};">'
        f'📦 Données en cache · {len(_cache_dates)} dates · jusqu\'au {_cache_dates[-1] if _cache_dates else "—"}'
        f'</div>',
        unsafe_allow_html=True,
    )

# ── Étape 2 : ajouter les nouvelles dates depuis les PDFs uploadés ──
if _perf_has_new:
    _asfim_sr_perf = compute_sr_from_perf_history(_perf_history)
    _perf_with_sr  = _asfim_sr_perf[_asfim_sr_perf["SR"].notna()] if not _asfim_sr_perf.empty else pd.DataFrame()

    if not _perf_with_sr.empty:
        _perf_dates_set = set(_perf_with_sr["Date"].dropna().unique())
        _existing_dates = set(asfim_sr["Date"].dropna().unique()) if not asfim_sr.empty else set()

        # Ne jamais écraser une date déjà dans l'Excel — uniquement ajouter les dates nouvelles
        # Exclure aussi les dates aberrantes (> 30 jours après la dernière date connue)
        _last_known = max(_existing_dates, key=mmdd_sort) if _existing_dates else None
        if _last_known:
            _lk = mmdd_sort(_last_known)
            def _perf_date_ok(d):
                s = mmdd_sort(d)
                delta = (s[0] - _lk[0]) * 31 + (s[1] - _lk[1])
                return delta <= 30
            _perf_dates_set = {d for d in _perf_dates_set if _perf_date_ok(d)}
        _new_dates = _perf_dates_set - _existing_dates

        if _new_dates:
            _new_rows = _perf_with_sr[_perf_with_sr["Date"].isin(_new_dates)].copy()

            asfim_sr = pd.concat([asfim_sr, _new_rows], ignore_index=True) \
                       if not asfim_sr.empty else _new_rows

            _tidy_new = build_tidy_from_asfim_sr(_new_rows)
            tidy = pd.concat([tidy, _tidy_new], ignore_index=True) \
                   if not tidy.empty else _tidy_new

            for _d in _new_dates:
                tcd_totals[_d] = _new_rows[_new_rows["Date"] == _d]["SR"].sum()

            save_asfim_sr_cache(asfim_sr)

            st.markdown(
                f'<div style="background:#F0FFF4;border:1px solid #A7F3D0;border-left:4px solid {GREEN};'
                f'border-radius:4px;padding:10px 16px;margin-bottom:12px;font-size:0.82rem;color:{PRIMARY};">'
                f'✓ Nouvelles dates ajoutées depuis le Tableau de performances : '
                f'{", ".join(sorted(_new_dates, key=mmdd_sort))}</div>',
                unsafe_allow_html=True,
            )

if tidy.empty:
    st.error("Impossible d'extraire les données.")
    st.stop()

# ── Remplacer tidy par les données ASFIM pour les dates où elles existent ──
# Le Recap S&R peut être mal rempli par des outils automatisés (doublons par section).
# La feuille ASFIM contient les valeurs correctes par fonds → on la préfère.
if not asfim_sr.empty:
    _asfim_sr_valid = asfim_sr[asfim_sr["SR"].notna()].copy()
    if not _asfim_sr_valid.empty:
        _asfim_dates = set(_asfim_sr_valid["Date"].dropna().unique())
        # Construire tidy depuis ASFIM pour ces dates
        _tidy_from_asfim = build_tidy_from_asfim_sr(_asfim_sr_valid)
        # Garder le tidy du Recap S&R pour les dates que l'ASFIM ne couvre pas
        _tidy_recap_only = tidy[~tidy["Date"].isin(_asfim_dates)]
        tidy = pd.concat([_tidy_recap_only, _tidy_from_asfim], ignore_index=True)
        # Reconstruire tcd_totals pour les dates ASFIM
        for _d in _asfim_dates:
            _day_sr = _asfim_sr_valid[_asfim_sr_valid["Date"] == _d]["SR"].sum(skipna=True)
            tcd_totals[_d] = _day_sr

# Enrichir tidy avec Date_dt
if "Date_dt" not in tidy.columns:
    tidy["Date_dt"] = tidy["Date"].apply(lambda s: ddmm_to_dt(s, YEAR))

if not asfim_cls_map.empty:
    class_options    = sorted(asfim_cls_map["Classification"].dropna().unique().tolist())
    class_filter     = class_options
    allowed_sg_sidebar = set(asfim_cls_map["SG"].dropna().unique().tolist())
else:
    class_options    = ["ACTIONS", "DIVERSIFIE", "OMLT"]
    class_filter     = class_options
    allowed_sg_sidebar = None

# Keep an unfiltered copy for drill-down category filtering.
tidy_all = tidy.copy()
if "Date_dt" not in tidy_all.columns:
    tidy_all["Date_dt"] = tidy_all["Date"].apply(lambda s: ddmm_to_dt(s, YEAR))

if allowed_sg_sidebar is not None:
    tidy = tidy[tidy["SG"].isin(allowed_sg_sidebar)]
else:
    tidy = tidy[tidy["Bloc"].isin(class_filter)]

if "Date_dt" not in tidy.columns:
    tidy["Date_dt"] = tidy["Date"].apply(lambda s: ddmm_to_dt(s, YEAR))

dates = sorted(tidy["Date"].dropna().unique().tolist(), key=mmdd_sort)
if not dates:
    st.error("Aucune date détectée.")
    st.stop()

last_date = dates[-1]

cA, _ = st.columns([2, 1])
with cA:
    date_sel = st.select_slider("Date (S&R)", options=dates, value=last_date)

# ──────────────────────────────────────────────
# RANKING TABLE (NO DEC, NO PART, KEEP YTD)
# ──────────────────────────────────────────────
pivot = (
    tidy.groupby(["Bloc", "SG", "Date"], as_index=False)["SR"]
    .sum(min_count=1)
    .pivot(index=["Bloc", "SG"], columns="Date", values="SR")
    .reset_index()
)
pivot.columns.name = None

ytd_map = tidy.groupby(["Bloc", "SG"])["YTD_row"].max().reset_index().rename(columns={"YTD_row": "YTD"})
ytd_sg_map = tidy.groupby("SG", as_index=False)["YTD_row"].max().rename(columns={"YTD_row": "YTD"})

df_rank = pivot.merge(ytd_map, on=["Bloc", "SG"], how="left")
df_rank["SR_sel"] = df_rank.get(date_sel, np.nan)

# KPI total must come directly from pivot table (Grand Total by date).
total_sel = tcd_totals.get(date_sel, np.nan)
if pd.isna(total_sel):
    total_sel = df_rank["SR_sel"].sum(skipna=True)

_ytd_note = "Le YTD provient directement de l'Excel." if _xlsx_available else "Le YTD est calculé par cumul des S&R issus des PDFs uploadés."
st.markdown(
    f'<div style="background:{SURFACE};border:1px solid {BORDER};border-radius:10px;padding:10px 14px;margin:10px 0 16px 0;font-size:0.86rem;color:{SECONDARY};">'
    f'<strong style="color:{PRIMARY};">Note :</strong> {_ytd_note}</div>',
    unsafe_allow_html=True,
)

sg_global = (
    df_rank.groupby("SG", as_index=False)["SR_sel"]
    .sum(min_count=1)
)
eps = 1e-9
sg_active = sg_global[sg_global["SR_sel"].abs() > eps]
nb_sg = int(sg_active["SG"].nunique())
nb_pos = int((sg_active["SR_sel"] > eps).sum())
nb_neg = int((sg_active["SR_sel"] < -eps).sum())

def _kpi(label, value, color=None):
    clr = color or PRIMARY
    return (
        f'<div style="background:#fff;border:1px solid {BORDER};border-top:3px solid {clr};'
        f'border-radius:4px;padding:14px 16px;box-shadow:0 1px 4px rgba(0,0,0,.05);">'
        f'<div style="font-size:.65rem!important;font-weight:800!important;text-transform:uppercase;letter-spacing:.09em;'
        f'color:{PRIMARY}!important;-webkit-text-fill-color:{PRIMARY}!important;margin-bottom:6px;">{label}</div>'
        f'<div style="font-size:1.25rem!important;font-weight:700!important;'
        f'color:{clr}!important;-webkit-text-fill-color:{clr}!important;'
        f'line-height:1.1;word-break:break-word;">{value}</div>'
        f'</div>'
    )

# Concentration : % des flux captés par le Top 3 SG (valeurs absolues)
_sg_global_abs = sg_global.copy()
_sg_global_abs["_abs"] = _sg_global_abs["SR_sel"].abs()
_top3_abs = _sg_global_abs.nlargest(3, "_abs")["_abs"].sum()
_total_abs = _sg_global_abs["_abs"].sum()
_conc_top3 = (_top3_abs / _total_abs) if _total_abs > 1e-9 else np.nan

k1, k2, k3, k4, k5 = st.columns(5)
k1.markdown(_kpi("Date", date_sel, ACCENT), unsafe_allow_html=True)
k2.markdown(_kpi("Collecte nette S&R", fmt_money(total_sel)), unsafe_allow_html=True)
k3.markdown(_kpi("Nb SG actives", nb_sg), unsafe_allow_html=True)
k4.markdown(_kpi("SG en collecte", nb_pos, GREEN), unsafe_allow_html=True)
k5.markdown(_kpi("SG en décollecte", nb_neg, RED), unsafe_allow_html=True)

# ── S&R total par catégorie ──
st.markdown("<div style='margin-top:12px'></div>", unsafe_allow_html=True)
_sr_by_cat = (
    tidy[tidy["Date"] == date_sel]
    .groupby("Bloc", as_index=False)["SR"].sum(min_count=1)
    .sort_values("SR", ascending=False)
)
if not _sr_by_cat.empty:
    _cat_cols = st.columns(1 + len(_sr_by_cat))
    _total_all = _sr_by_cat["SR"].sum(skipna=True)
    _cat_cols[0].markdown(
        _kpi("TOTAL — toutes catégories", fmt_money(_total_all),
             GREEN if _total_all >= 0 else RED),
        unsafe_allow_html=True
    )
    for _ci, _crow in enumerate(_sr_by_cat.itertuples()):
        _cv = _crow.SR
        _cc = GREEN if pd.notna(_cv) and _cv >= 0 else RED
        _cat_cols[_ci + 1].markdown(
            _kpi(_crow.Bloc, fmt_money(_cv), _cc),
            unsafe_allow_html=True
        )

st.markdown("---")

# ── CLASSEMENT ──
section_header("Classement par fonds", "S&R journalier par OPCVM · date sélectionnée")

_rf1, _rf2 = st.columns([1, 1])
with _rf1:
    if not asfim_sr.empty and "Classification" in asfim_sr.columns:
        cls_options_rank = ["TOUS"] + sorted(asfim_sr["Classification"].dropna().unique().tolist())
    else:
        cls_options_rank = ["TOUS"] + sorted(tidy["Bloc"].dropna().unique().tolist())
    rank_cls_sel = st.selectbox("Classification", cls_options_rank, index=0, key="rank_cls")
with _rf2:
    _rank_dates_opts = sorted(
        (asfim_sr["Date"].dropna().unique().tolist() if not asfim_sr.empty
         else tidy["Date"].dropna().unique().tolist()),
        key=mmdd_sort
    )
    rank_date_sel = st.selectbox("Date", _rank_dates_opts,
                                  index=len(_rank_dates_opts)-1, key="rank_date")

if not asfim_sr.empty:
    rank_src = asfim_sr[asfim_sr["Date"] == rank_date_sel].copy()
    if rank_cls_sel != "TOUS":
        rank_src = rank_src[rank_src["Classification"] == rank_cls_sel]
    rank_src = (
        rank_src[["Classification", "SG", "OPCVM", "SR", "YTD"]]
        .rename(columns={"SR": "S&R"})
        .sort_values("S&R", ascending=False)
        .reset_index(drop=True)
    )
    tmp = rank_src
    show_cols = ["Classification", "SG", "OPCVM", "S&R"]
else:
    tidy_sel = tidy[tidy["Date"] == rank_date_sel].copy()
    if rank_cls_sel != "TOUS":
        tidy_sel = tidy_sel[tidy_sel["Bloc"] == rank_cls_sel]
    rank_src = (
        tidy_sel
        .groupby(["Bloc", "SG"], as_index=False)["SR"]
        .sum(min_count=1)
        .rename(columns={"SR": "S&R"})
        .sort_values("S&R", ascending=False)
        .reset_index(drop=True)
    )
    tmp = rank_src.merge(ytd_map, on=["Bloc", "SG"], how="left")
    show_cols = ["Bloc", "SG", "S&R", "YTD"]

if tmp.empty:
    st.info("Aucune donnée de classement pour ce filtre.")
else:
    st.markdown(html_table(tmp[show_cols].head(100).reset_index(drop=True), show_cols, max_h=520), unsafe_allow_html=True)
    export_button(tmp[show_cols].head(100), f"classement_fonds_{rank_date_sel.replace('/','')}.csv")

st.markdown("---")

# ══════════════════════════════════════════════
# ── ANALYSE IA PAR FONDS BMCE ──
# ══════════════════════════════════════════════
section_header("Analyse IA — Fonds BMCE Capital Gestion", "Analyse narrative par fonds · propulsée par LLaMA 3.3 70B")

if not asfim_sr.empty:
    _bmce_sg_name = pick_bmce_sg(asfim_sr["SG"].dropna().unique().tolist()) or ""
    _bmce_fonds_df = asfim_sr[asfim_sr["SG"] == _bmce_sg_name].copy() if _bmce_sg_name else pd.DataFrame()

    if _bmce_fonds_df.empty:
        st.info("Aucun fonds BMCE Capital Gestion trouvé dans les données ASFIM.")
    else:
        _fonds_list = sorted(_bmce_fonds_df["OPCVM"].dropna().unique().tolist())
        _ia_f1, _ia_f2 = st.columns([2, 1])
        with _ia_f1:
            _fonds_sel = st.selectbox("Sélectionner un fonds", _fonds_list, key="ia_fonds_sel")
        with _ia_f2:
            _ia_periode = st.selectbox("Période d'analyse", ["Toutes les dates", "4 dernières semaines", "8 dernières semaines"], key="ia_periode")

        if st.button("🤖 Analyser ce fonds", key="btn_ia_fonds"):
            # ── Historique du fonds ──
            _fonds_hist = (
                _bmce_fonds_df[_bmce_fonds_df["OPCVM"] == _fonds_sel]
                [["Date", "Classification", "SR", "YTD"]]
                .sort_values("Date", key=lambda s: s.map(mmdd_sort))
                .reset_index(drop=True)
            )
            if _ia_periode == "4 dernières semaines":
                _fonds_hist = _fonds_hist.tail(4)
            elif _ia_periode == "8 dernières semaines":
                _fonds_hist = _fonds_hist.tail(8)

            _fonds_cat = _fonds_hist["Classification"].iloc[-1] if not _fonds_hist.empty else "—"
            _fonds_ytd = _fonds_hist["YTD"].iloc[-1] if not _fonds_hist.empty else np.nan
            _fonds_last_sr = _fonds_hist["SR"].iloc[-1] if not _fonds_hist.empty else np.nan
            _fonds_mean_sr = _fonds_hist["SR"].mean()
            _fonds_total_sr = _fonds_hist["SR"].sum()

            # Tendance : comparaison 1ère moitié vs 2ème moitié
            _mid = max(1, len(_fonds_hist) // 2)
            _trend_first = _fonds_hist["SR"].iloc[:_mid].mean()
            _trend_second = _fonds_hist["SR"].iloc[_mid:].mean()
            _trend_dir = "en amélioration" if _trend_second > _trend_first else "en dégradation"

            # Pic et creux
            _peak_row = _fonds_hist.loc[_fonds_hist["SR"].idxmax()] if not _fonds_hist.empty else None
            _trough_row = _fonds_hist.loc[_fonds_hist["SR"].idxmin()] if not _fonds_hist.empty else None

            # Semaines positives vs négatives
            _nb_pos_f = (_fonds_hist["SR"] > 0).sum()
            _nb_neg_f = (_fonds_hist["SR"] < 0).sum()

            # Stats supplémentaires
            _fonds_median_sr = _fonds_hist["SR"].median()
            _fonds_std_sr    = _fonds_hist["SR"].std() if len(_fonds_hist) > 1 else 0.0
            _outlier_mask    = (_fonds_hist["SR"] - _fonds_mean_sr).abs() > _fonds_std_sr if _fonds_std_sr > 0 else pd.Series([False]*len(_fonds_hist))
            _nb_outliers     = int(_outlier_mask.sum())
            _outliers_total_sr = _fonds_hist.loc[_outlier_mask, "SR"].sum()
            _trend_var       = (_trend_second - _trend_first) / abs(_trend_first) if _trend_first and abs(_trend_first) > 1e-9 else np.nan
            # Tendance 2ème moitié hors outliers
            _second_half     = _fonds_hist.iloc[_mid:].copy()
            _second_no_out   = _second_half[~_outlier_mask.iloc[_mid:].values]
            _trend_second_no_outliers = _second_no_out["SR"].mean() if not _second_no_out.empty else _trend_second
            # Flux semaines adjacentes au pic
            _peak_idx = _fonds_hist["SR"].idxmax()
            _pre_peak_sr  = _fonds_hist.loc[_peak_idx - 1, "SR"] if _peak_idx > 0 else np.nan
            _post_peak_sr = _fonds_hist.loc[_peak_idx + 1, "SR"] if _peak_idx < len(_fonds_hist) - 1 else np.nan
            if abs(_trend_first) < 10_000:
                _trend_var_label = f"{fmt_money(_trend_second - _trend_first)} (variation absolue — base 1ère moitié quasi-nulle)"
            elif pd.notna(_trend_var):
                _trend_var_label = f"{fmt_pct(_trend_var)} ({'+' if _trend_second > _trend_first else ''}{fmt_money(_trend_second - _trend_first)})"
            else:
                _trend_var_label = "non calculable"

            # Comparaison catégorie sur la même période
            _cat_hist = asfim_sr[
                (asfim_sr["Classification"] == _fonds_cat) &
                (asfim_sr["Date"].isin(_fonds_hist["Date"]))
            ].groupby("Date")["SR"].sum().reset_index()
            _cat_mean  = _cat_hist["SR"].mean() if not _cat_hist.empty else np.nan
            _cat_total = _cat_hist["SR"].sum() if not _cat_hist.empty else np.nan

            # Actif Net du fonds (dernière valeur disponible)
            _fonds_an = np.nan
            _cat_an_total = np.nan
            if "AN" in _bmce_fonds_df.columns:
                _an_series = _bmce_fonds_df[_bmce_fonds_df["OPCVM"] == _fonds_sel]["AN"].dropna()
                _fonds_an = _an_series.iloc[-1] if not _an_series.empty else np.nan
                _cat_an_df = asfim_sr[(asfim_sr["Classification"] == _fonds_cat) & asfim_sr["AN"].notna()]
                if not _cat_an_df.empty:
                    _last_an_date = _cat_an_df.sort_values("Date", key=lambda s: s.map(mmdd_sort))["Date"].iloc[-1]
                    _cat_an_total = _cat_an_df[_cat_an_df["Date"] == _last_an_date].groupby("OPCVM")["AN"].last().sum()

            def _an(v):
                return fmt_money(v) if pd.notna(v) else "non disponible"

            # Rang du fonds dans sa catégorie à la dernière date
            _last_date_f = _fonds_hist["Date"].iloc[-1] if not _fonds_hist.empty else None
            if _last_date_f:
                _cat_rank_df = (
                    asfim_sr[(asfim_sr["Classification"] == _fonds_cat) & (asfim_sr["Date"] == _last_date_f)]
                    .sort_values("SR", ascending=False)
                    .reset_index(drop=True)
                )
                _fonds_rank_idx = _cat_rank_df[_cat_rank_df["OPCVM"] == _fonds_sel].index
                _fonds_rank = int(_fonds_rank_idx[0] + 1) if len(_fonds_rank_idx) > 0 else None
            else:
                _fonds_rank = None

            # Historique formaté pour le prompt
            _hist_lines = "\n".join(
                f"  - {r.Date} : {fmt_money(r.SR)}"
                for r in _fonds_hist.itertuples()
            )

            _an_disponible     = pd.notna(_fonds_an)    and _fonds_an    != 0
            _cat_an_disponible = pd.notna(_cat_an_total) and _cat_an_total != 0
            _fonds_an_s    = fmt_money(_fonds_an)    if _an_disponible    else "non disponible"
            _cat_an_s      = fmt_money(_cat_an_total) if _cat_an_disponible else "non disponible"
            _rot_moy_s     = fmt_pct(_fonds_mean_sr / _fonds_an)    if _an_disponible else "non disponible"
            _rot_pic_s     = fmt_pct(_peak_row.SR   / _fonds_an)    if _an_disponible else "non disponible"
            _rot_ytd_s     = fmt_pct(_fonds_ytd     / _fonds_an)    if _an_disponible else "non disponible"
            _pdm_enc_s     = fmt_pct(_fonds_an / _cat_an_total)     if _an_disponible and _cat_an_disponible else "non disponible"
            _pdm_flux_s    = fmt_pct(_fonds_mean_sr / _cat_mean if pd.notna(_cat_mean) and _cat_mean != 0 else 0)

            _prompt_fonds = f"""Tu es analyste quantitatif senior spécialisé en fonds OPCVM marocains. Tu dois produire une analyse strictement factuelle, sans hallucination ni remplissage. Chaque affirmation doit être directement étayée par les chiffres fournis.

═══ DONNÉES ═══
Fonds : {_fonds_sel}
Catégorie ASFIM : {_fonds_cat}
Période : {_fonds_hist["Date"].iloc[0] if not _fonds_hist.empty else "—"} → {_fonds_hist["Date"].iloc[-1] if not _fonds_hist.empty else "—"}

Flux hebdomadaires de souscriptions/rachats nets (S&R en MAD) — ATTENTION : ce sont des FLUX, pas des cours ni des encours :
{_hist_lines}

Indicateurs statistiques sur ces flux :
- Dernier flux S&R : {fmt_money(_fonds_last_sr)}
- Flux moyen hebdomadaire : {fmt_money(_fonds_mean_sr)}
- Flux médian hebdomadaire : {fmt_money(_fonds_median_sr)}
- Écart-type des flux : {fmt_money(_fonds_std_sr)}
- Coefficient de variation (écart-type/moyenne) : {fmt_pct(_fonds_std_sr / abs(_fonds_mean_sr) if _fonds_mean_sr != 0 else 0)}
- Flux cumulé sur la période : {fmt_money(_fonds_total_sr)}
- Flux cumulé des semaines outliers uniquement : {fmt_money(_outliers_total_sr)}
- Part des outliers dans le flux cumulé total : {fmt_pct(_outliers_total_sr / _fonds_total_sr if _fonds_total_sr != 0 else 0)}
- YTD 2026 (flux cumulé depuis janvier) : {fmt_money(_fonds_ytd)}
- Nombre de semaines total : {_nb_pos_f + _nb_neg_f}
- Semaines en collecte nette positive : {_nb_pos_f} / {_nb_pos_f + _nb_neg_f} ({fmt_pct(_nb_pos_f / (_nb_pos_f + _nb_neg_f) if (_nb_pos_f + _nb_neg_f) > 0 else 0)})
- Nombre de semaines outliers (écart > 1 écart-type) : {_nb_outliers}
- Tendance 1ère moitié : {fmt_money(_trend_first)}
- Tendance 2ème moitié : {fmt_money(_trend_second)}
- Tendance 2ème moitié hors outliers : {fmt_money(_trend_second_no_outliers)}
- Variation de tendance : {_trend_var_label}
- Pic de collecte : {fmt_money(_peak_row.SR)} (semaine du {_peak_row.Date}) — représente {fmt_pct(_peak_row.SR / _fonds_total_sr if _fonds_total_sr != 0 else 0)} du flux cumulé total
- Flux de la semaine précédant le pic : {fmt_money(_pre_peak_sr)}
- Flux de la semaine suivant le pic : {fmt_money(_post_peak_sr)}
- Creux de rachats : {fmt_money(_trough_row.SR)} (semaine du {_trough_row.Date})
- Ratio creux/pic en valeur absolue : {fmt_pct(abs(_trough_row.SR) / abs(_peak_row.SR) if _peak_row is not None and _peak_row.SR != 0 else 0)}

Encours du fonds (Actif Net au {_fonds_hist["Date"].iloc[-1] if not _fonds_hist.empty else "—"}) :
- Actif Net (AN) = encours total du fonds : {_fonds_an_s}
- Ratio flux moyen / encours (rotation hebdomadaire) : {_rot_moy_s}
- Ratio pic / encours : {_rot_pic_s} — part du pic dans l'encours total
- Ratio flux cumulé YTD / encours : {_rot_ytd_s}

Comparaison catégorie {_fonds_cat} (brute, non normalisée par taille) :
- Flux moyen catégorie : {fmt_money(_cat_mean)}/sem
- Total catégorie : {fmt_money(_cat_total)}
- Ratio fonds/catégorie (flux) : {_pdm_flux_s}
- Actif Net total de la catégorie : {_cat_an_s}
- Part de marché du fonds par encours : {_pdm_enc_s}
- Rang du fonds dans sa catégorie (flux) : {"#" + str(_fonds_rank) + " sur " + str(len(_cat_rank_df)) if _fonds_rank else "non disponible"}

Contexte macro factuel (NE PAS aller au-delà de ces faits) :
- Taux directeur BAM : 2,25% — stable depuis mars 2025, statu quo confirmé lors de la réunion du 17 mars 2026, aucune modification en 2026
- Déficit de liquidité bancaire structurel estimé à 146,8 Mds MAD en 2026
- Croissance économique projetée : 5,6% en 2026 (Bank Al-Maghrib, mars 2026)
- Inflation projetée : 0,8% en 2026

═══ QUESTIONS ANALYTIQUES OBLIGATOIRES ═══
Ces questions doivent être traitées explicitement dans le corps du texte :

Q1. CONCENTRATION DES FLUX : Quelle est la part des {_nb_outliers} semaines outliers dans le flux cumulé total ({fmt_money(_fonds_total_sr)}) ? Ce ratio révèle-t-il un mode de collecte régulier ou par impulsions ponctuelles ? Conclure explicitement.

Q2. MÉDIANE VS MOYENNE : L'écart entre le flux médian ({fmt_money(_fonds_median_sr)}) et la moyenne ({fmt_money(_fonds_mean_sr)}) est-il significatif ? Que dit cet écart sur la distribution réelle des flux semaine par semaine — la majorité des semaines sont-elles proches de zéro ?

Q3. DOUBLE PIC : Comparer séparément :
- La semaine PRÉCÉDANT le pic : {fmt_money(_pre_peak_sr)} vs pic {fmt_money(_peak_row.SR)}
  → Ces deux valeurs sont-elles proches ? Si oui, signal de double saisie.
- La semaine SUIVANT le pic : {fmt_money(_post_peak_sr)} vs pic {fmt_money(_peak_row.SR)}
  → Une chute brutale après le pic confirme-t-elle le caractère isolé de l'opération ?
Traiter ces deux comparaisons séparément et conclure sur la nature probable du pic.

Q4. TENDANCE RÉELLE : La tendance haussière de la 2ème moitié ({fmt_money(_trend_second)}) se confirme-t-elle hors outliers ({fmt_money(_trend_second_no_outliers)}) ? Ou est-elle entièrement portée par les semaines anomales ? Conclure sur la robustesse de cette tendance.

Q5. CREUX RELATIF : Le creux de {fmt_money(_trough_row.SR)} représente {fmt_pct(abs(_trough_row.SR) / abs(_peak_row.SR) if _peak_row is not None and _peak_row.SR != 0 else 0)} du pic en valeur absolue. Est-ce cohérent avec un rachat réel dans un fonds de cette catégorie, ou faut-il suspecter une erreur de saisie ? Argumenter.

Q6. ROTATION PAR ENCOURS : Le ratio flux moyen hebdomadaire / encours est de {_rot_moy_s}. Est-ce un niveau de rotation normal pour un fonds {_fonds_cat} institutionnel ? Le pic représente {_rot_pic_s} de l'encours total — qualifier explicitement le caractère exceptionnel ou non de ce mouvement par rapport à la taille du fonds.

Q7. PART DE MARCHÉ PAR ENCOURS : La part de marché du fonds par encours est de {_pdm_enc_s}, alors que sa part par flux est de {_pdm_flux_s}. Ces deux ratios sont-ils cohérents entre eux ? Un écart important entre part de marché encours et part de marché flux suggère-t-il une sur ou sous-collecte relative à la taille du fonds ?

═══ RÈGLES ABSOLUES ═══
1. FLUX UNIQUEMENT : Ne jamais confondre flux S&R avec cours, valeur liquidative ou performance. L'Actif Net (AN) fourni ci-dessus représente l'encours total du fonds — il ne doit pas être confondu avec les flux S&R qui mesurent uniquement les souscriptions et rachats nets.

2. CAUSALITÉ INTERDITE : Toute hypothèse explicative doit être introduite par un marqueur explicite ("pourrait laisser supposer", "laisse hypothétiquement penser", "hypothèse non vérifiable ici"). Jamais comme un fait établi.

3. DÉFINITION D'UNE ANOMALIE : Une semaine est considérée anomale si son flux s'écarte de plus d'un écart-type ({fmt_money(_fonds_std_sr)}) par rapport à la moyenne ({fmt_money(_fonds_mean_sr)}). Toute semaine anomale doit être nommée, chiffrée et questionnée.

4. VARIATION DE TENDANCE : Si la base de la première moitié est quasi-nulle ou inférieure à 10 000 MAD, ne pas calculer de variation en pourcentage. Exprimer uniquement la variation en valeur absolue.

5. INTERDITS absolus : "les gestionnaires ont pris des décisions judicieuses", "les investisseurs ont pris confiance", "performance solide", "résultats encourageants", "dynamique positive", "vigilance accrue", "décisions éclairées", "suivre de près ces indicateurs", "surveillance étroite", et toute formule générique non étayée par un chiffre.

6. COMPARAISON CATÉGORIE : Distinguer systématiquement la comparaison par flux (brute, non normalisée) de la comparaison par encours (part de marché réelle). La part de marché par encours est la mesure la plus fiable du positionnement relatif du fonds.

7. SIGNAUX NÉGATIFS : Si le fonds présente des signaux défavorables, les formuler clairement, sans atténuation ni euphémisme.

8. NOUVEAUTÉ PAR PARAGRAPHE : Chaque paragraphe introduit un angle d'analyse distinct. Aucune reformulation ni répétition d'un paragraphe à l'autre. La dernière phrase de chaque paragraphe ne doit pas reformuler ce qui vient d'être dit.

9. CHIFFRES OBLIGATOIRES : Chaque phrase analytique contient au moins un chiffre issu des données fournies.

10. §4 ACTIONNABLE : Chaque point suit strictement la structure : [signal observé chiffré] → [risque concret] → [action spécifique réalisable par une équipe de gestion]. Les formulations vagues sont interdites.

11. CREUX SUSPECT : Si le ratio creux/pic en valeur absolue est inférieur à 1%, signaler explicitement que la valeur du creux est anormalement faible en regard du pic, et que cela pourrait indiquer une erreur de saisie plutôt qu'un rachat réel — recommander une vérification des données sources.

12. VOLATILITÉ ≠ TENDANCE DÉFAVORABLE : Ne pas conclure à une tendance globale défavorable sur la seule base d'un coefficient de variation élevé. Distinguer explicitement la volatilité des flux de la direction de la tendance.

13. ENCOURS COMME RÉFÉRENTIEL : Toute qualification d'un flux comme "important", "faible" ou "exceptionnel" doit être rapportée à l'encours du fonds ({_fonds_an_s}), pas uniquement à la moyenne des flux. Un flux de 100M est négligeable sur un fonds de 10Mds, mais exceptionnel sur un fonds de 200M.

Rédige un unique paragraphe de 4 à 5 phrases destiné à un gérant de fonds. Mentionne obligatoirement : le flux cumulé ({fmt_money(_fonds_total_sr)}), la tendance (hausse/baisse), le positionnement vs la catégorie {_fonds_cat} (rang #{_fonds_rank if _fonds_rank else "—"}), et un signal d'attention concret. Chiffres obligatoires dans chaque phrase. Français direct, sans introduction ni conclusion générique."""

            with st.spinner(f"LLaMA 3.3 analyse {_fonds_sel}..."):
                _fonds_analysis = call_gemini(_prompt_fonds)

            # ── Affichage dans le dashboard ──
            st.markdown(f"""
<div style="background:#FAFAFA;border-left:4px solid #C0001A;border-radius:0 6px 6px 0;
            padding:28px 32px;margin:20px 0;">
  <div style="font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.1em;
              color:#C0001A;margin-bottom:16px;">
    Analyse IA · {_fonds_sel} · {_fonds_cat}
  </div>
  {"".join(f'<p style="margin:0 0 16px 0;line-height:1.85;font-size:13.5px;color:#1C1C1C;text-align:justify;">{p.strip()}</p>' for p in _fonds_analysis.split(chr(10)) if p.strip())}
  <div style="font-size:9px;color:#BBBBBB;margin-top:8px;font-style:italic;">
    Généré par LLaMA 3.3 70B via Groq · Usage interne uniquement
  </div>
</div>
""", unsafe_allow_html=True)

            # ── Export PDF optionnel ──
            _fonds_paras_html = "".join(
                f'<p style="margin:0 0 18px 0;line-height:1.85;font-size:13.5px;color:#1C1C1C;text-align:justify;">{p.strip()}</p>'
                for p in _fonds_analysis.split("\n") if p.strip()
            )
            _logo_tag_f = f'<img src="data:image/{logo_ext};base64,{logo_b64}" style="height:48px;object-fit:contain;">' if logo_b64 else '<span style="font-size:16px;font-weight:900;color:#C0001A;">BMCE Capital Gestion</span>'
            _hist_tbl_rows = "".join(
                f'<tr><td style="padding:5px 12px;font-size:11px;border-bottom:1px solid #F0F0F0;">{r.Date}</td>'
                f'<td style="padding:5px 12px;font-size:11px;text-align:right;font-weight:600;color:{"#1A7A4A" if r.SR >= 0 else "#C0001A"};border-bottom:1px solid #F0F0F0;">{fmt_money(r.SR)}</td></tr>'
                for r in _fonds_hist.itertuples()
            )
            _html_fonds = f"""<!DOCTYPE html>
<html lang="fr"><head><meta charset="UTF-8">
<style>
  @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600;700;900&display=swap');
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{font-family:'Montserrat',sans-serif;background:#fff;color:#1C1C1C;}}
  @media print{{.no-print{{display:none!important}}}}
  .header{{background:#1C1C1C;padding:24px 48px;display:flex;align-items:center;justify-content:space-between;}}
  .header-right{{text-align:right;color:#fff;}}
  .header-title{{font-size:17px;font-weight:700;color:#fff;}}
  .header-sub{{font-size:10px;color:rgba(255,255,255,.55);margin-top:3px;}}
  .red-bar{{height:4px;background:#C0001A;}}
  .body{{padding:44px 52px;max-width:820px;margin:0 auto;}}
  .fund-title{{font-size:20px;font-weight:700;margin-bottom:4px;}}
  .fund-meta{{font-size:11px;color:#9A9A9A;text-transform:uppercase;letter-spacing:.08em;margin-bottom:36px;}}
  .section-lbl{{font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.1em;color:#C0001A;margin:28px 0 12px 0;}}
  .divider{{height:1px;background:#E8E8E8;margin:28px 0;}}
  .kpi-row{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:28px;}}
  .kpi{{border:1px solid #E8E8E8;border-top:3px solid #C0001A;border-radius:4px;padding:10px 14px;}}
  .kpi-lbl{{font-size:8px;font-weight:700;text-transform:uppercase;letter-spacing:.09em;color:#9A9A9A;margin-bottom:4px;}}
  .kpi-val{{font-size:15px;font-weight:700;}}
  .green{{color:#1A7A4A;}} .red{{color:#C0001A;}}
  .disclaimer{{font-size:9px;color:#BBBBBB;margin-top:28px;text-align:center;font-style:italic;}}
  .footer{{background:#FAFAFA;border-top:1px solid #E0E0E0;padding:14px 48px;display:flex;justify-content:space-between;margin-top:40px;}}
  .footer-text{{font-size:9px;color:#9A9A9A;}}
  .print-btn{{position:fixed;bottom:24px;right:24px;background:#C0001A;color:#fff;border:none;border-radius:6px;padding:10px 22px;font-family:Montserrat,sans-serif;font-size:12px;font-weight:700;cursor:pointer;}}
</style></head><body>
<div class="header">
  {_logo_tag_f}
  <div class="header-right">
    <div class="header-title">Analyse de fonds — OPCVM Maroc</div>
    <div class="header-sub">Généré le {date.today().strftime('%d/%m/%Y')}</div>
  </div>
</div>
<div class="red-bar"></div>
<div class="body">
  <div class="fund-title">{_fonds_sel}</div>
  <div class="fund-meta">{_fonds_cat} &nbsp;·&nbsp; BMCE Capital Gestion &nbsp;·&nbsp; {_fonds_hist["Date"].iloc[0] if not _fonds_hist.empty else ""} → {_fonds_hist["Date"].iloc[-1] if not _fonds_hist.empty else ""}</div>

  <div class="kpi-row">
    <div class="kpi"><div class="kpi-lbl">Dernier S&R</div><div class="kpi-val {'green' if pd.notna(_fonds_last_sr) and _fonds_last_sr>=0 else 'red'}">{fmt_money(_fonds_last_sr)}</div></div>
    <div class="kpi"><div class="kpi-lbl">YTD 2026</div><div class="kpi-val {'green' if pd.notna(_fonds_ytd) and _fonds_ytd>=0 else 'red'}">{fmt_money(_fonds_ytd)}</div></div>
    <div class="kpi"><div class="kpi-lbl">Semaines collecte</div><div class="kpi-val green">{_nb_pos_f}</div></div>
    <div class="kpi"><div class="kpi-lbl">Semaines décollecte</div><div class="kpi-val red">{_nb_neg_f}</div></div>
  </div>

  <div class="section-lbl">Analyse narrative</div>
  {_fonds_paras_html}

  <div class="divider"></div>
  <div class="section-lbl">Historique S&R</div>
  <table style="width:100%;border-collapse:collapse;">
    <thead><tr>
      <th style="padding:6px 12px;text-align:left;font-size:9px;font-weight:700;text-transform:uppercase;color:#9A9A9A;border-bottom:2px solid #C0001A;">Date</th>
      <th style="padding:6px 12px;text-align:right;font-size:9px;font-weight:700;text-transform:uppercase;color:#9A9A9A;border-bottom:2px solid #C0001A;">S&R</th>
    </tr></thead>
    <tbody>{_hist_tbl_rows}</tbody>
  </table>

  <div class="disclaimer">Analyse générée par LLaMA 3.3 70B (Groq) · Usage interne BMCE Capital Gestion · Ne constitue pas un conseil en investissement.</div>
</div>
<div class="footer">
  <div class="footer-text">BMCE Capital Gestion — Analyse fonds confidentielle</div>
  <div class="footer-text">Généré le {date.today().strftime('%d/%m/%Y')}</div>
</div>
<button class="print-btn no-print" onclick="window.print()">🖨 Exporter PDF</button>
</body></html>"""

            _buf_f = io.BytesIO(_html_fonds.encode("utf-8"))
            st.download_button(
                label="⬇ Exporter cette analyse en PDF",
                data=_buf_f.getvalue(),
                file_name=f"analyse_{_fonds_sel.replace(' ','_')}_{date.today().strftime('%Y%m%d')}.html",
                mime="text/html",
            )
            st.caption("Ouvrez le fichier dans votre navigateur → Ctrl+P → Enregistrer en PDF")
else:
    st.info("Les données ASFIM sont nécessaires pour cette fonctionnalité.")

st.markdown("---")

# ── CLASSEMENT GÉNÉRAL DES SG ──
section_header("Classement général des sociétés de gestion", "Toutes classifications confondues · date sélectionnée")

_sgr_col1, _sgr_col2 = st.columns([1, 1])
with _sgr_col1:
    _all_sg_rank_dates = sorted(tidy["Date"].dropna().unique().tolist(), key=mmdd_sort)
    _EXTRA_WEEKLY = {"24/03"}
    _sg_rank_dates = [d for d in _all_sg_rank_dates if ddmm_to_dt(d, YEAR).weekday() == 4 or d in _EXTRA_WEEKLY]
    if not _sg_rank_dates:
        _sg_rank_dates = _all_sg_rank_dates
    _sg_rank_date_sel = st.selectbox("Date (vendredis)", _sg_rank_dates, index=len(_sg_rank_dates) - 1, key="sg_rank_date_sel")
with _sgr_col2:
    if not asfim_sr.empty and "Classification" in asfim_sr.columns:
        _sgr_cls_opts = ["TOUS"] + sorted(asfim_sr["Classification"].dropna().unique().tolist())
    else:
        _sgr_cls_opts = ["TOUS"] + sorted(tidy["Bloc"].dropna().unique().tolist())
    _sg_rank_cls_sel = st.selectbox("Classification", _sgr_cls_opts, index=0, key="sg_rank_cls_sel")

_sgr_src = tidy[tidy["Date"] == _sg_rank_date_sel].copy()
if _sg_rank_cls_sel != "TOUS":
    _sgr_src = _sgr_src[_sgr_src["Bloc"] == _sg_rank_cls_sel]

sg_rank = (
    _sgr_src
    .groupby("SG", as_index=False)["SR"]
    .sum(min_count=1)
    .rename(columns={"SR": "S&R"})
    .sort_values("S&R", ascending=False)
    .reset_index(drop=True)
)
sg_rank.index += 1
if sg_rank.empty:
    st.info("Aucune donnée SG pour cette date.")
else:
    # Tableau pleine largeur
    st.markdown(html_table(sg_rank[["SG", "S&R"]].reset_index().rename(columns={"index": "#"}), ["#", "SG", "S&R"], max_h=460), unsafe_allow_html=True)
    export_button(sg_rank[["SG", "S&R"]].reset_index().rename(columns={"index": "#"}), f"classement_sg_{_sg_rank_date_sel.replace('/','')}.csv")

    # Graphique évolution du rang pleine largeur
    st.markdown(f'<div style="font-weight:700;color:{PRIMARY};font-size:.85rem;margin:24px 0 8px 0;">Évolution du rang dans le temps</div>', unsafe_allow_html=True)
    _rank_dates = sorted(tidy["Date"].dropna().unique().tolist(), key=mmdd_sort)
    _rank_rows = []
    for _d in _rank_dates:
        _day = (
            tidy[tidy["Date"] == _d]
            .groupby("SG", as_index=False)["SR"].sum(min_count=1)
            .sort_values("SR", ascending=False)
            .reset_index(drop=True)
        )
        _day["Rang"] = _day.index + 1
        _day["Date_dt"] = ddmm_to_dt(_d, YEAR)
        _rank_rows.append(_day[["SG", "Date_dt", "Rang"]])
    _rank_df = pd.concat(_rank_rows, ignore_index=True) if _rank_rows else pd.DataFrame()

    if not _rank_df.empty:
        _all_sg_rank = sorted(_rank_df["SG"].dropna().unique().tolist())
        _bmce_default = pick_bmce_sg(_all_sg_rank)
        _top3_default = [s for s in sg_rank["SG"].head(4).tolist() if s != _bmce_default][:3]
        _default_sel = ([_bmce_default] if _bmce_default else []) + _top3_default
        _sg_rank_sel = st.multiselect(
            "SG à afficher", _all_sg_rank,
            default=[s for s in _default_sel if s in _all_sg_rank],
            key="rank_sg_sel"
        )
        if _sg_rank_sel:
            _rank_top = _rank_df[_rank_df["SG"].isin(_sg_rank_sel)]
            _rank_chart = (
                alt.Chart(_rank_top)
                .mark_line(strokeWidth=2.5, point=alt.OverlayMarkDef(filled=True, size=50))
                .encode(
                    x=alt.X("Date_dt:T", title="Date", axis=alt.Axis(format="%m/%d")),
                    y=alt.Y("Rang:Q", title="Rang",
                            scale=alt.Scale(reverse=True),
                            axis=alt.Axis(tickMinStep=1)),
                    color=alt.Color("SG:N", title=None, scale=alt.Scale(range=PALETTE)),
                    tooltip=[
                        alt.Tooltip("Date_dt:T", title="Date", format="%d/%m/%Y"),
                        alt.Tooltip("SG:N", title="SG"),
                        alt.Tooltip("Rang:Q", title="Rang"),
                    ],
                )
                .properties(title="Évolution du rang", height=400, background=BG)
            )
            st.altair_chart(_rank_chart, use_container_width=True)
        else:
            st.info("Sélectionne au moins une SG.")

st.markdown("---")

# ── TENDANCES ──
section_header("Tendances de marché", "Évolution des flux nets S&R sur la période")

# Catégories dynamiques : toutes les classifications trouvées dans asfim_sr ou tidy
if not asfim_sr.empty and "Classification" in asfim_sr.columns:
    _all_cats = sorted(asfim_sr["Classification"].dropna().unique().tolist())
else:
    _all_cats = sorted(tidy["Bloc"].dropna().unique().tolist())

_tab_labels = ["Global"] + _all_cats
tabs = st.tabs(_tab_labels)

# Onglet 0 — Global
with tabs[0]:
    if tcd_totals:
        g = pd.DataFrame([{"Date": d, "SR": v} for d, v in tcd_totals.items() if pd.notna(v)])
        if not g.empty:
            g["k"] = g["Date"].apply(mmdd_sort)
            g = g.sort_values("k").drop(columns=["k"])
            g["Date_dt"] = g["Date"].apply(lambda s: ddmm_to_dt(s, YEAR))
        else:
            g = tidy.groupby("Date_dt", as_index=False)["SR"].sum().sort_values("Date_dt")
    else:
        g = tidy.groupby("Date_dt", as_index=False)["SR"].sum().sort_values("Date_dt")
    st.altair_chart(line_chart(g, "Date_dt", "SR", title="Collecte nette globale S&R"), use_container_width=True)

# Onglets dynamiques — une tab par catégorie
for _i, _cat in enumerate(_all_cats):
    with tabs[_i + 1]:
        # Chercher dans asfim_sr d'abord (données par fonds), sinon tidy (par SG/Bloc)
        if not asfim_sr.empty and "Classification" in asfim_sr.columns:
            _df_cat = (
                asfim_sr[asfim_sr["Classification"] == _cat]
                .copy()
            )
            if not _df_cat.empty:
                _df_cat["Date_dt"] = _df_cat["Date"].apply(lambda s: ddmm_to_dt(s, YEAR))
                _d = _df_cat.groupby("Date_dt", as_index=False)["SR"].sum().sort_values("Date_dt")
            else:
                _d = pd.DataFrame(columns=["Date_dt", "SR"])
        else:
            _d = tidy[tidy["Bloc"] == _cat].groupby("Date_dt", as_index=False)["SR"].sum().sort_values("Date_dt")

        if _d.empty:
            st.info(f"Aucune donnée pour {_cat}.")
        else:
            st.altair_chart(line_chart(_d, "Date_dt", "SR", title=f"{_cat} — S&R"), use_container_width=True)

st.markdown("---")

# ── DRILL-DOWN ──
section_header("Analyse par société de gestion", "Détail des flux et positionnement concurrentiel")

if not asfim_cls_map.empty:
    cls_values = sorted(asfim_cls_map["Classification"].dropna().unique().tolist())
else:
    cls_values = ["ACTIONS", "DIVERSIFIES", "OMLT"]

# Ligne 1 : Classification (1/3) + SG (2/3)
_fa, _fb = st.columns([1, 2])
with _fa:
    drill_cat = st.selectbox("Classification", ["TOUS"] + cls_values, index=0)
with _fb:
    if drill_cat == "TOUS":
        allowed_sg = set(tidy_all["SG"].dropna().unique().tolist())
    elif not asfim_cls_map.empty:
        allowed_sg = set(asfim_cls_map.loc[asfim_cls_map["Classification"] == drill_cat, "SG"].dropna().unique().tolist())
    else:
        allowed_sg = set(tidy_all[tidy_all["Bloc"] == drill_cat]["SG"].dropna().unique().tolist())
    sg_list = sorted([sg for sg in tidy_all["SG"].dropna().unique().tolist() if sg in allowed_sg])
    if not sg_list:
        st.warning("Aucune SG.")
        st.stop()
    sg_pick = st.selectbox("Société de gestion", sg_list, index=0)

# Ligne 2 : Date (petite colonne à gauche)
_fd, _ = st.columns([1, 2])
with _fd:
    _all_dates_sg = sorted(tidy_all["Date"].dropna().unique().tolist(), key=mmdd_sort)
    _sg_date_sel = st.selectbox("Date", _all_dates_sg, index=len(_all_dates_sg)-1, key="sg_date_sel")

# Source : ASFIM (par fonds) agrégé à la SG
if not asfim_sr.empty and sg_pick in asfim_sr["SG"].values:
    sg_src = asfim_sr[asfim_sr["SG"] == sg_pick].copy()
    # Filtrer par classification si sélectionnée
    if drill_cat != "TOUS":
        sg_src = sg_src[sg_src["Classification"] == drill_cat]
    sg_tidy = (
        sg_src.groupby(["Date", "Classification"], as_index=False)["SR"]
        .sum(min_count=1)
        .rename(columns={"Classification": "Bloc"})
    )
    sg_tidy["Date_dt"] = sg_tidy["Date"].apply(lambda s: ddmm_to_dt(s, YEAR))
    # YTD : par classification si filtrée, sinon ALL OPCVM global
    if drill_cat != "TOUS":
        _ytd_sg = sg_src.groupby("OPCVM")["YTD"].max().sum(skipna=True) if "YTD" in sg_src.columns else np.nan
        if pd.isna(_ytd_sg):
            _ytd_sg = tidy_all[(tidy_all["SG"] == sg_pick) & (tidy_all["Bloc"] == drill_cat)]["YTD_row"].max()
    else:
        _ytd_sg = all_opcvm_ytd.get(sg_pick, np.nan)
        if pd.isna(_ytd_sg):
            _ytd_sg = tidy_all[tidy_all["SG"] == sg_pick].groupby("Bloc")["YTD_row"].max().sum(skipna=True)
else:
    sg_tidy = tidy_all[tidy_all["SG"] == sg_pick].copy()
    if drill_cat != "TOUS":
        sg_tidy = sg_tidy[sg_tidy["Bloc"] == drill_cat]
    sg_tidy["Date_dt"] = sg_tidy["Date"].apply(lambda s: ddmm_to_dt(s, YEAR))
    if drill_cat != "TOUS":
        _ytd_sg = sg_tidy["YTD_row"].max()
    else:
        _ytd_sg = all_opcvm_ytd.get(sg_pick, np.nan)
        if pd.isna(_ytd_sg):
            _ytd_sg = sg_tidy.groupby("Bloc")["YTD_row"].max().sum(skipna=True)

# S&R pour la date sélectionnée
_sg_day       = sg_tidy[sg_tidy["Date"] == _sg_date_sel]
_sg_day_grp   = _sg_day.groupby("Bloc", as_index=False)["SR"].sum(min_count=1).sort_values("SR", ascending=False)
_sg_day_total = _sg_day["SR"].sum(skipna=True)

# Part de marché : filtrée par classification si applicable
if drill_cat != "TOUS":
    _bloc_key = drill_cat
    _mkt_total_day = tidy_all[(tidy_all["Date"] == _sg_date_sel) & (tidy_all["Bloc"] == _bloc_key)]["SR"].sum(skipna=True)
else:
    _mkt_total_day = tidy_all[tidy_all["Date"] == _sg_date_sel]["SR"].sum(skipna=True)
_sg_mkt_share = (_sg_day_total / _mkt_total_day) if abs(_mkt_total_day) > 1e-9 else np.nan

# Nb fonds en collecte vs décollecte (filtré par classification si applicable)
if not asfim_sr.empty and sg_pick in asfim_sr["SG"].values:
    _fonds_day = asfim_sr[(asfim_sr["SG"] == sg_pick) & (asfim_sr["Date"] == _sg_date_sel)]
    if drill_cat != "TOUS":
        _fonds_day = _fonds_day[_fonds_day["Classification"] == drill_cat]
    _fonds_collecte   = _fonds_day[_fonds_day["SR"] > 0]["OPCVM"].dropna().tolist()
    _fonds_decollecte = _fonds_day[_fonds_day["SR"] < 0]["OPCVM"].dropna().tolist()
    _nb_collecte   = len(_fonds_collecte)
    _nb_decollecte = len(_fonds_decollecte)
else:
    _fonds_collecte = _fonds_decollecte = []
    _nb_collecte = _nb_decollecte = 0

# KPIs
_k1, _k2, _k3, _k4, _k5 = st.columns(5)
_k1.markdown(_kpi("YTD", fmt_money(_ytd_sg), GREEN if (pd.notna(_ytd_sg) and _ytd_sg >= 0) else RED), unsafe_allow_html=True)
_k2.markdown(_kpi(f"S&R {_sg_date_sel}", fmt_money(_sg_day_total), GREEN if _sg_day_total >= 0 else RED), unsafe_allow_html=True)
_k3.markdown(_kpi("Part de marché", fmt_pct(_sg_mkt_share), ACCENT), unsafe_allow_html=True)
_k4.markdown(_kpi("Fonds en collecte", str(_nb_collecte), GREEN), unsafe_allow_html=True)
_k5.markdown(_kpi("Fonds en décollecte", str(_nb_decollecte), RED), unsafe_allow_html=True)

st.markdown("<div style='margin-top:20px'></div>", unsafe_allow_html=True)

# Tableau pleine largeur
st.markdown(f'<div style="font-weight:700;color:{PRIMARY};font-size:.85rem;margin-bottom:8px;">Par fonds — {_sg_date_sel}</div>', unsafe_allow_html=True)
if not asfim_sr.empty and sg_pick in asfim_sr["SG"].values and not _fonds_day.empty:
    _fonds_tbl = _fonds_day[["OPCVM", "Classification", "SR"]].copy()
    _fonds_tbl = _fonds_tbl.rename(columns={"OPCVM": "Fonds", "SR": "S&R"})
    _fonds_tbl = _fonds_tbl.sort_values("S&R", ascending=False).reset_index(drop=True)
    st.markdown(html_table(_fonds_tbl, ["Fonds", "Classification", "S&R"], max_h=400), unsafe_allow_html=True)
    export_button(_fonds_tbl, f"analyse_{sg_pick.replace(' ','_')}_{_sg_date_sel.replace('/','')}.csv")
elif not _sg_day_grp.empty:
    _sg_day_grp = _sg_day_grp.rename(columns={"Bloc": "Classification", "SR": "S&R"})
    st.markdown(html_table(_sg_day_grp.reset_index(drop=True), ["Classification", "S&R"], max_h=400), unsafe_allow_html=True)
    export_button(_sg_day_grp, f"analyse_{sg_pick.replace(' ','_')}_{_sg_date_sel.replace('/','')}.csv")
else:
    st.info("Aucune donnée.")

# Graphique pleine largeur
st.markdown(f'<div style="font-weight:700;color:{PRIMARY};font-size:.85rem;margin:24px 0 8px 0;">Évolution S&R par catégorie</div>', unsafe_allow_html=True)
sg_cd = sg_tidy.groupby(["Date_dt", "Bloc"], as_index=False)["SR"].sum()
if not sg_cd.empty:
    st.altair_chart(line_chart(sg_cd, "Date_dt", "SR", color="Bloc", title=f"{sg_pick}"), use_container_width=True)


# ══════════════════════════════════════════════
# ── RAPPORT PDF ──
# ══════════════════════════════════════════════
section_header("Rapport synthétique", "Résumé des indicateurs clés · exportable en PDF")

if st.button("📄 Générer le rapport"):
    _rpt_date = date_sel
    _rpt_bmce = pick_bmce_sg(tidy_all["SG"].dropna().unique().tolist()) or ""
    _rpt_ytd  = all_opcvm_ytd.get(_rpt_bmce, np.nan)

    # Stats pour le prompt
    _rpt_sg = (
        tidy_all[tidy_all["Date"] == _rpt_date]
        .groupby("SG", as_index=False)["SR"].sum(min_count=1)
        .sort_values("SR", ascending=False)
        .reset_index(drop=True)
    )
    _rpt_top5  = _rpt_sg.head(5)
    _rpt_flop5 = _rpt_sg.tail(5).iloc[::-1].reset_index(drop=True)
    _rpt_bmce_sr   = _rpt_sg[_rpt_sg["SG"] == _rpt_bmce]["SR"].sum() if _rpt_bmce else np.nan
    _rpt_bmce_rank = int(_rpt_sg[_rpt_sg["SG"] == _rpt_bmce].index[0] + 1) if _rpt_bmce and _rpt_bmce in _rpt_sg["SG"].values else "—"
    _rpt_mkt_share = (_rpt_bmce_sr / _rpt_sg["SR"].sum()) if abs(_rpt_sg["SR"].sum()) > 1e-9 else np.nan
    _top3_txt  = ", ".join([f"{r.SG} ({fmt_money(r.SR)})" for r in _rpt_top5.head(3).itertuples()])
    _flop3_txt = ", ".join([f"{r.SG} ({fmt_money(r.SR)})" for r in _rpt_flop5.head(3).itertuples()])

    # ── Prompt Gemini ──
    _gemini_prompt = f"""Tu es analyste financier senior chez BMCE Capital Gestion, société de gestion d'actifs marocaine.

Rédige un rapport analytique narratif (5 paragraphes) sur les flux de souscriptions et rachats (S&R) OPCVM au Maroc pour la semaine se terminant le {_rpt_date}.

Données factuelles à utiliser :
- Collecte nette totale marché : {fmt_money(total_sel)}
- Sociétés de gestion actives : {nb_sg} ({nb_pos} en collecte, {nb_neg} en décollecte)
- Concentration Top 3 SG : {fmt_pct(_conc_top3)}
- Meilleurs collecteurs : {_top3_txt}
- Plus forts rachats : {_flop3_txt}
- BMCE Capital Gestion — S&R période : {fmt_money(_rpt_bmce_sr)} | Rang : {_rpt_bmce_rank}/{len(_rpt_sg)} | YTD 2026 : {fmt_money(_rpt_ytd)} | Part de marché : {fmt_pct(_rpt_mkt_share)}

Rédige un unique paragraphe de 4 à 5 phrases destiné à un gérant. Couvre obligatoirement : collecte nette totale ({fmt_money(total_sel)}), les 2 meilleurs collecteurs avec leurs montants, le positionnement de BMCE Capital Gestion (rang {_rpt_bmce_rank}, S&R {fmt_money(_rpt_bmce_sr)}, part de marché {fmt_pct(_rpt_mkt_share)}), et un point de vigilance pour la semaine suivante. Français direct, chiffres intégrés dans le texte, sans titre ni introduction."""

    with st.spinner("LLaMA 3.3 rédige le rapport..."):
        _gemini_raw = call_gemini(_gemini_prompt)

    # ── Affichage direct dans le dashboard ──
    _paras = [p.strip() for p in _gemini_raw.split("\n") if p.strip()]
    _gemini_html_paras = "".join(
        f'<p style="margin:0 0 18px 0;line-height:1.85;font-size:13.5px;color:#1C1C1C;text-align:justify;">{p}</p>'
        for p in _paras
    )

    st.markdown(f"""
<div style="background:#FAFAFA;border-left:4px solid #C0001A;border-radius:0 6px 6px 0;padding:32px 36px;margin:24px 0;">
  <div style="font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.1em;color:#C0001A;margin-bottom:20px;">
    Rapport synthétique S&R · {_rpt_date} · BMCE Capital Gestion
  </div>
  {_gemini_html_paras}
  <div style="font-size:9px;color:#BBBBBB;margin-top:12px;font-style:italic;">
    Généré par LLaMA 3.3 70B via Groq · Usage interne uniquement
  </div>
</div>
""", unsafe_allow_html=True)

    _logo_tag = f'<img src="data:image/{logo_ext};base64,{logo_b64}" style="height:52px;object-fit:contain;">' if logo_b64 else '<span style="font-size:18px;font-weight:900;color:#C0001A;">BMCE Capital Gestion</span>'

    html_report = f"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<style>
  @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600;700;900&display=swap');
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{font-family:'Montserrat',sans-serif;background:#fff;color:#1C1C1C;}}
  @media print{{
    .no-print{{display:none!important}}
    body{{padding:0}}
  }}
  .header{{background:#1C1C1C;padding:28px 48px;display:flex;align-items:center;justify-content:space-between;}}
  .header-right{{text-align:right;color:#fff;}}
  .header-title{{font-size:20px;font-weight:700;color:#fff;letter-spacing:.02em;}}
  .header-sub{{font-size:11px;color:rgba(255,255,255,.55);margin-top:4px;}}
  .red-bar{{height:4px;background:#C0001A;}}
  .body{{padding:48px 56px;max-width:860px;margin:0 auto;}}
  .report-title{{font-size:22px;font-weight:700;color:#1C1C1C;margin-bottom:6px;}}
  .report-date{{font-size:12px;color:#9A9A9A;margin-bottom:40px;text-transform:uppercase;letter-spacing:.08em;}}
  .divider{{height:1px;background:#E0E0E0;margin:32px 0;}}
  .footer{{background:#FAFAFA;border-top:1px solid #E0E0E0;padding:16px 48px;display:flex;justify-content:space-between;align-items:center;margin-top:48px;}}
  .footer-text{{font-size:9px;color:#9A9A9A;}}
  .disclaimer{{font-size:9px;color:#BBBBBB;margin-top:28px;text-align:center;font-style:italic;}}
  .print-btn{{position:fixed;bottom:28px;right:28px;background:#C0001A;color:#fff;border:none;border-radius:6px;padding:12px 24px;font-family:Montserrat,sans-serif;font-size:13px;font-weight:700;cursor:pointer;box-shadow:0 4px 16px rgba(192,0,26,.3);z-index:999;}}
  .print-btn:hover{{background:#960014;}}
</style>
</head>
<body>

<div class="header">
  {_logo_tag}
  <div class="header-right">
    <div class="header-title">Rapport S&amp;R — OPCVM Maroc</div>
    <div class="header-sub">Généré le {date.today().strftime('%d/%m/%Y')}</div>
  </div>
</div>
<div class="red-bar"></div>

<div class="body">
  <div class="report-title">Analyse des flux de souscriptions &amp; rachats</div>
  <div class="report-date">Semaine au {_rpt_date} &nbsp;·&nbsp; ASFIM &nbsp;·&nbsp; BMCE Capital Gestion</div>

  {_gemini_html_paras}

  <div class="divider"></div>
  <div class="disclaimer">Ce rapport a été généré automatiquement par intelligence artificielle (LLaMA 3.3 70B via Groq) à partir des données ASFIM. Il est destiné à un usage interne et ne constitue pas un conseil en investissement. BMCE Capital Gestion.</div>
</div>

<div class="footer">
  <div class="footer-text">BMCE Capital Gestion — Rapport S&amp;R confidentiel</div>
  <div class="footer-text">Au {_rpt_date} · Généré le {date.today().strftime('%d/%m/%Y')}</div>
</div>

<button class="print-btn no-print" onclick="window.print()">🖨 Imprimer / Exporter PDF</button>

</body>
</html>"""

    buf = io.BytesIO(html_report.encode("utf-8"))
    st.download_button(
        label="⬇ Exporter ce rapport en PDF",
        data=buf.getvalue(),
        file_name=f"rapport_sr_{_rpt_date.replace('/','')}.html",
        mime="text/html",
    )
    st.caption("Ouvrez le fichier dans votre navigateur → Ctrl+P → Enregistrer en PDF")

if show_audit:
    st.markdown("---")
    section_header("Data (audit)", "Données brutes extraites pour vérification")
    audit = tidy.drop(columns=[c for c in ["Date_dt"] if c in tidy.columns]).reset_index(drop=True)
    acols = list(audit.columns)

    hdr = "".join(
        f'<th style="text-align:left;padding:8px 12px;font-weight:600;font-size:0.72rem;color:{MUTED};letter-spacing:0.05em;text-transform:uppercase;border-bottom:2px solid {BORDER};background:{SURFACE};position:sticky;top:0;z-index:1;">{c}</th>'
        for c in acols
    )

    bdy = ""
    for i in range(len(audit)):
        r = audit.iloc[i]
        row_bg = BG if i % 2 == 0 else SURFACE
        cells = "".join(
            f'<td style="text-align:left;padding:7px 12px;color:{PRIMARY};font-size:0.82rem;border-bottom:1px solid {BORDER};white-space:nowrap;">{"—" if pd.isna(r[c]) else str(r[c])}</td>'
            for c in acols
        )
        bdy += f'<tr style="background:{row_bg};">{cells}</tr>'

    st.markdown(
        f'<div style="border:1px solid {BORDER};border-radius:14px;overflow:hidden;">'
        f'<div style="max-height:560px;overflow-y:auto;">'
        f'<table style="width:100%;border-collapse:collapse;background:{BG};font-family:\'DM Sans\',sans-serif;">'
        f'<thead><tr>{hdr}</tr></thead><tbody>{bdy}</tbody></table></div></div>',
        unsafe_allow_html=True,
    )
