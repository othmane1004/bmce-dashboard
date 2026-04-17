import re
import io
from pathlib import Path
from datetime import date

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import altair as alt

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
PALETTE    = ["#C0001A", "#1A7A4A", "#1C52C8", "#B8870A", "#6D28D9",
              "#0891B2", "#D97706", "#065F46", "#7C3AED", "#0B7A50"]

_T = dict(P=PRIMARY, S=SECONDARY, M=MUTED, B=BG, SF=SURFACE,
          BD=BORDER, AC=ACCENT, AC2=ACCENT2, GR=GREEN, RD=RED)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

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
    """Read per-fund (OPCVM) S&R data from ASFIM sheet."""
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

    # S&R columns: those whose name starts with "S&R" and contain a date
    sr_cols = [c for c in df.columns if str(c).strip().lower().startswith("s&r") and norm_ddmm(str(c))]

    if not sr_cols:
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
        for c in sr_cols:
            date = norm_ddmm(str(c))
            if date:
                rows.append({
                    "OPCVM": opcvm,
                    "SG": sg,
                    "Classification": cls,
                    "Date": date,
                    "SR": to_float(row[c]),
                })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    # YTD per fund = sum of all S&R
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

with st.sidebar:
    st.markdown(f"<div style='font-weight:700;color:{PRIMARY};font-size:1rem;'>Paramètres</div>", unsafe_allow_html=True)
    st.markdown("---")
    up = st.file_uploader("Uploader Analyse_SR.xlsx", type=["xlsx"])
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

if up is None:
    st.markdown(
        f'<div style="margin:40px auto;max-width:480px;text-align:center;padding:48px 32px;'
        f'border:1px solid {BORDER};border-radius:8px;background:{SURFACE};">'
        f'<div style="font-size:2rem;margin-bottom:16px;">📂</div>'
        f'<div style="font-size:1rem;font-weight:700;color:{PRIMARY};margin-bottom:8px;">Importez votre fichier Excel</div>'
        f'<div style="font-size:.85rem;color:{MUTED};">Glissez <strong>Analyse_SR.xlsx</strong> dans le panneau latéral pour afficher le dashboard.</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.stop()

xlsx_bytes = up.getvalue()
xls = pd.ExcelFile(io.BytesIO(xlsx_bytes), engine="openpyxl")
tcd_totals = read_tcd_daily_totals(xlsx_bytes)
tcd_sg_daily = read_tcd_sg_daily(xlsx_bytes)
asfim_cls_map = read_asfim_sg_classification(xlsx_bytes)
asfim_sr = read_asfim_sr(xlsx_bytes)

if not asfim_cls_map.empty:
    class_options = sorted(asfim_cls_map["Classification"].dropna().unique().tolist())
    class_filter = class_options  # toutes les classifications actives par défaut
    allowed_sg_sidebar = set(
        asfim_cls_map["SG"].dropna().unique().tolist()
    )
else:
    class_options = ["ACTIONS", "DIVERSIFIES", "OMLT"]
    class_filter = class_options
    allowed_sg_sidebar = None

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

df_recap = read_recap(xlsx_bytes, recap_name)
tidy = parse_recap_sr(df_recap)
tidy = apply_weekly_bridge_fix(tidy)
all_opcvm_ytd = parse_all_opcvm_ytd(df_recap)  # {SG: total_YTD} from aggregate section


if tidy.empty:
    st.error("Impossible d'extraire les données.")
    st.stop()

# Keep an unfiltered copy for drill-down category filtering.
tidy_all = tidy.copy()

YEAR = date.today().year
tidy["Date_dt"] = tidy["Date"].apply(lambda s: ddmm_to_dt(s, YEAR))

if allowed_sg_sidebar is not None:
    tidy = tidy[tidy["SG"].isin(allowed_sg_sidebar)]
else:
    tidy = tidy[tidy["Bloc"].isin(class_filter)]

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

st.markdown(
    f'<div style="background:{SURFACE};border:1px solid {BORDER};border-radius:10px;padding:10px 14px;margin:10px 0 16px 0;font-size:0.86rem;color:{SECONDARY};">'
    f'<strong style="color:{PRIMARY};">Note :</strong> Le YTD provient directement de l\'Excel.</div>',
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
        f'<div style="font-size:.65rem!important;font-weight:700!important;text-transform:uppercase;letter-spacing:.09em;'
        f'color:{MUTED}!important;-webkit-text-fill-color:{MUTED}!important;margin-bottom:6px;">{label}</div>'
        f'<div style="font-size:1.25rem!important;font-weight:700!important;'
        f'color:{clr}!important;-webkit-text-fill-color:{clr}!important;'
        f'line-height:1.1;word-break:break-word;">{value}</div>'
        f'</div>'
    )

k1, k2, k3, k4, k5 = st.columns(5)
k1.markdown(_kpi("Date", date_sel, ACCENT), unsafe_allow_html=True)
k2.markdown(_kpi("Collecte nette S&R", fmt_money(total_sel)), unsafe_allow_html=True)
k3.markdown(_kpi("Nb SG actives", nb_sg), unsafe_allow_html=True)
k4.markdown(_kpi("SG en collecte", nb_pos, GREEN), unsafe_allow_html=True)
k5.markdown(_kpi("SG en décollecte", nb_neg, RED), unsafe_allow_html=True)

st.markdown("---")

# ── CLASSEMENT ──
section_header("Classement par fonds", "S&R journalier par OPCVM · date sélectionnée")

# Filter options: classifications from ASFIM, fallback to blocs from tidy
if not asfim_sr.empty and "Classification" in asfim_sr.columns:
    cls_options_rank = ["TOUS"] + sorted(asfim_sr["Classification"].dropna().unique().tolist())
else:
    cls_options_rank = ["TOUS"] + sorted(tidy["Bloc"].dropna().unique().tolist())
rank_cls_sel = st.selectbox("Filtre par classification", cls_options_rank, index=0)

if not asfim_sr.empty:
    # Use ASFIM fund-level data for the ranking
    rank_src = asfim_sr[asfim_sr["Date"] == date_sel].copy()
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
    # Fallback: use tidy (SG-level from Recap S&R)
    tidy_sel = tidy[tidy["Date"] == date_sel].copy()
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
    st.markdown(
        html_table(tmp[show_cols].head(100).reset_index(drop=True), show_cols, max_h=520),
        unsafe_allow_html=True,
    )

st.markdown("---")

# ── CLASSEMENT GÉNÉRAL DES SG ──
section_header("Classement général des sociétés de gestion", "Toutes classifications confondues · date sélectionnée")
sg_rank = (
    tidy[tidy["Date"] == date_sel]
    .groupby("SG", as_index=False)["SR"]
    .sum(min_count=1)
    .rename(columns={"SR": "S&R"})
    .sort_values("S&R", ascending=False)
    .reset_index(drop=True)
)
sg_rank.index += 1  # classement à partir de 1
if sg_rank.empty:
    st.info("Aucune donnée SG pour cette date.")
else:
    st.markdown(
        html_table(sg_rank[["SG", "S&R"]].reset_index().rename(columns={"index": "#"}), ["#", "SG", "S&R"], max_h=520),
        unsafe_allow_html=True,
    )

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

# Source : ASFIM (par fonds) agrégé à la SG — plus fiable que Recap S&R
if not asfim_sr.empty and sg_pick in asfim_sr["SG"].values:
    sg_src = asfim_sr[asfim_sr["SG"] == sg_pick].copy()
    sg_tidy = (
        sg_src.groupby(["Date", "Classification"], as_index=False)["SR"]
        .sum(min_count=1)
        .rename(columns={"Classification": "Bloc"})
    )
    sg_tidy["Date_dt"] = sg_tidy["Date"].apply(lambda s: ddmm_to_dt(s, YEAR))
    # YTD = section ALL OPCVM (agrégat total multi-catégories) → fallback somme par bloc
    _ytd_sg = all_opcvm_ytd.get(sg_pick, np.nan)
    if pd.isna(_ytd_sg):
        _ytd_sg = tidy_all[tidy_all["SG"] == sg_pick].groupby("Bloc")["YTD_row"].max().sum(skipna=True)
else:
    sg_tidy = tidy_all[tidy_all["SG"] == sg_pick].copy()
    sg_tidy["Date_dt"] = sg_tidy["Date"].apply(lambda s: ddmm_to_dt(s, YEAR))
    _ytd_sg = all_opcvm_ytd.get(sg_pick, np.nan)
    if pd.isna(_ytd_sg):
        _ytd_sg = sg_tidy.groupby("Bloc")["YTD_row"].max().sum(skipna=True)

# S&R pour la date sélectionnée
_sg_day      = sg_tidy[sg_tidy["Date"] == _sg_date_sel]
_sg_day_grp  = _sg_day.groupby("Bloc", as_index=False)["SR"].sum(min_count=1).sort_values("SR", ascending=False)
_sg_day_total = _sg_day["SR"].sum(skipna=True)

# KPIs
_k1, _k2 = st.columns(2)
_k1.markdown(_kpi("YTD", fmt_money(_ytd_sg), GREEN if (pd.notna(_ytd_sg) and _ytd_sg >= 0) else RED), unsafe_allow_html=True)
_k2.markdown(_kpi(f"S&R {_sg_date_sel}", fmt_money(_sg_day_total), GREEN if _sg_day_total >= 0 else RED), unsafe_allow_html=True)

st.markdown("<div style='margin-top:20px'></div>", unsafe_allow_html=True)

# Tableau (gauche) + Graphique (droite)
_col_tbl, _col_chart = st.columns([1, 2])

with _col_tbl:
    st.markdown(f'<div style="font-weight:700;color:{PRIMARY};font-size:.85rem;margin-bottom:8px;">Par catégorie — {_sg_date_sel}</div>', unsafe_allow_html=True)
    if _sg_day_grp.empty:
        st.info("Aucune donnée.")
    else:
        _sg_day_grp = _sg_day_grp.rename(columns={"Bloc": "Classification", "SR": "S&R"})
        st.markdown(html_table(_sg_day_grp.reset_index(drop=True), ["Classification", "S&R"], max_h=300), unsafe_allow_html=True)

with _col_chart:
    st.markdown(f'<div style="font-weight:700;color:{PRIMARY};font-size:.85rem;margin-bottom:8px;">Évolution S&R par catégorie</div>', unsafe_allow_html=True)
    sg_cd = sg_tidy.groupby(["Date_dt", "Bloc"], as_index=False)["SR"].sum()
    if not sg_cd.empty:
        st.altair_chart(line_chart(sg_cd, "Date_dt", "SR", color="Bloc", title=f"{sg_pick}"), use_container_width=True)

st.markdown("---")

# ── COMPETITIVE FLOW PERFORMANCE ──
section_header("Performance concurrentielle des flux", "Comparaison S&R entre sociétés de gestion")

# Source : ASFIM (par fonds) agrégé par SG/Date — même source que Analyse par SG
# Fallback : Recap S&R si ASFIM non disponible
if not asfim_sr.empty:
    flow_daily = asfim_sr.groupby(["SG", "Date"], as_index=False)["SR"].sum(min_count=1)
    _flow_source = "ASFIM"
else:
    flow_daily = tidy_all.groupby(["SG", "Date"], as_index=False)["SR"].sum(min_count=1)
    _flow_source = "Recap S&R"

flow_daily["Date_dt"] = flow_daily["Date"].apply(lambda s: ddmm_to_dt(s, YEAR))
flow_daily = flow_daily.dropna(subset=["SG", "Date_dt"]).copy()

if flow_daily.empty:
    st.info("Données SG insuffisantes pour lancer l'analyse comparative des flux.")
else:
    if not asfim_cls_map.empty:
        cmp_classes = sorted(asfim_cls_map["Classification"].dropna().unique().tolist())
        cmp_class = st.selectbox("Type de fonds (comparaison)", ["TOUS"] + cmp_classes, index=0)
        if cmp_class != "TOUS":
            allowed_cmp_sg = set(
                asfim_cls_map.loc[asfim_cls_map["Classification"] == cmp_class, "SG"]
                .dropna()
                .unique()
                .tolist()
            )
            flow_daily = flow_daily[flow_daily["SG"].isin(allowed_cmp_sg)]
    else:
        cmp_classes = sorted(tidy_all["Bloc"].dropna().unique().tolist())
        cmp_class = st.selectbox("Type de fonds (comparaison)", ["TOUS"] + cmp_classes, index=0)
        if cmp_class != "TOUS":
            allowed_cmp_sg = set(
                tidy_all.loc[tidy_all["Bloc"] == cmp_class, "SG"]
                .dropna()
                .unique()
                .tolist()
            )
            flow_daily = flow_daily[flow_daily["SG"].isin(allowed_cmp_sg)]

    if flow_daily.empty or flow_daily["SG"].nunique() < 2:
        st.info("Pas assez de SG dans ce type de fonds pour comparer BMCE à un concurrent.")
        st.stop()

    market_dates = sorted(flow_daily["Date_dt"].dropna().unique().tolist())
    market_abs = (
        flow_daily.groupby("Date_dt")["SR"]
        .sum(min_count=1)
        .reindex(market_dates, fill_value=0.0)
        .abs()
    )

    sg_pool = sorted(flow_daily["SG"].dropna().unique().tolist())
    bmce_default = pick_bmce_sg(sg_pool) or sg_pool[0]

    c_cmp1, c_cmp2 = st.columns(2)
    with c_cmp1:
        bmce_sg = st.selectbox(
            "Entité BMCE (base flux)",
            sg_pool,
            index=sg_pool.index(bmce_default) if bmce_default in sg_pool else 0,
        )
    peer_options = [sg for sg in sg_pool if sg != bmce_sg]
    peer_default = peer_options[0]
    if peer_options:
        peer_rank = (
            flow_daily[flow_daily["SG"].isin(peer_options)]
            .groupby("SG", as_index=False)["SR"]
            .sum(min_count=1)
        )
        if not peer_rank.empty:
            peer_rank["abs_flow"] = peer_rank["SR"].abs()
            peer_default = peer_rank.sort_values("abs_flow", ascending=False)["SG"].iloc[0]
    with c_cmp2:
        peer_sg = st.selectbox(
            "Concurrent",
            peer_options,
            index=peer_options.index(peer_default) if peer_default in peer_options else 0,
        )

    bmce_ts = flow_series_on_market_days(flow_daily, bmce_sg, market_dates)
    peer_ts = flow_series_on_market_days(flow_daily, peer_sg, market_dates)
    bmce_kpi = compute_flow_kpis(bmce_ts, market_abs)
    peer_kpi = compute_flow_kpis(peer_ts, market_abs)

    # MTD : depuis ASFIM via bmce_kpi/peer_kpi — chaque publication hebdomadaire couvre la semaine entière
    # Somme des semaines du mois courant = MTD correct
    _bmce_mtd = bmce_kpi["mtd_net_flow"]
    _peer_mtd  = peer_kpi["mtd_net_flow"]
    # Pour le debug : identifier les dates ASFIM du mois courant
    _last_dt_asfim = pd.Timestamp(max(market_dates))
    _mtd_month, _mtd_year = _last_dt_asfim.month, _last_dt_asfim.year

    # ── DEBUG ──────────────────────────────────────────────────────────────
    with st.expander("🔍 Debug — Performance concurrentielle"):
        st.markdown(f"**Source des flux :** `{_flow_source}` agrégé par SG/Date  |  **Filtre :** `{cmp_class}`")
        st.markdown(f"**Dates de marché utilisées :** {len(market_dates)} dates · de `{market_dates[0].date() if market_dates else '—'}` à `{market_dates[-1].date() if market_dates else '—'}`")

        _dc1, _dc2 = st.columns(2)
        with _dc1:
            st.markdown(f"**{bmce_sg} — flux bruts (flow_daily)**")
            _bmce_raw = flow_daily[flow_daily["SG"] == bmce_sg].sort_values("Date_dt")
            st.dataframe(_bmce_raw[["Date", "SR"]].reset_index(drop=True), height=200)
            st.markdown(f"Nb lignes : `{len(_bmce_raw)}`  |  Somme SR : `{fmt_money(_bmce_raw['SR'].sum())}`")
            st.markdown("**Série reindexée sur dates marché (bmce_ts) :**")
            st.dataframe(bmce_ts, height=200)
            st.markdown(f"YTD calculé (somme séries) : `{fmt_money(bmce_kpi['ytd_net_flow'])}`")
            st.markdown(f"YTD Excel (ALL OPCVM) : **`{fmt_money(all_opcvm_ytd.get(bmce_sg, float('nan')))}`** ← valeur utilisée")

        with _dc2:
            st.markdown(f"**{peer_sg} — flux bruts (flow_daily)**")
            _peer_raw = flow_daily[flow_daily["SG"] == peer_sg].sort_values("Date_dt")
            st.dataframe(_peer_raw[["Date", "SR"]].reset_index(drop=True), height=200)
            st.markdown(f"Nb lignes : `{len(_peer_raw)}`  |  Somme SR : `{fmt_money(_peer_raw['SR'].sum())}`")
            st.markdown("**Série reindexée sur dates marché (peer_ts) :**")
            st.dataframe(peer_ts, height=200)
            st.markdown(f"YTD calculé (somme séries) : `{fmt_money(peer_kpi['ytd_net_flow'])}`")
            st.markdown(f"YTD Excel (ALL OPCVM) : **`{fmt_money(all_opcvm_ytd.get(peer_sg, float('nan')))}`** ← valeur utilisée")

        st.markdown(f"**MTD — mois retenu : {_mtd_month:02d}/{_mtd_year} (source : ASFIM — publications hebdomadaires)**")
        _dc3, _dc4 = st.columns(2)
        with _dc3:
            st.markdown(f"**{bmce_sg} — semaines ASFIM du mois**")
            _bmce_asfim_mtd = (
                flow_daily[
                    (flow_daily["SG"] == bmce_sg) &
                    (flow_daily["Date_dt"].dt.month == _mtd_month) &
                    (flow_daily["Date_dt"].dt.year == _mtd_year)
                ][["Date","SR"]].sort_values("Date")
            )
            st.dataframe(_bmce_asfim_mtd.reset_index(drop=True), height=200)
            st.markdown(f"**Total MTD : `{fmt_money(_bmce_mtd)}`**")
        with _dc4:
            st.markdown(f"**{peer_sg} — semaines ASFIM du mois**")
            _peer_asfim_mtd = (
                flow_daily[
                    (flow_daily["SG"] == peer_sg) &
                    (flow_daily["Date_dt"].dt.month == _mtd_month) &
                    (flow_daily["Date_dt"].dt.year == _mtd_year)
                ][["Date","SR"]].sort_values("Date")
            )
            st.dataframe(_peer_asfim_mtd.reset_index(drop=True), height=200)
            st.markdown(f"**Total MTD : `{fmt_money(_peer_mtd)}`**")

        st.markdown("**Flux marché total par date (market_abs) :**")
        _mkt_df = pd.DataFrame({"Date_dt": market_abs.index, "Market_abs": market_abs.values})
        st.dataframe(_mkt_df.tail(20), height=180)

        st.markdown("**Résumé des KPIs bruts :**")
        _pct_keys = {"positive_day_ratio", "abs_market_share"}
        _kpi_debug = pd.DataFrame([
            {"KPI": k,
             bmce_sg: (fmt_pct(bmce_kpi[k]) if k in _pct_keys else fmt_money(bmce_kpi[k])) if isinstance(bmce_kpi[k], float) else str(bmce_kpi[k]),
             peer_sg: (fmt_pct(peer_kpi[k]) if k in _pct_keys else fmt_money(peer_kpi[k])) if isinstance(peer_kpi[k], float) else str(peer_kpi[k])}
            for k in bmce_kpi
        ])
        st.dataframe(_kpi_debug, height=280)
    # ── FIN DEBUG ──────────────────────────────────────────────────────────

    # YTD : section ALL OPCVM du Recap S&R (valeur agrégée exacte)
    _bmce_ytd = all_opcvm_ytd.get(bmce_sg, bmce_kpi["ytd_net_flow"])
    _peer_ytd = all_opcvm_ytd.get(peer_sg, peer_kpi["ytd_net_flow"])

    _ytd_ecart  = ((_bmce_ytd - _peer_ytd)
                   if pd.notna(_bmce_ytd) and pd.notna(_peer_ytd) else np.nan)
    _mtd_ecart  = _bmce_mtd - _peer_mtd
    _vol_ecart  = peer_kpi["flow_volatility"] - bmce_kpi["flow_volatility"]
    m1, m2, m3 = st.columns(3)
    m1.markdown(_kpi("Écart flux net YTD",
        fmt_money(_ytd_ecart), GREEN if (pd.notna(_ytd_ecart) and _ytd_ecart >= 0) else RED),
        unsafe_allow_html=True)
    m2.markdown(_kpi("Écart flux net MTD",
        fmt_money(_mtd_ecart), GREEN if (pd.notna(_mtd_ecart) and _mtd_ecart >= 0) else RED),
        unsafe_allow_html=True)
    m3.markdown(_kpi("Écart volatilité des flux",
        fmt_money(_vol_ecart), GREEN if (pd.notna(_vol_ecart) and _vol_ecart >= 0) else RED),
        unsafe_allow_html=True)

    cmp_rows = [
        {
            "Indicateur": "Flux net YTD",
            "BMCE": fmt_money(_bmce_ytd),
            "Concurrent": fmt_money(_peer_ytd),
            "Écart (BMCE - Concurrent)": fmt_money(_ytd_ecart),
        },
        {
            "Indicateur": "Flux net MTD",
            "BMCE": fmt_money(_bmce_mtd),
            "Concurrent": fmt_money(_peer_mtd),
            "Écart (BMCE - Concurrent)": fmt_money(_mtd_ecart),
        },
        {
            "Indicateur": "Flux net 5 derniers jours",
            "BMCE": fmt_money(bmce_kpi["last_5d_net_flow"]),
            "Concurrent": fmt_money(peer_kpi["last_5d_net_flow"]),
            "Écart (BMCE - Concurrent)": fmt_money(bmce_kpi["last_5d_net_flow"] - peer_kpi["last_5d_net_flow"]),
        },
        {
            "Indicateur": "Volatilité des flux",
            "BMCE": fmt_money(bmce_kpi["flow_volatility"]),
            "Concurrent": fmt_money(peer_kpi["flow_volatility"]),
            "Écart (BMCE - Concurrent)": fmt_money(bmce_kpi["flow_volatility"] - peer_kpi["flow_volatility"]),
        },
    ]
    st.markdown(
        html_table(pd.DataFrame(cmp_rows), ["Indicateur", "BMCE", "Concurrent", "Écart (BMCE - Concurrent)"], max_h=380),
        unsafe_allow_html=True,
    )

    # Graphique cumulé par catégorie — un graphique par SG, une courbe par bloc
    def _cumul_by_bloc(sg):
        s = (
            tidy_all[tidy_all["SG"] == sg]
            .groupby(["Date", "Bloc"], as_index=False)["SR"]
            .sum(min_count=1)
        )
        s["Date_dt"] = s["Date"].apply(lambda d: ddmm_to_dt(d, YEAR))
        s = s.sort_values(["Bloc", "Date_dt"])
        s["Cumulative_SR"] = s.groupby("Bloc")["SR"].cumsum()
        return s

    _gc1, _gc2 = st.columns(2)
    with _gc1:
        _bmce_bloc = _cumul_by_bloc(bmce_sg)
        if not _bmce_bloc.empty:
            st.altair_chart(
                line_chart(_bmce_bloc, "Date_dt", "Cumulative_SR", color="Bloc",
                           title=f"{bmce_sg} — flux cumulé par catégorie"),
                use_container_width=True,
            )
    with _gc2:
        _peer_bloc = _cumul_by_bloc(peer_sg)
        if not _peer_bloc.empty:
            st.altair_chart(
                line_chart(_peer_bloc, "Date_dt", "Cumulative_SR", color="Bloc",
                           title=f"{peer_sg} — flux cumulé par catégorie"),
                use_container_width=True,
            )

if show_audit:
    st.markdown("---")
    section_header("Data (audit)", "Données brutes extraites pour vérification")
    audit = tidy.drop(columns=["Date_dt"]).reset_index(drop=True)
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
