import re
import io
from pathlib import Path
from datetime import date

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

st.set_page_config(page_title="BMCE Capital — Dashboard S&R", layout="wide")

# ──────────────────────────────────────────────
# DESIGN TOKENS
# ──────────────────────────────────────────────
PRIMARY = "#0F172A"
SECONDARY = "#475569"
MUTED = "#94A3B8"
BG = "#FFFFFF"
SURFACE = "#F8FAFC"
BORDER = "#E2E8F0"
ACCENT = "#2563EB"
GREEN = "#16A34A"
RED = "#DC2626"
CHART_GRID = "#F1F5F9"
PALETTE = [ACCENT, "#10B981", "#F59E0B", "#EF4444", "#8B5CF6", "#EC4899"]

# ──────────────────────────────────────────────
# CSS  (%-format to avoid brace conflicts)
# ──────────────────────────────────────────────
_T = dict(P=PRIMARY, S=SECONDARY, M=MUTED, B=BG, SF=SURFACE,
          BD=BORDER, AC=ACCENT, GR=GREEN, RD=RED)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,500;0,9..40,700;1,9..40,400&display=swap');

html,body,.stApp,[class*="css"]{font-family:'DM Sans',-apple-system,BlinkMacSystemFont,sans-serif!important;color:%(P)s!important;-webkit-text-fill-color:%(P)s!important}
html,body,.stApp,.main,.block-container,div[data-testid="stAppViewContainer"],div[data-testid="stAppViewContainer"]>.main,div[data-testid="stMain"],header[data-testid="stHeader"]{background:%(B)s!important}

section[data-testid="stSidebar"]{background:%(SF)s!important;border-right:1px solid %(BD)s!important}
section[data-testid="stSidebar"] *{color:%(P)s!important;-webkit-text-fill-color:%(P)s!important}

h1{font-size:1.75rem!important;font-weight:700!important;color:%(P)s!important}
h2{font-size:1.35rem!important;font-weight:700!important;color:%(P)s!important}
h3{font-size:1.10rem!important;font-weight:600!important;color:%(P)s!important}
p,span,label,.stMarkdown,.stMarkdown *{color:%(P)s!important;-webkit-text-fill-color:%(P)s!important}

div[data-baseweb="select"]>div{border-radius:10px!important;border:1px solid %(BD)s!important;background:%(B)s!important}
input,textarea{border-radius:10px!important;border:1px solid %(BD)s!important;background:%(B)s!important;color:%(P)s!important}
div[data-baseweb="popover"],div[data-baseweb="menu"],ul[role="listbox"]{background-color:%(B)s!important;border:1px solid %(BD)s!important;border-radius:10px!important}
div[data-baseweb="menu"] *,li[role="option"]{background-color:%(B)s!important;color:%(P)s!important;-webkit-text-fill-color:%(P)s!important}
div[data-baseweb="menu"] [role="option"]:hover,li[role="option"]:hover{background-color:%(SF)s!important}

.stButton>button{background:%(B)s!important;color:%(P)s!important;border:1px solid %(BD)s!important;border-radius:10px!important;font-weight:600!important}
.stButton>button:hover{background:%(SF)s!important}

div[data-testid="metric-container"]{background:%(B)s!important;border:1px solid %(BD)s!important;padding:16px!important;border-radius:12px!important;box-shadow:0 1px 3px rgba(15,23,42,.04)!important}
div[data-testid="metric-container"] *{color:%(P)s!important;-webkit-text-fill-color:%(P)s!important;font-weight:600!important;opacity:1!important}
div[data-testid="metric-container"] [data-testid="stMetricDelta"] *{-webkit-text-fill-color:inherit!important}
div[data-testid="metric-container"] [data-direction="up"] *{color:%(GR)s!important;-webkit-text-fill-color:%(GR)s!important}
div[data-testid="metric-container"] [data-direction="down"] *{color:%(RD)s!important;-webkit-text-fill-color:%(RD)s!important}

details{background:%(B)s!important;border:1px solid %(BD)s!important;border-radius:12px!important}
details summary{color:%(P)s!important;font-weight:600!important}

.stTabs [data-baseweb="tab-list"]{border-bottom:1px solid %(BD)s!important}
.stTabs [data-baseweb="tab"]{background:transparent!important;color:%(M)s!important;-webkit-text-fill-color:%(M)s!important;font-weight:600!important;border-bottom:2px solid transparent!important}
.stTabs [aria-selected="true"]{color:%(P)s!important;-webkit-text-fill-color:%(P)s!important;border-bottom:2px solid %(AC)s!important;font-weight:700!important}

div[data-testid="stVegaLiteChart"]{background:%(B)s!important;border:1px solid %(BD)s!important;border-radius:12px!important;padding:12px!important}
div[data-testid="stVegaLiteChart"] canvas,div[data-testid="stVegaLiteChart"] svg{background:%(B)s!important}

/* ═══ VEGA TOOLTIP — force white ═══ */
#vg-tooltip-element,
#vg-tooltip-element.vg-tooltip,
.vg-tooltip,
div.vg-tooltip{
  background-color:%(B)s!important;
  color:%(P)s!important;
  border:1px solid %(BD)s!important;
  border-radius:8px!important;
  padding:8px 12px!important;
  font-family:'DM Sans',sans-serif!important;
  font-size:0.82rem!important;
  box-shadow:0 4px 12px rgba(15,23,42,.10)!important;
}
#vg-tooltip-element *,.vg-tooltip *,div.vg-tooltip *{
  color:%(P)s!important;-webkit-text-fill-color:%(P)s!important;
  background:transparent!important;
}
#vg-tooltip-element td.key,.vg-tooltip td.key{color:%(M)s!important;-webkit-text-fill-color:%(M)s!important;font-weight:500!important}
#vg-tooltip-element td.value,.vg-tooltip td.value{color:%(P)s!important;-webkit-text-fill-color:%(P)s!important;font-weight:700!important}
#vg-tooltip-element table,.vg-tooltip table{border-collapse:collapse!important}
#vg-tooltip-element td,.vg-tooltip td{padding:2px 6px!important;border:none!important}

/* ═══ TOOLBAR — keep icons nice ═══ */
div[data-testid="stElementToolbar"]{background:transparent!important}
div[data-testid="stElementToolbar"] button,
div[data-testid="stElementToolbar"] [role="button"]{
  background:%(B)s!important;color:%(P)s!important;-webkit-text-fill-color:%(P)s!important;
  border:1px solid %(BD)s!important;border-radius:10px!important;
  width:34px!important;height:34px!important;
  display:inline-flex!important;align-items:center!important;justify-content:center!important;
  cursor:pointer!important;
  box-shadow:0 1px 2px rgba(15,23,42,.06)!important;
}
div[data-testid="stElementToolbar"] button:hover{
  background:%(SF)s!important;
  box-shadow:0 4px 12px rgba(15,23,42,.10)!important;
  transform:translateY(-1px);
}
div[data-testid="stElementToolbar"] svg,div[data-testid="stElementToolbar"] path{fill:%(P)s!important;color:%(P)s!important}

button[kind="icon"],button[kind="icon"] *{background:%(B)s!important;color:%(P)s!important;fill:%(P)s!important}

section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"]{background:%(B)s!important;border:1.5px dashed %(BD)s!important;border-radius:12px!important}
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] *{background:transparent!important;color:%(P)s!important}

hr{border:none!important;border-top:1px solid %(BD)s!important;margin:1.25rem 0!important}
div[data-testid="stSlider"] *{color:%(P)s!important;-webkit-text-fill-color:%(P)s!important}
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
                "tickColor": BORDER, "labelColor": SECONDARY,
                "titleColor": PRIMARY, "labelFont": "DM Sans",
                "titleFont": "DM Sans", "labelFontSize": 11,
                "titleFontSize": 12, "titleFontWeight": 600,
            },
            "legend": {"labelColor": SECONDARY, "titleColor": PRIMARY,
                       "labelFont": "DM Sans", "titleFont": "DM Sans"},
            "title": {"color": PRIMARY, "font": "DM Sans", "fontWeight": 700, "fontSize": 14},
            "line": {"strokeWidth": 2.5},
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
    m = _RE_DATE.search(str(x) if x is not None else "")
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
    cols = unique_cols(hdr)
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

    date_cols = [c for c in df.columns if norm_ddmm(c)]

    ytd_col = None
    for c in df.columns:
        if str(c).strip().upper() in {"TOTAL YTD", "TOTAL"}:
            ytd_col = c
            break

    # REMOVE DEC + PART: we do NOT look for them anymore
    non_date = [c for c in df.columns if c not in date_cols and c != ytd_col]

    best, bsc = None, -1e9
    for c in non_date:
        s = df[c].astype(str).fillna("")
        txt = s.str.strip().str.lower()
        ne = (txt != "") & (txt != "nan") & (txt != "none")
        num = s.apply(lambda v: pd.notna(to_float(v)))
        sc = ne.mean() - 0.6 * num.mean()
        if sc > bsc:
            bsc, best = sc, c
    sg_col = best if best in df.columns else df.columns[0]

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
        if "OPCVM" in t and ("OMLT" in t or "MON" in t or "OBLIG" in t):
            bloc = "OMLT"
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

        for c in date_cols:
            rows.append({
                "Bloc": bloc,
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
# ALTAIR LINE CHART
# ──────────────────────────────────────────────
def line_chart(data, x, y, color=None, title="", h=340):
    base = alt.Chart(data).mark_line(strokeWidth=2.5, point=alt.OverlayMarkDef(filled=True, size=40))
    enc = {"x": alt.X(f"{x}:T", title="Date"), "y": alt.Y(f"{y}:Q", title="S&R")}
    tips = [
        alt.Tooltip(f"{x}:T", title="Date", format="%d/%m/%Y"),
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
        al = "right" if c in _RIGHT else "left"
        lab = _COL_LABELS.get(c, c)
        hdr += (
            f'<th style="text-align:{al};padding:11px 16px;font-weight:600;font-size:0.72rem;color:{MUTED};'
            f'letter-spacing:0.06em;text-transform:uppercase;border-bottom:2px solid {BORDER};'
            f'background:{SURFACE};position:sticky;top:0;z-index:1;">{lab}</th>'
        )

    body = ""
    for i in range(len(df)):
        r = df.iloc[i]
        row_bg = BG if i % 2 == 0 else SURFACE
        cells = ""
        for c in columns:
            val = r[c]
            al = "right" if c in _RIGHT else "left"
            txt = _fmt_cell(c, val)
            clr = _cell_color(c, val)
            fw = "700" if c == "S&R" else "600" if c == "SG" else "500"
            fs = "0.87rem" if c == "SG" else "0.84rem"
            cells += (
                f'<td style="text-align:{al};padding:10px 16px;color:{clr};font-weight:{fw};'
                f'font-size:{fs};border-bottom:1px solid {BORDER};white-space:nowrap;">{txt}</td>'
            )
        body += (
            f'<tr style="background:{row_bg};transition:background .12s;" '
            f'onmouseover="this.style.background=\'{SURFACE}\'" '
            f'onmouseout="this.style.background=\'{row_bg}\'">{cells}</tr>'
        )

    return (
        f'<div style="border:1px solid {BORDER};border-radius:14px;overflow:hidden;">'
        f'<div style="max-height:{max_h}px;overflow-y:auto;">'
        f'<table style="width:100%;border-collapse:collapse;background:{BG};font-family:\'DM Sans\',sans-serif;">'
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

# ══════════════════════════════════════════════
#  LAYOUT
# ══════════════════════════════════════════════
cH1, cH2 = st.columns([1, 6])
with cH1:
    if logo_path:
        st.image(str(logo_path), use_container_width=True)
with cH2:
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:14px;padding:8px 0;">'
        f'<div style="width:4px;height:48px;border-radius:4px;background:{ACCENT};"></div><div>'
        f'<div style="font-size:1.65rem;font-weight:700;color:{PRIMARY};line-height:1.1;">Dashboard S&amp;R — ASFIM</div>'
        f'<div style="color:{SECONDARY};font-weight:500;font-size:0.95rem;margin-top:2px;">Pilotage des flux — vue marché, classements et tendances</div>'
        f'</div></div>',
        unsafe_allow_html=True,
    )

with st.sidebar:
    st.markdown(f"<div style='font-weight:700;color:{PRIMARY};font-size:1rem;'>Paramètres</div>", unsafe_allow_html=True)
    st.markdown("---")
    up = st.file_uploader("Uploader Analyse_SR.xlsx", type=["xlsx"])
    st.markdown("---")
    show_audit = st.checkbox("Afficher l'onglet Data (audit)", value=False)

if up is None:
    st.info("Importez le fichier **Analyse_SR.xlsx** pour afficher le dashboard.")
    st.stop()

xlsx_bytes = up.getvalue()
xls = pd.ExcelFile(io.BytesIO(xlsx_bytes), engine="openpyxl")
tcd_totals = read_tcd_daily_totals(xlsx_bytes)
tcd_sg_daily = read_tcd_sg_daily(xlsx_bytes)
asfim_cls_map = read_asfim_sg_classification(xlsx_bytes)

if not asfim_cls_map.empty:
    class_options = sorted(asfim_cls_map["Classification"].dropna().unique().tolist())
    with st.sidebar:
        st.markdown("---")
        class_filter = st.multiselect("Classification", class_options, default=class_options)
        search_sg = st.text_input("Rechercher une SG", value="")
    allowed_sg_sidebar = set(
        asfim_cls_map.loc[asfim_cls_map["Classification"].isin(class_filter), "SG"]
        .dropna()
        .unique()
        .tolist()
    )
else:
    class_options = ["ACTIONS", "DIVERSIFIES", "OMLT"]
    with st.sidebar:
        st.markdown("---")
        class_filter = st.multiselect("Blocs", class_options, default=class_options)
        search_sg = st.text_input("Rechercher une SG", value="")
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
if tidy.empty:
    st.error("Impossible d'extraire les données.")
    st.stop()

# Keep an unfiltered copy for drill-down category filtering.
tidy_all = tidy.copy()

YEAR = date.today().year
tidy["Date_dt"] = tidy["Date"].apply(lambda s: ddmm_to_dt(s, YEAR))

with st.expander("Diagnostic détection blocs"):
    st.write("Feuille :", recap_name)
    st.write("Dates :", sorted(tidy["Date"].unique().tolist(), key=mmdd_sort))
    st.write("Lignes/bloc :", tidy["Bloc"].value_counts())

if allowed_sg_sidebar is not None:
    tidy = tidy[tidy["SG"].isin(allowed_sg_sidebar)]
else:
    tidy = tidy[tidy["Bloc"].isin(class_filter)]
if search_sg.strip():
    tidy = tidy[tidy["SG"].str.contains(search_sg.strip(), case=False, na=False)]

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

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Date", date_sel)
k2.metric("Collecte nette (S&R)", fmt_money(total_sel))
k3.metric("# SG", nb_sg)
k4.metric("SG en collecte", nb_pos)
k5.metric("SG en décollecte", nb_neg)

st.markdown("---")

# ── CLASSEMENT ──
st.subheader("Classement SG (S&R)")
rank_cls_options = ["TOUS"] + sorted(asfim_cls_map["Classification"].dropna().unique().tolist()) if not asfim_cls_map.empty else ["TOUS"]
rank_cls_sel = st.selectbox("Filtre classement (overall)", rank_cls_options, index=0)

if not tcd_sg_daily.empty:
    rank_src = tcd_sg_daily.copy()
    if not asfim_cls_map.empty:
        rank_src = rank_src.merge(asfim_cls_map, on="SG", how="left")
    else:
        rank_src["Classification"] = "TOUS"

    if rank_cls_sel != "TOUS":
        rank_src = rank_src[rank_src["Classification"] == rank_cls_sel]
    if search_sg.strip():
        rank_src = rank_src[rank_src["SG"].str.contains(search_sg.strip(), case=False, na=False)]

    tmp = (
        rank_src.groupby(["Classification", "SG"], as_index=False)["SR"]
        .sum(min_count=1)
        .rename(columns={"SR": "S&R"})
        .sort_values("S&R", ascending=False)
        .reset_index(drop=True)
    )
    show_cols = ["Classification", "SG", "S&R"]
else:
    tmp = df_rank.copy().sort_values("SR_sel", ascending=False)
    tmp["S&R"] = tmp["SR_sel"]
    show_cols = ["Bloc", "SG", "S&R", "YTD"]

if tmp.empty:
    st.info("Aucune donnée de classement pour ce filtre.")
else:
    st.markdown(
        html_table(tmp[show_cols].head(100).reset_index(drop=True), show_cols, max_h=520),
        unsafe_allow_html=True,
    )

st.markdown("---")

# ── TOP / FLOP ──
st.subheader("Top / Flop")
if "S&R" in tmp.columns:
    top3 = tmp.sort_values("S&R", ascending=False).head(3)
    flop3 = tmp.sort_values("S&R", ascending=True).head(3)
else:
    top3 = tmp.head(3)
    flop3 = tmp.sort_values("SR_sel", ascending=True).head(3)

col_t, col_f = st.columns(2)
with col_t:
    st.markdown(f'<div style="font-weight:700;color:{GREEN};font-size:0.95rem;margin-bottom:8px;">Top 3 — collecte</div>', unsafe_allow_html=True)
    for _, r in top3.iterrows():
        sr_val = r["S&R"] if "S&R" in r else r.get("SR_sel", np.nan)
        bloc_val = r.get("Classification", r.get("Bloc", "TOUS"))
        st.markdown(_card(r["SG"], bloc_val, sr_val, r.get("YTD", np.nan), GREEN), unsafe_allow_html=True)

with col_f:
    st.markdown(f'<div style="font-weight:700;color:{RED};font-size:0.95rem;margin-bottom:8px;">Flop 3 — décollecte</div>', unsafe_allow_html=True)
    for _, r in flop3.iterrows():
        sr_val = r["S&R"] if "S&R" in r else r.get("SR_sel", np.nan)
        bloc_val = r.get("Classification", r.get("Bloc", "TOUS"))
        st.markdown(_card(r["SG"], bloc_val, sr_val, r.get("YTD", np.nan), RED), unsafe_allow_html=True)

st.markdown("---")

# ── TENDANCES ──
st.subheader("Tendances (market view)")
tabs = st.tabs(["Global", "ACTIONS", "DIVERSIFIES", "OMLT"])

with tabs[0]:
    if tcd_totals:
        g = pd.DataFrame(
            [{"Date": d, "SR": v} for d, v in tcd_totals.items() if pd.notna(v)]
        )
        if not g.empty:
            g["k"] = g["Date"].apply(mmdd_sort)
            g = g.sort_values("k").drop(columns=["k"])
            g["Date_dt"] = g["Date"].apply(lambda s: ddmm_to_dt(s, YEAR))
        else:
            g = tidy.groupby("Date_dt", as_index=False)["SR"].sum().sort_values("Date_dt")
    else:
        g = tidy.groupby("Date_dt", as_index=False)["SR"].sum().sort_values("Date_dt")
    st.altair_chart(line_chart(g, "Date_dt", "SR", title="Collecte nette globale S&R"), use_container_width=True)

with tabs[1]:
    d = tidy[tidy["Bloc"] == "ACTIONS"].groupby("Date_dt", as_index=False)["SR"].sum().sort_values("Date_dt")
    st.altair_chart(line_chart(d, "Date_dt", "SR", title="ACTIONS — S&R"), use_container_width=True)

with tabs[2]:
    d = tidy[tidy["Bloc"] == "DIVERSIFIES"].groupby("Date_dt", as_index=False)["SR"].sum().sort_values("Date_dt")
    st.altair_chart(line_chart(d, "Date_dt", "SR", title="DIVERSIFIÉS — S&R"), use_container_width=True)

with tabs[3]:
    d = tidy[tidy["Bloc"] == "OMLT"].groupby("Date_dt", as_index=False)["SR"].sum().sort_values("Date_dt")
    st.altair_chart(line_chart(d, "Date_dt", "SR", title="OMLT — S&R"), use_container_width=True)

st.markdown("---")

# ── DRILL-DOWN ──
st.subheader("Société de Gestion")
if not asfim_cls_map.empty:
    cls_values = sorted(asfim_cls_map["Classification"].dropna().unique().tolist())
else:
    cls_values = ["ACTIONS", "DIVERSIFIES", "OMLT"]

drill_cat = st.selectbox("Classification", ["TOUS"] + cls_values, index=0)

if drill_cat == "TOUS":
    allowed_sg = set(tidy_all["SG"].dropna().unique().tolist())
elif not asfim_cls_map.empty:
    allowed_sg = set(asfim_cls_map.loc[asfim_cls_map["Classification"] == drill_cat, "SG"].dropna().unique().tolist())
else:
    allowed_sg = set(tidy_all[tidy_all["Bloc"] == drill_cat]["SG"].dropna().unique().tolist())

if not tcd_sg_daily.empty:
    sg_list = sorted([sg for sg in tcd_sg_daily["SG"].dropna().unique().tolist() if sg in allowed_sg])
else:
    sg_list = sorted([sg for sg in df_rank["SG"].dropna().unique().tolist() if sg in allowed_sg])
if not sg_list:
    st.warning("Aucune SG.")
    st.stop()

sg_pick = st.selectbox("Choisir une SG", sg_list, index=0)

if not tcd_sg_daily.empty:
    sg_tidy = tcd_sg_daily[tcd_sg_daily["SG"] == sg_pick].copy()
    sg_tidy["Date_dt"] = sg_tidy["Date"].apply(lambda s: ddmm_to_dt(s, YEAR))
    sg_tidy["Bloc"] = "GLOBAL"
    st.metric("YTD", "—")
else:
    sg_meta = df_rank[df_rank["SG"] == sg_pick].head(1)
    st.metric("YTD", fmt_money(float(sg_meta["YTD"].iloc[0])) if ("YTD" in sg_meta and len(sg_meta) > 0) else "—")
    sg_tidy = tidy[tidy["SG"] == sg_pick].copy()

c1, c2 = st.columns([1.2, 1])

with c1:
    st.markdown(f'<div style="font-weight:700;color:{PRIMARY};margin-bottom:4px;">Courbe S&R par bloc</div>', unsafe_allow_html=True)
    sg_cd = sg_tidy.groupby(["Date_dt", "Bloc"], as_index=False)["SR"].sum()
    if not sg_cd.empty:
        st.altair_chart(line_chart(sg_cd, "Date_dt", "SR", color="Bloc", title=f"{sg_pick} — S&R par bloc"), use_container_width=True)

with c2:
    st.markdown(f'<div style="font-weight:700;color:{PRIMARY};margin-bottom:4px;">Résumé à la dernière date</div>', unsafe_allow_html=True)
    sg_piv = (
        sg_tidy.pivot_table(index="Date_dt", columns="Bloc", values="SR", aggfunc="sum")
        .fillna(0)
        .sort_index()
    )
    if not sg_piv.empty:
        row = sg_piv.iloc[-1]
        if "GLOBAL" in row.index:
            st.metric("GLOBAL", fmt_money(float(row["GLOBAL"])))
        else:
            for b in ["ACTIONS", "DIVERSIFIES", "OMLT"]:
                if b in row.index:
                    st.metric(b, fmt_money(float(row[b])))

if show_audit:
    st.markdown("---")
    st.subheader("Data (audit)")
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
