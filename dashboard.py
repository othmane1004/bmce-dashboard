import re
import io
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="BMCE Capital — Dashboard S&R", layout="wide")

# ==========================================================
# STYLE — TEXTE NOIR PARTOUT + DROPZONE UPLOADER BLANCHE
# + DELTA KPI EN VERT/ROUGE
# ==========================================================
BLACK = "#000000"
BMCE_BG     = "#F6F8FB"
BMCE_CARD   = "#FFFFFF"
BMCE_BORDER = "#E5E7EB"

st.markdown(
    f"""
    <style>
      /* ==========================================================
         FORCE ABSOLUE: tout le texte en NOIR (color + text-fill)
         ========================================================== */
      html, body, .stApp, [class*="css"], * {{
        color: {BLACK} !important;
        -webkit-text-fill-color: {BLACK} !important;
        text-shadow: none !important;
        opacity: 1 !important;
      }}

      /* Background app */
      .stApp {{
        background: {BMCE_BG} !important;
      }}

      /* Sidebar background */
      section[data-testid="stSidebar"] {{
        background: {BMCE_CARD} !important;
        border-right: 1px solid {BMCE_BORDER} !important;
      }}

      /* Titles */
      h1, h2, h3, h4, h5, h6 {{
        color: {BLACK} !important;
        -webkit-text-fill-color: {BLACK} !important;
        font-weight: 900 !important;
        letter-spacing: -0.02em;
      }}

      /* Markdown */
      .stMarkdown, .stMarkdown * {{
        color: {BLACK} !important;
        -webkit-text-fill-color: {BLACK} !important;
      }}

      /* Inputs */
      div[data-baseweb="select"] > div {{
        border-radius: 12px !important;
        border: 1px solid {BMCE_BORDER} !important;
        background: {BMCE_CARD} !important;
      }}
      input, textarea {{
        border-radius: 12px !important;
        border: 1px solid {BMCE_BORDER} !important;
        background: {BMCE_CARD} !important;
        color: {BLACK} !important;
        -webkit-text-fill-color: {BLACK} !important;
      }}

      /* Buttons (texte noir) */
      .stButton>button {{
        background: {BMCE_CARD} !important;
        color: {BLACK} !important;
        -webkit-text-fill-color: {BLACK} !important;
        border: 1px solid {BMCE_BORDER} !important;
        border-radius: 12px !important;
        padding: 0.62rem 1.05rem !important;
        font-weight: 900 !important;
      }}
      .stButton>button:hover {{
        background: #F2F4F7 !important;
      }}

      /* Expanders */
      details {{
        background: {BMCE_CARD} !important;
        border: 1px solid {BMCE_BORDER} !important;
        border-radius: 14px !important;
        padding: 6px 10px !important;
      }}
      details summary {{
        color: {BLACK} !important;
        -webkit-text-fill-color: {BLACK} !important;
        font-weight: 900 !important;
      }}

      /* Metrics */
      div[data-testid="metric-container"] {{
        background: {BMCE_CARD} !important;
        border: 1px solid {BMCE_BORDER} !important;
        padding: 14px 14px 10px 14px !important;
        border-radius: 16px !important;
        box-shadow: 0 2px 10px rgba(15, 23, 42, 0.06) !important;
      }}
      div[data-testid="metric-container"] * {{
        color: {BLACK} !important;
        -webkit-text-fill-color: {BLACK} !important;
        font-weight: 900 !important;
        opacity: 1 !important;
      }}

      /* ==========================================================
         DELTA st.metric : vert si positif, rouge si négatif
         (override uniquement le delta)
         ========================================================== */
      div[data-testid="metric-container"] [data-testid="stMetricDelta"] * {{
        -webkit-text-fill-color: inherit !important;
      }}
      div[data-testid="metric-container"] [data-testid="stMetricDelta"][data-direction="up"] * {{
        color: #15803D !important;
        -webkit-text-fill-color: #15803D !important;
        font-weight: 900 !important;
      }}
      div[data-testid="metric-container"] [data-testid="stMetricDelta"][data-direction="down"] * {{
        color: #B91C1C !important;
        -webkit-text-fill-color: #B91C1C !important;
        font-weight: 900 !important;
      }}

      /* Dataframe container */
      div[data-testid="stDataFrame"] {{
        border-radius: 14px !important;
        border: 1px solid {BMCE_BORDER} !important;
        overflow: hidden !important;
        background: {BMCE_CARD} !important;
      }}
      /* AG Grid text */
      div[data-testid="stDataFrame"] * {{
        color: {BLACK} !important;
        -webkit-text-fill-color: {BLACK} !important;
        opacity: 1 !important;
      }}
      /* Header cells */
      div[data-testid="stDataFrame"] .ag-header,
      div[data-testid="stDataFrame"] .ag-header-row,
      div[data-testid="stDataFrame"] .ag-header-cell,
      div[data-testid="stDataFrame"] .ag-header-cell-label {{
        background: {BMCE_BG} !important;
        border-bottom: 1px solid {BMCE_BORDER} !important;
      }}
      /* Body cells */
      div[data-testid="stDataFrame"] .ag-cell {{
        background: {BMCE_CARD} !important;
        border-color: {BMCE_BORDER} !important;
      }}
      /* Zebra rows */
      div[data-testid="stDataFrame"] .ag-row:nth-child(even) .ag-cell {{
        background: #FBFDFF !important;
      }}

      /* Tabs */
      button[data-baseweb="tab"], button[data-baseweb="tab"] * {{
        color: {BLACK} !important;
        -webkit-text-fill-color: {BLACK} !important;
        font-weight: 900 !important;
      }}

      /* Captions, labels */
      label, small, .stCaption, .stCaption * {{
        color: {BLACK} !important;
        -webkit-text-fill-color: {BLACK} !important;
        opacity: 1 !important;
      }}

      /* Horizontal rule */
      hr {{
        border: none !important;
        border-top: 1px solid {BMCE_BORDER} !important;
        margin: 1rem 0 !important;
      }}

      /* ==========================================================
         FILE UPLOADER — DROPZONE BLANCHE (SIDEBAR)
         ========================================================== */
      section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {{
        background: #FFFFFF !important;
        border: 1px solid {BMCE_BORDER} !important;
        border-radius: 14px !important;
      }}
      section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] * {{
        background: transparent !important;
        color: {BLACK} !important;
        -webkit-text-fill-color: {BLACK} !important;
      }}
      section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button {{
        background: #FFFFFF !important;
        border: 1px solid {BMCE_BORDER} !important;
        border-radius: 10px !important;
        color: {BLACK} !important;
        -webkit-text-fill-color: {BLACK} !important;
        font-weight: 900 !important;
      }}
      section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button:hover {{
        background: #F2F4F7 !important;
      }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ==========================================================
# Logo paths
# ==========================================================
LOGO_CANDIDATES = [
    Path("/Users/mac/Desktop/bmce_capital_logo.jpeg"),
    Path("bmce_capital_logo.jpeg"),
    Path("bmce_capital_logo.jpg"),
    Path("bmce_capital_logo.png"),
]

def resolve_logo_path() -> Path | None:
    for p in LOGO_CANDIDATES:
        try:
            if p.exists():
                return p
        except Exception:
            continue
    return None

logo_path = resolve_logo_path()

# ==========================================================
# Helpers
# ==========================================================
DATE_IN_TEXT = re.compile(r"(\d{1,2})\s*[\/\-\.]\s*(\d{1,2})")

def norm_ddmm(x: str) -> str | None:
    m = DATE_IN_TEXT.search(str(x) if x is not None else "")
    if not m:
        return None
    dd, mm = int(m.group(1)), int(m.group(2))
    if 1 <= dd <= 31 and 1 <= mm <= 12:
        return f"{dd:02d}/{mm:02d}"
    return None

def mmdd_sort_key(ddmm: str):
    dd, mm = ddmm.split("/")
    return (int(mm), int(dd))

def to_float(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.number)):
        return float(x)

    s = str(x).strip()
    if s.lower() in {"", "-", "—", "n/a", "na", "nan", "null", "none"}:
        return np.nan

    s = s.replace("\u00A0", " ").replace("\u202F", " ").replace(" ", "")
    is_pct = s.endswith("%")
    if is_pct:
        s = s[:-1]
    s = s.replace(",", ".")

    try:
        v = float(s)
    except Exception:
        return np.nan

    return v / 100.0 if is_pct else v

def format_money(x):
    if pd.isna(x):
        return "—"
    return f"{x:,.0f}".replace(",", " ")

def format_pct(x):
    if pd.isna(x):
        return "—"
    return f"{100*x:.2f}%"

def is_total_like(s):
    t = str(s).strip().lower()
    return ("total" in t and ("gén" in t or "gen" in t)) or t == "total"

def make_unique_cols(cols):
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

def find_col_like(cols, patterns):
    for c in list(cols):
        lc = str(c).strip().lower()
        for p in patterns:
            if p in lc:
                return c
    return None

# ==========================================================
# Robust reader for Recap sheet
# ==========================================================
def read_recap_sheet_bytes(xlsx_bytes: bytes, sheet_name: str) -> pd.DataFrame:
    raw = pd.read_excel(io.BytesIO(xlsx_bytes), sheet_name=sheet_name, header=None, engine="openpyxl")
    if raw.empty or raw.shape[0] < 2:
        return raw

    header_row = raw.iloc[0].tolist()
    cols = make_unique_cols(header_row)

    df = raw.iloc[1:].copy()
    df.columns = cols

    # inject bloc label as a data row in first column
    first_col = df.columns[0]
    inject = {c: np.nan for c in df.columns}
    inject[first_col] = str(header_row[0]).strip()
    df = pd.concat([pd.DataFrame([inject]), df], ignore_index=True)

    return df.dropna(how="all").reset_index(drop=True)

# ==========================================================
# Parser (enriched)
# ==========================================================
def parse_recap_sr(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = make_unique_cols(df.columns)
    df = df.reset_index(drop=True)

    date_cols = [c for c in df.columns if norm_ddmm(c)]

    ytd_col = None
    for c in df.columns:
        if str(c).strip().upper() in {"TOTAL YTD", "TOTAL"}:
            ytd_col = c
            break

    dec_col  = find_col_like(df.columns, ["decembre"])
    part_col = find_col_like(df.columns, ["part s&r", "part sr", "part"])

    # detect SG column among non-date columns (excluding YTD)
    non_date_cols = [c for c in df.columns if c not in date_cols and c != ytd_col]
    best_col, best_score = None, -1e9
    for c in non_date_cols:
        s = df[c].astype(str).fillna("")
        txt = s.str.strip().str.lower()
        non_empty = (txt != "") & (txt != "nan") & (txt != "none")
        num_like = s.apply(lambda v: pd.notna(to_float(v)))
        score = non_empty.mean() - 0.6 * num_like.mean()
        if score > best_score:
            best_score, best_col = score, c
    sg_col = best_col if (best_col in df.columns) else df.columns[0]

    def row_text(row: pd.Series) -> str:
        parts = []
        for v in row.values:
            if pd.isna(v):
                continue
            t = str(v).strip()
            if t and t.lower() not in {"nan", "none"}:
                parts.append(t)
        return " ".join(parts).upper()

    rows = []
    bloc = None

    for i in range(len(df)):
        row = df.iloc[i]
        trow = row_text(row)

        if "OPCVM" in trow and "ACTION" in trow:
            bloc = "ACTIONS"; continue
        if "OPCVM" in trow and "DIVERS" in trow:
            bloc = "DIVERSIFIES"; continue
        if "OPCVM" in trow and ("OMLT" in trow or "MON" in trow or "OBLIG" in trow):
            bloc = "OMLT"; continue
        if bloc is None:
            continue

        sg = row.get(sg_col, row.iloc[0])
        if pd.isna(sg):
            continue
        sg = str(sg).strip()
        if not sg or sg.lower() in {"nan", "none"}:
            continue
        if is_total_like(sg):
            continue

        # accept row if it has any numeric in date cols (or YTD/dec)
        has_num = any(pd.notna(to_float(row.get(c))) for c in date_cols)
        if not has_num and ytd_col:
            has_num = pd.notna(to_float(row.get(ytd_col)))
        if not has_num and dec_col:
            has_num = pd.notna(to_float(row.get(dec_col)))
        if not has_num:
            continue

        dec_val  = to_float(row.get(dec_col)) if dec_col else np.nan
        ytd_val  = to_float(row.get(ytd_col)) if ytd_col else np.nan
        part_val = to_float(row.get(part_col)) if part_col else np.nan

        for c in date_cols:
            ddmm = norm_ddmm(c)
            rows.append({
                "Bloc": bloc,
                "SG": sg,
                "Date": ddmm,
                "SR": to_float(row.get(c)),
                "DEC": dec_val,
                "YTD_row": ytd_val,
                "PART_row": part_val,
            })

    tidy = pd.DataFrame(rows)
    if tidy.empty:
        return tidy

    tidy["k"] = tidy["Date"].apply(mmdd_sort_key)
    tidy = tidy.sort_values(["k", "Bloc", "SG"]).drop(columns=["k"])
    return tidy

# ==========================================================
# Header
# ==========================================================
cH1, cH2 = st.columns([1, 6])
with cH1:
    if logo_path is not None:
        st.image(str(logo_path), use_container_width=True)
with cH2:
    st.markdown(
        f"""
        <div style="display:flex; align-items:center; gap:12px;">
          <div style="width:10px; height:44px; border-radius:10px; background:{BLACK};"></div>
          <div>
            <div style="font-size:28px; font-weight:900; color:{BLACK}; line-height:1.05;">
              Dashboard S&amp;R — ASFIM
            </div>
            <div style="color:{BLACK}; font-weight:650;">
              Pilotage des flux — vue marché, classements et tendances
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ==========================================================
# Sidebar
# ==========================================================
with st.sidebar:
    st.markdown(f"<div style='font-weight:900; color:{BLACK}; font-size:16px;'>Paramètres</div>", unsafe_allow_html=True)
    st.markdown("---")
    up = st.file_uploader("Uploader Analyse_SR.xlsx", type=["xlsx"])
    st.markdown("---")
    bloc_filter = st.multiselect("Blocs", ["ACTIONS", "DIVERSIFIES", "OMLT"], default=["ACTIONS", "DIVERSIFIES", "OMLT"])
    search_sg = st.text_input("Rechercher une SG", value="")
    st.markdown("---")
    show_data_tab = st.checkbox("Afficher l’onglet Data (audit)", value=False)

if up is None:
    st.info("Upload le fichier pour afficher le dashboard.")
    st.stop()

# ==========================================================
# Read + parse
# ==========================================================
xlsx_bytes = up.getvalue()
xls = pd.ExcelFile(io.BytesIO(xlsx_bytes), engine="openpyxl")

recap_name = None
for sh in xls.sheet_names:
    l = sh.lower()
    if "recap" in l and ("s&r" in l or "sr" in l):
        recap_name = sh
        break
if recap_name is None:
    for sh in xls.sheet_names:
        if "recap" in sh.lower():
            recap_name = sh
            break
if recap_name is None:
    st.error("Je ne trouve pas de feuille 'Recap S&R' dans ce fichier.")
    st.stop()

df_recap = read_recap_sheet_bytes(xlsx_bytes, recap_name)
tidy = parse_recap_sr(df_recap)

if tidy.empty:
    st.error("Impossible d’extraire les données de la feuille Recap S&R (structure non détectée).")
    st.stop()

with st.expander("🔎 Diagnostic détection blocs"):
    st.write("Feuille utilisée:", recap_name)
    st.write("Lignes par bloc:", tidy["Bloc"].value_counts())

# filters
tidy = tidy[tidy["Bloc"].isin(bloc_filter)]
if search_sg.strip():
    tidy = tidy[tidy["SG"].str.contains(search_sg.strip(), case=False, na=False)]

dates = sorted(tidy["Date"].dropna().unique().tolist(), key=mmdd_sort_key)
if not dates:
    st.error("Aucune date détectée.")
    st.stop()

last_date = dates[-1]
prev_date = dates[-2] if len(dates) >= 2 else None

cA, cB = st.columns([2, 1])
with cA:
    date_selected = st.select_slider("Date (S&R)", options=dates, value=last_date)
with cB:
    st.caption("Comparaison")
    st.write(f"Semaine précédente: **{prev_date or '—'}**")

pivot = tidy.pivot_table(index=["Bloc", "SG"], columns="Date", values="SR", aggfunc="sum").reset_index()
pivot.columns.name = None

# enrich per SG
ytd_map  = tidy.groupby(["Bloc", "SG"])["YTD_row"].max().reset_index().rename(columns={"YTD_row": "YTD"})
dec_map  = tidy.groupby(["Bloc", "SG"])["DEC"].max().reset_index().rename(columns={"DEC": "DEC"})
part_map = tidy.groupby(["Bloc", "SG"])["PART_row"].max().reset_index().rename(columns={"PART_row": "PART"})

df_rank = (
    pivot.merge(ytd_map, on=["Bloc", "SG"], how="left")
         .merge(dec_map, on=["Bloc", "SG"], how="left")
         .merge(part_map, on=["Bloc", "SG"], how="left")
)

df_rank["SR_sel"] = df_rank.get(date_selected, np.nan)
if prev_date:
    df_rank["SR_prev"] = df_rank.get(prev_date, np.nan)
    df_rank["WoW_abs"] = df_rank["SR_sel"] - df_rank["SR_prev"]
else:
    df_rank["SR_prev"] = np.nan
    df_rank["WoW_abs"] = np.nan

total_sel = df_rank["SR_sel"].sum(skipna=True)
total_prev = df_rank["SR_prev"].sum(skipna=True) if prev_date else np.nan
wow_total = total_sel - total_prev if prev_date else np.nan

# note card
st.markdown(
    f"""
    <div style="background:{BMCE_CARD}; border:1px solid {BMCE_BORDER}; border-radius:14px; padding:12px 14px; margin-top:10px;">
      <span style="color:{BLACK}; font-weight:900;">Note :</span>
      <span style="color:{BLACK}; font-weight:650;">
        Décembre / YTD / Part S&amp;R sont affichés tels qu’ils existent dans l’Excel.
      </span>
    </div>
    """,
    unsafe_allow_html=True,
)

# =========================
# KPI cohérents (1 ligne par SG, tous blocs confondus)
# =========================
sg_global = (
    df_rank.groupby("SG", as_index=False)
           .agg(SR_sel=("SR_sel", "sum"))
)

nb_sg   = int(sg_global["SG"].nunique())
nb_pos  = int((sg_global["SR_sel"] > 0).sum())
nb_neg  = int((sg_global["SR_sel"] < 0).sum())
nb_zero = int((sg_global["SR_sel"] == 0).sum())

# KPIs
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Date", date_selected)
k2.metric("Collecte nette (S&R)", format_money(total_sel), delta=(format_money(wow_total) if prev_date else None))
k3.metric("# SG", nb_sg)
k4.metric("SG en collecte", nb_pos)
k5.metric("SG en décollecte", nb_neg)

st.markdown("---")

# ==========================================================
# Leaderboards
# ==========================================================
left, right = st.columns([1.2, 1])

with left:
    st.subheader("🏆 Classement SG (S&R)")
    tmp = df_rank.copy().sort_values("SR_sel", ascending=False)
    tmp["S&R"] = tmp["SR_sel"]
    tmp["WoW"] = tmp["WoW_abs"]

    cols = ["Bloc", "SG", "S&R", "WoW", "DEC", "YTD", "PART"]
    cols = [c for c in cols if c in tmp.columns]

    st.dataframe(tmp[cols].head(50), use_container_width=True, height=560)

with right:
    st.subheader("⚡ Top / Flop")

    top3 = df_rank.sort_values("SR_sel", ascending=False).head(3)
    flop3 = df_rank.sort_values("SR_sel", ascending=True).head(3)

    def card(sg, sr, wow, dec, ytd, part):
        return f"""
        <div style="background:{BMCE_CARD}; border:1px solid {BMCE_BORDER}; border-radius:14px; padding:12px 12px; margin-bottom:10px;">
          <div style="font-weight:900; color:{BLACK}; font-size:21px; letter-spacing:-0.6px; margin-bottom:6px;">{sg}</div>
          <div style="margin-top:6px; font-weight:900; color:{BLACK};">S&amp;R: {format_money(sr)}</div>
          <div style="color:{BLACK}; font-weight:650;">WoW: {format_money(wow) if prev_date else "—"}</div>
          <div style="color:{BLACK}; font-weight:650;">Déc: {format_money(dec)}</div>
          <div style="color:{BLACK}; font-weight:650;">YTD: {format_money(ytd)}</div>
          <div style="color:{BLACK}; font-weight:650;">Part: {format_pct(part)}</div>
        </div>
        """

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Top 3 (collecte)**")
        for _, r in top3.iterrows():
            st.markdown(
                card(
                    r["SG"], r["SR_sel"], r["WoW_abs"],
                    r.get("DEC", np.nan), r.get("YTD", np.nan), r.get("PART", np.nan)
                ),
                unsafe_allow_html=True
            )

    with c2:
        st.markdown("**Flop 3 (décollecte)**")
        for _, r in flop3.iterrows():
            st.markdown(
                card(
                    r["SG"], r["SR_sel"], r["WoW_abs"],
                    r.get("DEC", np.nan), r.get("YTD", np.nan), r.get("PART", np.nan)
                ),
                unsafe_allow_html=True
            )

st.markdown("---")

# ==========================================================
# Trends
# ==========================================================
st.subheader("📈 Tendances (market view)")
trend = tidy.groupby(["Date", "Bloc"])["SR"].sum().reset_index()
trend["k"] = trend["Date"].apply(mmdd_sort_key)
trend = trend.sort_values("k").drop(columns=["k"])

tabs = st.tabs(["Global", "ACTIONS", "DIVERSIFIES", "OMLT"])
with tabs[0]:
    g = tidy.groupby("Date")["SR"].sum().reset_index()
    g["k"] = g["Date"].apply(mmdd_sort_key)
    g = g.sort_values("k").drop(columns=["k"]).set_index("Date")
    st.line_chart(g["SR"])
with tabs[1]:
    tr = trend[trend["Bloc"] == "ACTIONS"].pivot_table(index="Date", values="SR", aggfunc="sum")
    st.line_chart(tr)
with tabs[2]:
    tr = trend[trend["Bloc"] == "DIVERSIFIES"].pivot_table(index="Date", values="SR", aggfunc="sum")
    st.line_chart(tr)
with tabs[3]:
    tr = trend[trend["Bloc"] == "OMLT"].pivot_table(index="Date", values="SR", aggfunc="sum")
    st.line_chart(tr)

st.markdown("---")

# ==========================================================
# Drill-down
# ==========================================================
st.subheader("🔎 Drill-down : une Société de Gestion")
sg_list = sorted(df_rank["SG"].unique().tolist())
if not sg_list:
    st.warning("Aucune SG après filtres.")
    st.stop()

sg_pick = st.selectbox("Choisir une SG", sg_list, index=0)

sg_meta = df_rank[df_rank["SG"] == sg_pick].head(1)

m1, m2, m3 = st.columns(3)
with m1:
    st.metric("Décembre", format_money(float(sg_meta["DEC"].iloc[0])) if "DEC" in sg_meta else "—")
with m2:
    st.metric("YTD", format_money(float(sg_meta["YTD"].iloc[0])) if "YTD" in sg_meta else "—")
with m3:
    st.metric("Part S&R", format_pct(float(sg_meta["PART"].iloc[0])) if "PART" in sg_meta else "—")

sg_tidy = tidy[tidy["SG"] == sg_pick].copy()
sg_piv = sg_tidy.pivot_table(index="Date", columns="Bloc", values="SR", aggfunc="sum").fillna(0)
sg_piv = sg_piv.reset_index()
sg_piv["k"] = sg_piv["Date"].apply(mmdd_sort_key)
sg_piv = sg_piv.sort_values("k").drop(columns=["k"]).set_index("Date")

c1, c2 = st.columns([1.2, 1])
with c1:
    st.markdown("**Courbe S&R par bloc**")
    st.line_chart(sg_piv)
with c2:
    st.markdown("**Résumé à la date sélectionnée**")
    row = sg_piv.loc[date_selected] if date_selected in sg_piv.index else sg_piv.iloc[-1]
    for b in ["ACTIONS", "DIVERSIFIES", "OMLT"]:
        if b in row.index:
            st.metric(b, format_money(float(row[b])))

# ==========================================================
# Data / Audit
# ==========================================================
if show_data_tab:
    st.markdown("---")
    st.subheader("🧾 Data (audit)")
    st.dataframe(tidy, use_container_width=True, height=560)
