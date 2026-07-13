"""
Aplikasi DSS — Deteksi Persamaan pada Pokoknya Nama Merek Dagang
================================================================
Sistem Pendukung Keputusan berbasis model hybrid: tekstual (Jaro-Winkler),
fonetik (Double Metaphone), dan semantik (FastText pretrained cc.id.100).

Bobot & threshold default = hasil grid search (Langkah F):
    tekstual 0.5 · fonetik 0.1 · semantik 0.4 · threshold 0.55

Cara menjalankan (lokal):
    pip install streamlit rapidfuzz metaphone pandas numpy fasttext-wheel
    streamlit run app_merek_pdki.py

File yang dibutuhkan di folder yang sama:
    haystack_didaftar_fonID.csv   (wajib — basis pembanding merek terdaftar)
    cc.id.100.bin                 (opsional — tanpa ini skor semantik = 0)

Tema warna mengikuti color guide situs PDKI:
    Navy #162657 · Kuning #ffce28 · Abu kebiruan #eaeff5 · Putih #ffffff
Tema Streamlit (slider, fokus, dsb.) di-set otomatis dari script ini —
tidak perlu membuat config.toml manual.
"""

import re
import html as html_lib
import unicodedata
import numpy as np
import pandas as pd
import streamlit as st
from rapidfuzz import process
from rapidfuzz.distance import JaroWinkler
from metaphone import doublemetaphone

# ============================================================
# KONFIGURASI
# ============================================================
HAYSTACK_PATH = "haystack_didaftar_fonID.csv"
SEM_MODEL_PATH = "cc.id.100.bin"
BOBOT_DEFAULT = {"teks": 0.50, "fonetik": 0.10, "semantik": 0.40}  # hasil Langkah F
THRESHOLD_DEFAULT = 0.55

# Palet PDKI
C_NAVY = "#162657"
C_YELLOW = "#ffce28"
C_LIGHT = "#eaeff5"
C_WHITE = "#ffffff"
C_GREEN = "#1fae5e"   # hijau solid untuk status sukses

# ------------------------------------------------------------
# Terapkan tema Streamlit secara programatik (agar slider dkk.
# ikut kuning PDKI tanpa perlu file .streamlit/config.toml).
# ------------------------------------------------------------
def _set_theme():
    pairs = {
        "theme.base": "light",
        "theme.primaryColor": C_YELLOW,
        "theme.backgroundColor": C_LIGHT,
        "theme.secondaryBackgroundColor": C_WHITE,
        "theme.textColor": C_NAVY,
    }
    for k, v in pairs.items():
        try:
            st.config.set_option(k, v)          # Streamlit >= 1.30
        except Exception:
            try:
                from streamlit import config as _cfg
                _cfg.set_option(k, v)           # fallback versi lama
            except Exception:
                pass

_set_theme()

st.set_page_config(page_title="DSS Kemiripan Merek", page_icon="🔍", layout="wide")

# ============================================================
# TEMA PDKI (CSS)
# ============================================================
st.markdown(f"""
<style>
/* ---------- Latar utama ---------- */
.stApp {{ background-color: {C_LIGHT}; }}
[data-testid="stHeader"] {{ background-color: {C_NAVY}; }}
[data-testid="stHeader"] * {{ color: {C_WHITE} !important; }}

.stApp, .stMarkdown, .stCaption, p, li {{ color: {C_NAVY}; }}
h1, h2, h3, h4, h5, h6 {{ color: {C_NAVY} !important; }}

/* ---------- Sidebar (navy PDKI) ---------- */
[data-testid="stSidebar"] {{
    background-color: {C_NAVY};
    border-right: 4px solid {C_YELLOW};
}}
[data-testid="stSidebar"] * {{ color: {C_WHITE} !important; }}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {{ color: {C_YELLOW} !important; }}
[data-testid="stSidebar"] hr {{ border-color: {C_YELLOW}; opacity: 0.6; }}

/* ---------- Slider: paksa kuning PDKI ---------- */
/* rel/jalur slider → putih transparan netral (menimpa merah default) */
div[data-testid="stSlider"] [data-baseweb="slider"] > div:first-child > div {{
    background: rgba(255, 255, 255, 0.35) !important;
}}
[data-testid="stMain"] div[data-testid="stSlider"] [data-baseweb="slider"] > div:first-child > div {{
    background: {C_NAVY}26 !important;
}}
/* thumb (bulatan) kuning */
div[data-testid="stSlider"] div[role="slider"] {{
    background-color: {C_YELLOW} !important;
    border: 2px solid {C_WHITE} !important;
    box-shadow: 0 0 0 2px {C_NAVY}55 !important;
}}
/* angka nilai di atas thumb */
div[data-testid="stSliderThumbValue"] {{
    color: {C_YELLOW} !important;
    font-weight: 700;
}}
/* angka min–max di bawah slider */
div[data-testid="stSliderTickBar"] div,
div[data-testid="stTickBar"] div {{
    color: {C_LIGHT} !important;
}}

/* ---------- Alert di sidebar ---------- */
/* SUKSES → hijau SOLID */
[data-testid="stSidebar"] [data-testid="stAlert"]:has([data-testid="stAlertContentSuccess"]) {{
    background-color: {C_GREEN} !important;
    border: none !important;
    border-radius: 8px;
}}
[data-testid="stSidebar"] [data-testid="stAlert"]:has([data-testid="stAlertContentSuccess"]) * {{
    color: {C_WHITE} !important;
    font-weight: 600;
}}
/* PERINGATAN → kuning solid, teks navy */
[data-testid="stSidebar"] [data-testid="stAlert"]:has([data-testid="stAlertContentWarning"]) {{
    background-color: {C_YELLOW} !important;
    border: none !important;
    border-radius: 8px;
}}
[data-testid="stSidebar"] [data-testid="stAlert"]:has([data-testid="stAlertContentWarning"]) * {{
    color: {C_NAVY} !important;
    font-weight: 600;
}}

/* Metric di sidebar */
[data-testid="stSidebar"] [data-testid="stMetric"] {{
    background-color: rgba(255, 255, 255, 0.08);
    border-left: 4px solid {C_YELLOW};
    border-radius: 8px;
    padding: 10px 14px;
}}
[data-testid="stSidebar"] [data-testid="stMetricValue"] {{ color: {C_YELLOW} !important; }}

/* ---------- Banner judul ala navbar PDKI ---------- */
.pdki-banner {{
    background-color: {C_NAVY};
    border-radius: 12px;
    padding: 22px 28px;
    margin-bottom: 6px;
    border-bottom: 5px solid {C_YELLOW};
}}
.pdki-banner h1 {{ color: {C_WHITE} !important; font-size: 1.7rem; margin: 0 0 6px 0; }}
.pdki-banner p {{ color: {C_LIGHT}; margin: 0; font-size: 0.92rem; }}
.pdki-banner b {{ color: {C_YELLOW}; }}

/* ---------- Input ---------- */
.stTextInput input {{
    background-color: {C_WHITE};
    color: {C_NAVY};
    border: 1.5px solid {C_NAVY}33;
    border-radius: 999px;
    padding-left: 18px;
}}
.stTextInput input:focus {{
    border-color: {C_YELLOW};
    box-shadow: 0 0 0 2px {C_YELLOW}66;
}}
.stTextInput label {{ color: {C_NAVY} !important; font-weight: 600; }}

/* ---------- Tombol utama: SELALU kuning terlihat jelas ---------- */
.stButton button[kind="primary"],
button[data-testid="stBaseButton-primary"],
button[data-testid="baseButton-primary"] {{
    background-color: {C_YELLOW} !important;
    color: {C_NAVY} !important;
    border: 2px solid {C_NAVY} !important;
    border-radius: 999px !important;
    font-weight: 700 !important;
    box-shadow: 0 2px 6px rgba(22, 38, 87, 0.25);
}}
.stButton button[kind="primary"]:hover,
button[data-testid="stBaseButton-primary"]:hover,
button[data-testid="baseButton-primary"]:hover {{
    background-color: {C_NAVY} !important;
    color: {C_YELLOW} !important;
    border-color: {C_YELLOW} !important;
}}
.stButton button[kind="primary"] p,
button[data-testid="stBaseButton-primary"] p {{
    color: inherit !important;
}}

/* ---------- Kartu metrik utama ---------- */
[data-testid="stMain"] [data-testid="stMetric"] {{
    background-color: {C_WHITE};
    border-radius: 12px;
    border-top: 4px solid {C_YELLOW};
    padding: 14px 18px;
    box-shadow: 0 1px 4px rgba(22, 38, 87, 0.10);
}}
[data-testid="stMain"] [data-testid="stMetricValue"] {{ color: {C_NAVY}; }}
[data-testid="stMain"] [data-testid="stMetricLabel"] p {{ color: {C_NAVY}; opacity: 0.75; }}

/* ============================================================
   KARTU KANDIDAT — HTML kustom (tidak bergantung DOM Streamlit)
   ============================================================ */
.pdki-card {{
    background-color: {C_WHITE};
    border: 1px solid rgba(22, 38, 87, 0.18);
    border-left: 6px solid {C_YELLOW};
    border-radius: 12px;
    padding: 18px 22px 16px 22px;
    margin-bottom: 16px;
    box-shadow: 0 3px 10px rgba(22, 38, 87, 0.14);
}}
.pdki-card-head {{
    display: flex;
    align-items: flex-start;
    gap: 18px;
    margin-bottom: 14px;
}}
.pdki-rank {{
    font-size: 1.6rem;
    font-weight: 800;
    color: {C_NAVY};
    min-width: 52px;
}}
.pdki-nama {{
    flex: 1;
}}
.pdki-nama .nm {{
    font-size: 1.05rem;
    font-weight: 800;
    color: {C_NAVY};
}}
.pdki-nama .sub {{
    font-size: 0.88rem;
    color: {C_NAVY};
    opacity: 0.75;
}}
.pdki-status {{
    min-width: 140px;
    font-size: 0.88rem;
    color: {C_NAVY};
}}
.pdki-status .lbl {{ opacity: 0.65; }}
.pdki-skor {{
    display: flex;
    flex-direction: column;
    align-items: flex-end;
    gap: 5px;
    min-width: 110px;
}}
.pdki-badge {{
    display: inline-block;
    padding: 3px 14px;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 800;
    letter-spacing: 0.08em;
    color: {C_WHITE};
}}
.badge-mirip {{ background-color: #d93025; }}
.badge-aman  {{ background-color: {C_GREEN}; }}
.pdki-skor-val {{
    font-size: 1.55rem;
    font-weight: 800;
    color: {C_NAVY};
    line-height: 1;
}}
/* baris tiga dimensi */
.pdki-dims {{
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 22px;
}}
.pdki-dim-label {{
    font-size: 0.85rem;
    font-weight: 600;
    color: {C_NAVY};
    margin-bottom: 5px;
}}
.pdki-dim-label b {{ color: {C_NAVY}; }}
/* progress bar kustom: jalur biru keabu-abuan, isi kuning PDKI */
.pdki-bar {{
    height: 13px;
    background-color: #b9c7dd;
    border-radius: 999px;
    overflow: hidden;
    border: 1px solid rgba(22, 38, 87, 0.15);
}}
.pdki-bar-fill {{
    height: 100%;
    background: linear-gradient(90deg, {C_YELLOW}, #f0b400);
    border-radius: 999px;
}}
.pdki-alasan {{
    margin-top: 12px;
    padding-top: 10px;
    border-top: 1px dashed rgba(22, 38, 87, 0.25);
    font-size: 0.86rem;
    color: {C_NAVY};
    opacity: 0.85;
}}
@media (max-width: 900px) {{
    .pdki-dims {{ grid-template-columns: 1fr; gap: 12px; }}
    .pdki-card-head {{ flex-wrap: wrap; }}
}}

/* ---------- Divider & alert utama ---------- */
[data-testid="stMain"] hr {{ border-color: {C_YELLOW}; }}
[data-testid="stMain"] [data-testid="stAlert"] {{ border-radius: 10px; }}
</style>
""", unsafe_allow_html=True)

# ============================================================
# PREPROCESSING (identik dengan Langkah B)
# ============================================================
LEET = {"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t"}
FIGUR = r"(lukisan|logo|gambar|device)"
PHON_RULES = [("sy", "sh"), ("ny", "n"), ("kh", "h")]


def strip_accents(s):
    return "".join(c for c in unicodedata.normalize("NFKD", str(s)) if not unicodedata.combining(c))


def base_normalize(nama):
    s = strip_accents(nama).lower()
    s = re.sub(r"\+\s*" + FIGUR + r"\b", " ", s)
    s = re.sub(r"\b" + FIGUR + r"\b", " ", s)
    s = re.sub(r"(?<=[a-z])[013457](?=[a-z])", lambda m: LEET.get(m.group(0), m.group(0)), s)
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def phonetic_normalize(base_str, aktif=True):
    if not aktif:
        return base_str
    x = base_str
    for a, b in PHON_RULES:
        x = x.replace(a, b)
    x = re.sub(r"([aeiou])\1+", r"\1", x)
    return re.sub(r"\s+", " ", x).strip()


def dm_code(text):
    return "".join(doublemetaphone(t)[0] for t in text.split())


# ============================================================
# MUAT DATA & MODEL (cache)
# ============================================================
@st.cache_data(show_spinner="Memuat basis data merek...")
def load_haystack(path):
    hay = pd.read_csv(path, dtype=str).fillna("")
    hay.columns = [c.strip().lstrip("\ufeff") for c in hay.columns]
    if "nama_base" not in hay.columns:
        hay["nama_base"] = hay["nama_merek"].apply(base_normalize)
    if "nama_phon" not in hay.columns:
        hay["nama_phon"] = hay["nama_base"].apply(phonetic_normalize)
    hay = hay[hay["nama_base"].str.len() > 0].reset_index(drop=True)
    hay["dm"] = hay["nama_phon"].apply(dm_code)
    return hay


@st.cache_resource(show_spinner="Memuat model semantik...")
def load_sem_model(path):
    try:
        import fasttext
        return fasttext.load_model(path)
    except Exception:
        return None


@st.cache_data(show_spinner="Menghitung embedding basis data...")
def compute_haystack_emb(_model, names, dim):
    if _model is None:
        return None
    vecs = [_model.get_sentence_vector(str(n).replace("\n", " ")) for n in names]
    mat = np.vstack(vecs).astype("float32")
    return mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9)


# ============================================================
# MESIN PENCARIAN
# ============================================================
def cari_kandidat(query, hay, sem_model, hay_emb_n, bobot, threshold, topn=10):
    qb = base_normalize(query)
    qp = phonetic_normalize(qb)
    qd = dm_code(qp)

    jw = process.cdist([qb], hay["nama_base"].tolist(), scorer=JaroWinkler.normalized_similarity)[0]
    ph = process.cdist([qd], hay["dm"].tolist(), scorer=JaroWinkler.normalized_similarity)[0]

    if sem_model is not None and hay_emb_n is not None:
        qv = sem_model.get_sentence_vector(qb.replace("\n", " "))
        qv = qv / (np.linalg.norm(qv) + 1e-9)
        se = np.clip(hay_emb_n @ qv, 0, 1)
    else:
        se = np.zeros(len(hay))

    skor = bobot["teks"] * jw + bobot["fonetik"] * ph + bobot["semantik"] * se
    order = np.argsort(-skor)

    hasil = []
    for i in order:
        if hay.iloc[i]["nama_base"] == qb:
            continue
        hasil.append({
            "nama_merek": hay.iloc[i]["nama_merek"],
            "kelas": hay.iloc[i].get("kelas", "-"),
            "pemilik": hay.iloc[i].get("pemilik", "-"),
            "status": hay.iloc[i].get("status_permohonan", "-"),
            "skor": float(skor[i]),
            "teks": float(jw[i]),
            "fonetik": float(ph[i]),
            "semantik": float(se[i]),
        })
        if len(hasil) >= topn:
            break
    return qb, hasil


def alasan_dimensi(kand):
    peta = [
        ("persamaan tulisan", kand["teks"]),
        ("persamaan bunyi ucapan", kand["fonetik"]),
        ("persamaan makna", kand["semantik"]),
    ]
    peta.sort(key=lambda x: x[1], reverse=True)
    tinggi = [pp for pp in peta if pp[1] >= 0.75]
    if not tinggi:
        return "Tidak ada dimensi dengan kemiripan menonjol; kemiripan bersifat menyeluruh namun sedang."
    frasa = ", ".join(f"{pp[0]} ({pp[1]:.0%})" for pp in tinggi)
    return f"Ditandai terutama karena {frasa}."


# ============================================================
# ANTARMUKA
# ============================================================
st.markdown("""
<div class="pdki-banner">
    <h1>🔍 Sistem Pendukung Keputusan: Kemiripan Nama Merek</h1>
    <p>Deteksi persamaan pada pokoknya melalui model hybrid: tekstual (Jaro-Winkler),
    fonetik (Double Metaphone), dan semantik (FastText).
    <b>Hanya alat bantu pemeriksaan, keputusan akhir tetap pada pemeriksa.</b></p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ Konfigurasi Model")
    st.caption("Nilai default = hasil grid search.")
    w_teks = st.slider("Bobot Tekstual", 0.0, 1.0, BOBOT_DEFAULT["teks"], 0.05)
    w_fon = st.slider("Bobot Fonetik", 0.0, 1.0, BOBOT_DEFAULT["fonetik"], 0.05)
    w_sem = st.slider("Bobot Semantik", 0.0, 1.0, BOBOT_DEFAULT["semantik"], 0.05)
    tot = w_teks + w_fon + w_sem or 1
    bobot = {"teks": w_teks / tot, "fonetik": w_fon / tot, "semantik": w_sem / tot}
    st.caption(f"Ternormalisasi → teks {bobot['teks']:.2f} · fonetik {bobot['fonetik']:.2f} · semantik {bobot['semantik']:.2f}")
    st.divider()
    threshold = st.slider("Threshold keputusan (mirip jika skor ≥)", 0.0, 1.0, THRESHOLD_DEFAULT, 0.01)
    topn = st.slider("Jumlah kandidat ditampilkan", 3, 20, 10)

try:
    hay = load_haystack(HAYSTACK_PATH)
except Exception as e:
    st.error(f"Gagal memuat '{HAYSTACK_PATH}'. Pastikan file ada di folder yang sama. ({e})")
    st.stop()

sem_model = load_sem_model(SEM_MODEL_PATH)
hay_emb_n = None
if sem_model is not None:
    hay_emb_n = compute_haystack_emb(sem_model, hay["nama_base"].tolist(), sem_model.get_dimension())
    st.sidebar.success("Model semantik aktif.")
else:
    st.sidebar.warning("Model semantik tidak dimuat — skor semantik = 0. "
                       "Letakkan cc.id.100.bin untuk mengaktifkan.")
st.sidebar.metric("Merek dalam basis data", f"{len(hay):,}")

col_in, col_btn = st.columns([4, 1])
with col_in:
    query = st.text_input("Nama merek yang akan diperiksa",
                          placeholder="e.g. KOJIE SAN, MICROBIKE, GLOW SKIN")
with col_btn:
    st.write("")
    st.write("")
    cari = st.button("🔎 Pencarian Data", type="primary", use_container_width=True)

if cari and query.strip():
    qb, hasil = cari_kandidat(query, hay, sem_model, hay_emb_n, bobot, threshold, topn)
    n_mirip = sum(1 for h in hasil if h["skor"] >= threshold)
    skor_tertinggi = hasil[0]["skor"] if hasil else 0.0

    c1, c2, c3 = st.columns(3)
    c1.metric("Kandidat diperiksa", len(hasil))
    c2.metric("Berpotensi mirip", n_mirip)
    c3.metric("Skor tertinggi", f"{skor_tertinggi:.3f}")

    if n_mirip > 0:
        st.error(f"⚠️ {n_mirip} merek berpotensi mirip — perlu pemeriksaan mendalam oleh pemeriksa.")
    else:
        st.success("✅ Tidak ditemukan kandidat dengan kemiripan signifikan di atas threshold.")

    st.divider()
    st.subheader("📋 Daftar Kandidat Termirip")

    def bar_html(label, val):
        pct = min(max(val, 0.0), 1.0) * 100
        return (f'<div><div class="pdki-dim-label">{label} <b>{val:.0%}</b></div>'
                f'<div class="pdki-bar"><div class="pdki-bar-fill" style="width:{pct:.1f}%"></div></div></div>')

    for idx, h in enumerate(hasil, 1):
        mirip = h["skor"] >= threshold
        nama = html_lib.escape(str(h["nama_merek"]))
        pemilik = html_lib.escape(str(h["pemilik"])[:30])
        kelas = html_lib.escape(str(h["kelas"]))
        status = html_lib.escape(str(h["status"]))
        label = "MIRIP" if mirip else "AMAN"
        kelas_badge = "badge-mirip" if mirip else "badge-aman"
        alasan = (f'<div class="pdki-alasan">🧠 {html_lib.escape(alasan_dimensi(h))}</div>'
                  if mirip else "")

        st.markdown(f"""
<div class="pdki-card">
  <div class="pdki-card-head">
    <div class="pdki-rank">#{idx}</div>
    <div class="pdki-nama">
      <div class="nm">{nama}</div>
      <div class="sub">Kelas {kelas} · {pemilik}</div>
    </div>
    <div class="pdki-status"><span class="lbl">Status:</span><br>{status}</div>
    <div class="pdki-skor">
      <span class="pdki-badge {kelas_badge}">{label}</span>
      <span class="pdki-skor-val">{h["skor"]:.3f}</span>
    </div>
  </div>
  <div class="pdki-dims">
    {bar_html("Tekstual (tulisan)", h["teks"])}
    {bar_html("Fonetik (bunyi)", h["fonetik"])}
    {bar_html("Semantik (makna)", h["semantik"])}
  </div>
  {alasan}
</div>
""", unsafe_allow_html=True)

    st.divider()
    st.caption("ℹ️ Skor = kombinasi berbobot tiga dimensi, dipetakan ke parameter Penjelasan Pasal 21 "
               "UU 20/2016 (tulisan, bunyi ucapan, makna). Sistem bersifat pendukung keputusan; "
               "hasil merupakan indikasi awal, bukan penetapan hukum.")
elif cari:
    st.warning("Masukkan nama merek terlebih dahulu.")
else:
    st.info("Masukkan nama merek lalu klik **Pencarian Data** untuk melihat kandidat termirip beserta alasannya.")
