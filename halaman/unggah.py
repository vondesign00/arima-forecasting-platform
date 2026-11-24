from pathlib import Path

import streamlit as st

from halaman import komponen as kp
from modules.data_loader import GalatData, baca_file, ringkasan_dataset, tebak_kolom
from modules.data_preprocessing import FREKUENSI, label_periode, siapkan_seri

CONTOH = Path(__file__).resolve().parent.parent / "sample_data" / "sample_dataset.xlsx"

st.title("1. Unggah Data")
st.caption("Unggah file Excel (.xlsx) atau CSV (.csv). Satu baris = satu periode.")

c1, c2 = st.columns([3, 1])
unggahan = c1.file_uploader("Pilih file", type=["xlsx", "xls", "csv"])
c2.write("")
c2.write("")
pakai_contoh = c2.button("Pakai data contoh", help="Data sintetis bulanan untuk mencoba aplikasi.")

id_unggahan = getattr(unggahan, "file_id", None) or (unggahan.name if unggahan else None)
if unggahan is not None and id_unggahan != st.session_state.get("id_unggahan"):
    st.session_state["id_unggahan"] = id_unggahan  # hanya dibaca sekali per file baru
    try:
        st.session_state[kp.DF] = baca_file(unggahan)
        st.session_state[kp.NAMA_FILE] = unggahan.name
        kp.reset_data()
    except GalatData as e:
        st.error(str(e))
        st.stop()
elif pakai_contoh:
    st.session_state[kp.DF] = baca_file(CONTOH)
    st.session_state[kp.NAMA_FILE] = "sample_dataset.xlsx (contoh sintetis)"
    kp.reset_data()

df = st.session_state.get(kp.DF)
if df is None:
    st.info("Belum ada file. Unggah file Anda atau klik **Pakai data contoh**.")
    st.stop()

st.success(f"File: **{st.session_state[kp.NAMA_FILE]}**")
ringkas = ringkasan_dataset(df)
m1, m2, m3 = st.columns(3)
m1.metric("Jumlah baris (observasi)", ringkas["jumlah_baris"])
m2.metric("Jumlah kolom (variabel)", ringkas["jumlah_kolom"])
m3.metric("Total sel kosong", int(df.isna().sum().sum()))

t1, t2 = st.tabs(["Pratinjau data", "Informasi kolom"])
t1.dataframe(df.head(50))
t2.dataframe(ringkas["info_kolom"])

st.subheader("Pilih kolom dan frekuensi")
tebak_tgl, tebak_target = tebak_kolom(df)
kolom = list(df.columns)
c1, c2, c3 = st.columns(3)
kolom_tgl = c1.selectbox("Kolom tanggal/periode", kolom, index=kolom.index(tebak_tgl),
                         help="Format yang dikenali: 2021-01-31, 31/01/2021, tanggal Excel, 2021M01, 2021Q1.")
kolom_target = c2.selectbox("Variabel yang diramal (target)", kolom, index=kolom.index(tebak_target),
                            help="Kolom angka yang ingin diproyeksikan, dalam satuan aslinya.")
frekuensi = c3.selectbox("Frekuensi data", list(FREKUENSI), index=0,
                         help="Seberapa sering data dicatat. Menentukan panjang musim bawaan.")

validasi = siapkan_seri(df, kolom_tgl, kolom_target, frekuensi)
for p in validasi.peringatan:
    st.warning(p)
if not validasi.ok:
    for g in validasi.galat:
        st.error(g)
    st.session_state.pop(kp.SERI, None)
    st.stop()

seri = validasi.seri
lama = st.session_state.get(kp.SERI)
if lama is None or not lama.equals(seri) or st.session_state.get(kp.TARGET) != kolom_target:
    kp.reset_hasil()
st.session_state[kp.SERI] = seri
st.session_state[kp.FREKUENSI] = frekuensi
st.session_state[kp.TARGET] = kolom_target

m1, m2, m3 = st.columns(3)
m1.metric("Rentang periode", f"{label_periode(seri.index.min())} – {label_periode(seri.index.max())}")
m2.metric("Jumlah periode", len(seri))
m3.metric("Periode kosong", int(seri.isna().sum()))
st.success("Data siap. Lanjutkan ke eksplorasi atau langsung ke konfigurasi model.")
c1, c2 = st.columns(2)
if c1.button("Eksplorasi data", icon=":material/query_stats:"):
    st.switch_page("halaman/eksplorasi.py")
if c2.button("Konfigurasi model", type="primary", icon=":material/tune:"):
    st.switch_page("halaman/konfigurasi.py")
