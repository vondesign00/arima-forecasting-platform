"""Komponen tampilan bersama dan pengelolaan session state."""
from __future__ import annotations

import streamlit as st

from modules.data_preprocessing import label_periode

# Kunci session state
DF = "df_mentah"
NAMA_FILE = "nama_file"
SERI = "seri"
FREKUENSI = "frekuensi"
TARGET = "kolom_target"
HASIL = "hasil"


def gaya():
    st.markdown(
        """
        <style>
        .block-container {padding-top: 2rem; max-width: 1300px;}
        div[data-testid="stMetricValue"] {font-size: 1.5rem;}
        .langkah {border: 1px solid rgba(128,128,128,.25); border-radius: 10px; padding: 14px 16px; height: 100%;}
        .langkah b {color: #2563eb;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def reset_hasil():
    st.session_state.pop(HASIL, None)


def reset_data():
    for k in (SERI, HASIL):
        st.session_state.pop(k, None)


def status_sidebar():
    with st.sidebar:
        st.markdown("### Status proyek")
        nama = st.session_state.get(NAMA_FILE)
        seri = st.session_state.get(SERI)
        hasil = st.session_state.get(HASIL)
        st.markdown(f"{'✅' if nama else '⬜'} Data: {nama or 'belum diunggah'}")
        if seri is not None:
            st.caption(f"{st.session_state.get(TARGET)} · {st.session_state.get(FREKUENSI)} · "
                       f"{label_periode(seri.index.min())} – {label_periode(seri.index.max())}")
        st.markdown(f"{'✅' if hasil else '⬜'} Model: {hasil.pengaturan.spek.label() if hasil else 'belum dijalankan'}")


def perlu_seri():
    """Hentikan halaman bila data belum siap."""
    if st.session_state.get(SERI) is None:
        st.info("Belum ada data yang siap dipakai. Mulai dari halaman **Unggah Data**.")
        st.page_link("halaman/unggah.py", label="Ke halaman Unggah Data", icon=":material/upload_file:")
        st.stop()
    return st.session_state[SERI]


def perlu_hasil():
    if st.session_state.get(HASIL) is None:
        st.info("Model belum dijalankan. Atur model lalu klik **Jalankan Proyeksi** di halaman Konfigurasi.")
        st.page_link("halaman/konfigurasi.py", label="Ke halaman Konfigurasi", icon=":material/tune:")
        st.stop()
    return st.session_state[HASIL]


def fmt_angka(v, desimal: int = 2) -> str:
    """Format angka gaya Indonesia: 1.234.567,89"""
    try:
        teks = f"{float(v):,.{desimal}f}"
    except (TypeError, ValueError):
        return str(v)
    return teks.replace(",", "_").replace(".", ",").replace("_", ".")
