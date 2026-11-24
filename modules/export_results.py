"""Membuat file unduhan dari hasil model yang benar-benar dijalankan."""
from __future__ import annotations

import io
from datetime import datetime

import pandas as pd

from .data_preprocessing import TRANSFORMASI, label_periode
from .pipeline import HasilProyeksi, tabel_hasil


def tabel_pengaturan(h: HasilProyeksi) -> pd.DataFrame:
    p = h.pengaturan
    baris = {
        "Model": p.spek.label(),
        "Order (p,d,q)": str(p.spek.order),
        "Seasonal order (P,D,Q,s)": str(p.spek.seasonal_order),
        "Konstanta C": "Ya" if p.spek.konstanta else "Tidak",
        "Transformasi": TRANSFORMASI[p.transformasi],
        "Data latih": f"{label_periode(p.awal_latih)} s.d. {label_periode(p.akhir_latih)}",
        "Data uji": f"{label_periode(p.akhir_latih + 1)} s.d. {label_periode(p.akhir_uji)}" if p.akhir_uji else "-",
        "Data untuk proyeksi": f"{label_periode(p.awal_latih)} s.d. {label_periode(h.akhir_data_proyeksi)}",
        "Horizon": p.horizon,
        "Tingkat kepercayaan": f"{p.tingkat_kepercayaan}%",
        "Koreksi bias exp()": "Ya" if p.koreksi_bias else "Tidak",
        "Paksa stasioner / invertible": f"{p.spek.paksa_stasioner} / {p.spek.paksa_invertible}",
        "Dibuat": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    return pd.DataFrame({"Nilai": baris}).rename_axis("Pengaturan")


def csv_hasil(h: HasilProyeksi) -> bytes:
    return tabel_hasil(h).to_csv().encode("utf-8-sig")  # utf-8-sig agar terbaca benar di Excel


def excel_laporan(h: HasilProyeksi) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        tabel_pengaturan(h).to_excel(w, sheet_name="Pengaturan")
        tabel_hasil(h).to_excel(w, sheet_name="Hasil")
        if h.akurasi is not None:
            h.akurasi.to_excel(w, sheet_name="Evaluasi")
        h.koefisien_akhir.to_excel(w, sheet_name="Koefisien")
        h.statistik_akhir.to_frame("Nilai").to_excel(w, sheet_name="Statistik model")
        if h.backtest is not None:
            from .arima_model import tabel_koefisien
            tabel_koefisien(h.backtest["hasil"]).to_excel(w, sheet_name="Koefisien backtest")
    return buf.getvalue()


def excel_hasil_saja(h: HasilProyeksi) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        tabel_hasil(h).to_excel(w, sheet_name="Hasil")
    return buf.getvalue()


def teks_ringkasan(h: HasilProyeksi) -> bytes:
    from .arima_model import ringkasan_teks
    bagian = ["ARIMA Forecasting Platform — ringkasan model", "=" * 60, "",
              tabel_pengaturan(h).to_string(), "",
              "MODEL UNTUK PROYEKSI KE DEPAN", "-" * 60,
              ringkasan_teks(h.hasil_akhir, h.pengaturan.spek, h.statistik_akhir)]
    if h.diagnostik_akhir["peringatan"]:
        bagian += ["", "PERINGATAN MODEL"] + [f"- {p}" for p in h.diagnostik_akhir["peringatan"]]
    if h.akurasi is not None:
        bagian += ["", "EVALUASI PERIODE UJI", "-" * 60, h.akurasi.to_string(float_format=lambda v: f"{v:,.4f}")]
    if h.peringatan:
        bagian += ["", "CATATAN"] + [f"- {p}" for p in h.peringatan]
    return "\n".join(bagian).encode("utf-8")
