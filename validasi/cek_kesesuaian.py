"""
Validasi aplikasi terhadap (1) hasil EViews dan (2) kode acuan reference/arima_engine.py,
memakai data dan pengaturan yang SAMA.

Pengaturan acuan (model ARIMA terpilih, EViews):
    log(outflow), SARIMA(0,0,0)(1,0,0)12 + C,
    data latih 2021M01–2025M06, data uji 2025M07–2026M06.

Pemakaian (dari folder proyek):
    python validasi/cek_kesesuaian.py "C:/path/data_outflow.xlsx" --tanggal tanggal --target outflow
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

AKAR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(AKAR))
sys.path.insert(0, str(AKAR / "reference"))

from modules.arima_model import SpesifikasiModel  # noqa: E402
from modules.data_loader import baca_file  # noqa: E402
from modules.data_preprocessing import siapkan_seri  # noqa: E402
from modules.pipeline import Pengaturan, jalankan  # noqa: E402

# Angka EViews (catatan proyek, model SAR(12) pada log outflow).
# Dibulatkan seperti di catatan, sehingga toleransinya disesuaikan.
EVIEWS = {
    # nama: (nilai, toleransi absolut)
    "C": (12.682, 0.01),
    "SAR(12)": (0.448, 0.02),
    "Akaike info criterion": (2.450, 0.02),
    "Schwarz criterion": (2.560, 0.02),
    "MAPE (%)": (76.0, 1.0),
    "sMAPE (%)": (32.8, 1.0),
    "MAE": (87165, 1000),
    "RMSE": (116741, 1000),
    "Theil U2": (0.36, 0.02),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--tanggal", default="tanggal")
    ap.add_argument("--target", default="outflow")
    ap.add_argument("--awal", default="2021-01")
    ap.add_argument("--akhir-latih", default="2025-06")
    ap.add_argument("--akhir-uji", default="2026-06")
    args = ap.parse_args()

    v = siapkan_seri(baca_file(args.file), args.tanggal, args.target, "Bulanan")
    for p in v.peringatan:
        print("PERINGATAN DATA:", p)
    if not v.ok:
        raise SystemExit("\n".join(v.galat))

    spek = SpesifikasiModel(order=(0, 0, 0), seasonal_order=(1, 0, 0, 12), konstanta=True)
    peng = Pengaturan(spek=spek, transformasi="log",
                      awal_latih=pd.Period(args.awal, "M"), akhir_latih=pd.Period(args.akhir_latih, "M"),
                      akhir_uji=pd.Period(args.akhir_uji, "M"), horizon=12)
    h = jalankan(v.seri, peng)

    # ---- (1) dibandingkan dengan EViews: model yang diestimasi pada data LATIH
    from modules.arima_model import statistik_ringkas, tabel_koefisien
    hasil_bt = h.backtest["hasil"]
    koef = tabel_koefisien(hasil_bt)["Koefisien"]
    stat = statistik_ringkas(hasil_bt, h.backtest["y_latih"])
    akur = h.akurasi.loc[spek.label()]
    aplikasi = {
        "C": koef.get("C"), "SAR(12)": koef.get("SAR(12)"),
        "Akaike info criterion": stat["Akaike info criterion (per obs., EViews)"],
        "Schwarz criterion": stat["Schwarz criterion (per obs., EViews)"],
        **{k: akur.get(k) for k in ("MAPE (%)", "sMAPE (%)", "MAE", "RMSE", "Theil U2")},
    }
    baris = []
    for nama, (acuan, tol) in EVIEWS.items():
        nilai = aplikasi.get(nama)
        selisih = None if nilai is None else nilai - acuan
        baris.append({"Ukuran": nama, "EViews": acuan, "Aplikasi": nilai, "Selisih": selisih,
                      "Toleransi": tol, "Status": "SESUAI" if selisih is not None and abs(selisih) <= tol else "BEDA"})
    t1 = pd.DataFrame(baris).set_index("Ukuran")
    print(f"\n(1) Aplikasi vs EViews — {spek.label()}, latih {peng.awal_latih}–{peng.akhir_latih}\n")
    print(t1.to_string(float_format=lambda x: f"{x:,.4f}"))

    # ---- (2) dibandingkan dengan kode acuan (notebook/arima_engine.py lama)
    import arima_engine as ref
    spek_ref = ref.SpesifikasiModel(order=(0, 0, 0), seasonal_order=(1, 0, 0, 12), konstanta=True)
    bt_ref = ref.backtest(v.seri, spek_ref, args.awal, args.akhir_latih, args.akhir_uji)
    r_ref = bt_ref["tabel"][spek_ref.label()]
    r_app = h.backtest["tabel"]["ramalan"]
    beda_maks = float(np.nanmax(np.abs(r_ref.to_numpy() - r_app.reindex(r_ref.index).to_numpy())))
    akur_ref = bt_ref["akurasi"].loc[spek_ref.label()]
    print("\n(2) Aplikasi vs kode acuan (reference/arima_engine.py), periode uji yang sama\n")
    print(pd.DataFrame({"Kode acuan": akur_ref, "Aplikasi": akur.reindex(akur_ref.index)}).to_string(
        float_format=lambda x: f"{x:,.6f}"))
    print(f"\nSelisih ramalan terbesar per periode: {beda_maks:.6f} "
          f"→ {'SESUAI' if beda_maks < 1e-6 * max(1, float(r_ref.abs().max())) else 'BEDA'}")

    print("\nCatatan: angka EViews di atas dibulatkan; selisih kecil di koefisien/AIC wajar karena "
          "inisialisasi likelihood berbeda. Theil U2 aplikasi dihitung di dalam periode uji.")


if __name__ == "__main__":
    main()
