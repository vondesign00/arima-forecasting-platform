"""Satu pintu untuk menjalankan seluruh alur: validasi → backtest → proyeksi → evaluasi."""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from . import arima_model as am
from . import evaluation as ev
from . import forecasting as fc
from .data_preprocessing import cek_transformasi, ke_skala_asli, ke_skala_model, label_periode


@dataclass
class Pengaturan:
    spek: am.SpesifikasiModel
    transformasi: str                 # "log" | "sudah_log" | "tanpa"
    awal_latih: pd.Period
    akhir_latih: pd.Period
    akhir_uji: pd.Period | None       # None = tanpa periode uji
    horizon: int = 12
    tingkat_kepercayaan: int = 95
    koreksi_bias: bool = False


@dataclass
class HasilProyeksi:
    pengaturan: Pengaturan
    y_asli: pd.Series
    backtest: dict | None
    akurasi: pd.DataFrame | None
    hasil_akhir: object
    koefisien_akhir: pd.DataFrame
    statistik_akhir: pd.Series
    diagnostik_akhir: dict
    ramalan: pd.DataFrame
    akhir_data_proyeksi: pd.Period
    peringatan: list[str] = field(default_factory=list)


class GalatPengaturan(Exception):
    pass


def periksa(seri: pd.Series, peng: Pengaturan) -> list[str]:
    galat = list(cek_transformasi(seri.loc[peng.awal_latih:], peng.transformasi))
    if peng.akhir_latih < peng.awal_latih:
        galat.append("Akhir data latih harus setelah awal data latih.")
    if peng.akhir_uji is not None and peng.akhir_uji <= peng.akhir_latih:
        galat.append("Akhir data uji harus setelah akhir data latih.")
    n_latih = int(seri.loc[peng.awal_latih:peng.akhir_latih].notna().sum())
    minimal = peng.spek.minimal_observasi()
    if n_latih < minimal:
        galat.append(f"Data latih hanya {n_latih} observasi; model {peng.spek.label()} butuh minimal ±{minimal}. "
                     "Perpanjang data latih atau sederhanakan model.")
    if peng.horizon < 1:
        galat.append("Horizon proyeksi minimal 1 periode.")
    return galat


def jalankan(seri: pd.Series, peng: Pengaturan) -> HasilProyeksi:
    galat = periksa(seri, peng)
    if galat:
        raise GalatPengaturan("\n".join(galat))

    alpha = 1 - peng.tingkat_kepercayaan / 100
    y_model = ke_skala_model(seri, peng.transformasi)
    y_asli = ke_skala_asli(y_model, peng.transformasi)
    peringatan = []

    bt, akurasi = None, None
    if peng.akhir_uji is not None:
        bt = fc.backtest(y_model, peng.spek, peng.awal_latih, peng.akhir_latih, peng.akhir_uji,
                         peng.transformasi, alpha, peng.koreksi_bias)
        aktual = bt["tabel"]["aktual"]
        akurasi = ev.tabel_akurasi(aktual, {peng.spek.label(): bt["tabel"]["ramalan"], **bt["pembanding"]})
        if aktual.isna().any():
            peringatan.append(f"{int(aktual.isna().sum())} periode uji tanpa nilai aktual tidak dihitung di metrik.")

    # Proyeksi ke depan: estimasi ulang dengan data s.d. observasi terakhir yang terisi
    akhir_data = y_model.loc[peng.awal_latih:].last_valid_index()
    y_final = y_model.loc[peng.awal_latih:akhir_data]
    hasil_akhir = am.estimasi(y_final, peng.spek)
    stat = am.statistik_ringkas(hasil_akhir, y_final)
    ramalan = fc.proyeksi(hasil_akhir, peng.horizon, peng.spek, peng.transformasi, alpha, peng.koreksi_bias)
    if peng.akhir_uji is not None and akhir_data > peng.akhir_latih:
        peringatan.append(f"Proyeksi ke depan memakai model yang diestimasi ulang dengan data "
                          f"{label_periode(peng.awal_latih)} s.d. {label_periode(akhir_data)} "
                          "(termasuk periode uji), sehingga koefisiennya bisa berbeda dari model backtest.")

    return HasilProyeksi(
        pengaturan=peng, y_asli=y_asli, backtest=bt, akurasi=akurasi,
        hasil_akhir=hasil_akhir, koefisien_akhir=am.tabel_koefisien(hasil_akhir),
        statistik_akhir=stat, diagnostik_akhir=am.diagnostik(hasil_akhir, peng.spek),
        ramalan=ramalan, akhir_data_proyeksi=akhir_data, peringatan=peringatan,
    )


def tabel_hasil(h: HasilProyeksi) -> pd.DataFrame:
    """Tabel gabungan: periode uji (aktual, ramalan, galat) + proyeksi ke depan."""
    bagian = []
    if h.backtest is not None:
        t = h.backtest["tabel"].copy()
        t.insert(0, "Jenis", "Uji (backtest)")
        bagian.append(t)
    r = h.ramalan[["ramalan", "batas_bawah", "batas_atas"]].copy()
    r.insert(0, "Jenis", "Proyeksi ke depan")
    r["aktual"], r["galat"] = float("nan"), float("nan")
    bagian.append(r)
    tabel = pd.concat(bagian)[["Jenis", "aktual", "ramalan", "galat", "batas_bawah", "batas_atas"]]
    tabel.index = [label_periode(p) for p in tabel.index]
    tabel.index.name = "Periode"
    return tabel.rename(columns={"aktual": "Aktual", "ramalan": "Ramalan", "galat": "Galat (aktual − ramalan)",
                                 "batas_bawah": "Batas bawah", "batas_atas": "Batas atas"})
