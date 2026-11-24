"""
Backtest dan proyeksi. Metodologi sama dengan kode acuan:
- pembagian latih/uji kronologis (tidak diacak);
- model diestimasi HANYA dengan data latih, lalu diramal dinamis multi-langkah
  sepanjang periode uji (tidak memakai nilai aktual periode uji);
- proyeksi ke depan: model diestimasi ulang dengan data s.d. periode terakhir;
- hasil dikembalikan ke skala asli dengan exp() bila model memakai log.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import arima_model as am
from .data_preprocessing import ke_skala_asli


def proyeksi(hasil, langkah: int, spek: am.SpesifikasiModel, mode_transformasi: str,
             alpha: float = 0.05, koreksi_bias: bool = False) -> pd.DataFrame:
    """Ramalan dinamis `langkah` periode setelah akhir data estimasi, dalam skala asli."""
    akhir = hasil.fittedvalues.index[-1]
    index_depan = pd.period_range(akhir + 1, periods=langkah, freq=akhir.freq)
    ramalan = hasil.get_forecast(steps=langkah, exog=am.exog_konstanta(index_depan, spek.konstanta))
    rata = np.asarray(ramalan.predicted_mean)
    se = np.asarray(ramalan.se_mean)
    ci = np.asarray(ramalan.conf_int(alpha=alpha))
    pakai_exp = mode_transformasi in ("log", "sudah_log")
    faktor = np.exp(se ** 2 / 2) if (koreksi_bias and pakai_exp) else 1.0
    return pd.DataFrame({
        "ramalan_skala_model": rata,
        "se_skala_model": se,
        "ramalan": ke_skala_asli(rata, mode_transformasi) * faktor,
        "batas_bawah": ke_skala_asli(ci[:, 0], mode_transformasi),
        "batas_atas": ke_skala_asli(ci[:, 1], mode_transformasi),
    }, index=index_depan)


def baseline_naif_musiman(y_model_latih: pd.Series, periode_uji: pd.PeriodIndex, s: int) -> pd.Series:
    """Ramalan = nilai periode yang sama satu musim sebelumnya (berulang bila horizon > s)."""
    nilai = y_model_latih.copy()
    for p in periode_uji:
        nilai.loc[p] = nilai.get(p - s, np.nan)
    return nilai.loc[periode_uji]


def baseline_rata_musiman(y_model_latih: pd.Series, periode_uji: pd.PeriodIndex, s: int) -> pd.Series:
    """Ramalan = rata-rata (di skala model) per posisi musim pada data latih."""
    y = y_model_latih.dropna()
    posisi = pd.Series([p.ordinal % s for p in y.index], index=y.index)
    rata = y.groupby(posisi).mean()
    return pd.Series([rata.get(p.ordinal % s, np.nan) for p in periode_uji], index=periode_uji)


def backtest(y_model: pd.Series, spek: am.SpesifikasiModel, awal_latih, akhir_latih, akhir_uji,
             mode_transformasi: str, alpha: float = 0.05, koreksi_bias: bool = False) -> dict:
    s = spek.seasonal_order[3]
    y_latih = y_model.loc[awal_latih:akhir_latih]
    y_uji = y_model.loc[akhir_latih + 1:akhir_uji]
    hasil = am.estimasi(y_latih, spek)
    ramal = proyeksi(hasil, len(y_uji), spek, mode_transformasi, alpha, koreksi_bias)
    aktual = ke_skala_asli(y_uji, mode_transformasi)
    tabel = pd.DataFrame({
        "aktual": aktual,
        "ramalan": ramal["ramalan"],
        "galat": aktual - ramal["ramalan"],
        "batas_bawah": ramal["batas_bawah"],
        "batas_atas": ramal["batas_atas"],
    })
    pembanding = {
        "Naif musiman": ke_skala_asli(baseline_naif_musiman(y_latih, y_uji.index, s), mode_transformasi),
        "Rata-rata musiman": ke_skala_asli(baseline_rata_musiman(y_latih, y_uji.index, s), mode_transformasi),
    }
    return {"hasil": hasil, "y_latih": y_latih, "tabel": tabel, "pembanding": pembanding}
