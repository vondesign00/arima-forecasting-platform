"""
Membentuk deret waktu dari tabel, memeriksa masalah data, dan transformasi log.

Aturan: data tidak pernah dihapus atau diubah diam-diam. Setiap periode yang
hilang, nilai kosong, atau nilai yang tidak bisa dibaca dilaporkan ke pengguna.
"""
from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# nama tampilan -> (kode periode pandas, panjang musim bawaan)
FREKUENSI = {
    "Bulanan": ("M", 12),
    "Triwulanan": ("Q", 4),
    "Mingguan": ("W", 52),
    "Harian": ("D", 7),
}

TRANSFORMASI = {
    "log": "Terapkan log natural (data diunggah dalam nilai asli)",
    "sudah_log": "Data sudah dalam bentuk log (jangan di-log lagi)",
    "tanpa": "Tanpa transformasi",
}

_POLA_BULANAN = re.compile(r"^\s*(\d{4})\s*[Mm]\s*(\d{1,2})\s*$")   # 2021M01
_POLA_TRIWULAN = re.compile(r"^\s*(\d{4})\s*[Qq]\s*([1-4])\s*$")   # 2021Q1


def ke_periode(kolom: pd.Series, kode_frek: str) -> pd.PeriodIndex:
    """Ubah kolom tanggal ke PeriodIndex. Mendukung format EViews (2021M01, 2021Q1)."""
    teks = kolom.astype(str).str.strip()
    if kode_frek == "M" and teks.str.match(_POLA_BULANAN).all():
        b = teks.str.extract(_POLA_BULANAN)
        return pd.PeriodIndex(b[0] + "-" + b[1].str.zfill(2), freq="M")
    if kode_frek == "Q" and teks.str.match(_POLA_TRIWULAN).all():
        b = teks.str.extract(_POLA_TRIWULAN)
        return pd.PeriodIndex(b[0] + "Q" + b[1], freq="Q")
    if pd.api.types.is_datetime64_any_dtype(kolom):
        tanggal = pd.DatetimeIndex(kolom)
    else:
        # Teks tanggal: format Indonesia hari/bulan/tahun (01/02/2021 = 1 Februari). ISO (2021-02-01) tetap benar.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            tanggal = pd.DatetimeIndex(pd.to_datetime(kolom, errors="raise", dayfirst=True))
    return tanggal.to_period(kode_frek)


def label_periode(p: pd.Period) -> str:
    """Label ringkas untuk tampilan: 2021-01, 2021Q1, atau tanggal awal minggu/hari."""
    kode = p.freqstr[0]
    if kode == "M":
        return p.strftime("%Y-%m")
    if kode == "Q":
        return f"{p.year}Q{p.quarter}"
    return p.start_time.strftime("%Y-%m-%d")


def ke_angka(kolom: pd.Series) -> pd.Series:
    """Ubah teks ke angka. Mengenali format Indonesia (1.234.567,89)."""
    if pd.api.types.is_numeric_dtype(kolom):
        return kolom.astype(float)
    teks = kolom.astype(str).str.strip().replace({"": np.nan, "nan": np.nan, "None": np.nan})
    format_id = teks.str.contains(r",\d+$", na=False) | teks.str.contains(r"\.\d{3}(?:\.|$)", na=False)
    teks = teks.where(~format_id, teks.str.replace(".", "", regex=False).str.replace(",", ".", regex=False))
    return pd.to_numeric(teks, errors="coerce")


@dataclass
class HasilValidasi:
    seri: pd.Series
    galat: list[str] = field(default_factory=list)
    peringatan: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.galat


def siapkan_seri(df: pd.DataFrame, kolom_tanggal: str, kolom_target: str,
                 frekuensi: str = "Bulanan") -> HasilValidasi:
    """Bentuk deret waktu reguler dan laporkan semua masalah data."""
    kode, _ = FREKUENSI[frekuensi]
    galat, peringatan = [], []
    if kolom_tanggal == kolom_target:
        return HasilValidasi(pd.Series(dtype=float), ["Kolom tanggal dan kolom target tidak boleh sama."])

    tanggal_kosong = df[kolom_tanggal].isna()
    if tanggal_kosong.any():
        peringatan.append(f"{int(tanggal_kosong.sum())} baris tanpa tanggal tidak diikutkan.")
    df = df.loc[~tanggal_kosong]

    try:
        periode = ke_periode(df[kolom_tanggal], kode)
    except Exception:
        contoh = ", ".join(df[kolom_tanggal].astype(str).head(3))
        return HasilValidasi(pd.Series(dtype=float), [
            f"Kolom tanggal '{kolom_tanggal}' tidak bisa dibaca sebagai tanggal (contoh isi: {contoh}). "
            "Gunakan format seperti 2021-01-31, 31/01/2021, 2021M01, atau 2021Q1."])

    asli = df[kolom_target]
    nilai = ke_angka(asli)
    tak_terbaca = nilai.isna() & asli.notna()
    if len(nilai) and tak_terbaca.mean() > 0.5:
        return HasilValidasi(pd.Series(dtype=float), [
            f"Kolom '{kolom_target}' sebagian besar bukan angka. Pilih kolom target yang berisi nilai numerik."])
    if tak_terbaca.any():
        peringatan.append(f"{int(tak_terbaca.sum())} nilai di kolom target bukan angka dan dianggap kosong.")

    seri = pd.Series(nilai.to_numpy(), index=periode, name=kolom_target).sort_index()

    if seri.index.duplicated().any():
        dup = sorted({label_periode(p) for p in seri.index[seri.index.duplicated()]})
        galat.append(
            f"Ada {len(dup)} periode ganda untuk frekuensi {frekuensi.lower()} (misalnya {', '.join(dup[:5])}). "
            "Periksa apakah frekuensi yang dipilih sudah benar, atau agregasikan data terlebih dahulu.")
        return HasilValidasi(seri, galat, peringatan)

    lengkap = pd.period_range(seri.index.min(), seri.index.max(), freq=seri.index.freq)
    hilang = lengkap.difference(seri.index)
    if len(hilang):
        peringatan.append(
            f"{len(hilang)} periode tidak ada di data ({', '.join(label_periode(p) for p in hilang[:6])}"
            f"{' ...' if len(hilang) > 6 else ''}). Periode ini dibiarkan kosong; model tidak mengisinya dengan nilai karangan.")
    seri = seri.reindex(lengkap)

    kosong = seri.isna() & ~seri.index.isin(hilang)
    if kosong.any() and not tak_terbaca.any():
        peringatan.append(f"{int(kosong.sum())} nilai target kosong.")

    if seri.notna().sum() < 10:
        galat.append("Data terlalu sedikit (kurang dari 10 observasi terisi) untuk model ARIMA.")
    return HasilValidasi(seri, galat, peringatan)


# --------------------------------------------------------------------------- transformasi

def cek_transformasi(seri: pd.Series, mode: str) -> list[str]:
    """Galat yang mencegah transformasi dijalankan."""
    if mode == "log":
        buruk = seri.dropna() <= 0
        if buruk.any():
            contoh = ", ".join(label_periode(p) for p in buruk[buruk].index[:6])
            return [f"Transformasi log membutuhkan nilai > 0. Ada {int(buruk.sum())} nilai ≤ 0 (misalnya {contoh}). "
                    "Pilih 'Tanpa transformasi' atau perbaiki datanya."]
    return []


def petunjuk_log_ganda(seri: pd.Series, mode: str) -> str | None:
    """Peringatan bila data tampaknya sudah dalam log tetapi akan di-log lagi."""
    s = seri.dropna()
    if mode == "log" and len(s) and s.min() > 0 and s.max() < 30 and (s.max() / s.min()) < 3:
        return ("Nilai data kecil dan rentangnya sempit (seperti data yang sudah di-log). Jika kolom ini sudah "
                "berupa log, pilih 'Data sudah dalam bentuk log' agar tidak di-log dua kali.")
    return None


def ke_skala_model(seri: pd.Series, mode: str) -> pd.Series:
    return np.log(seri) if mode == "log" else seri.astype(float)


def ke_skala_asli(x, mode: str):
    """Kembalikan dari skala model ke skala asli data."""
    return np.exp(x) if mode in ("log", "sudah_log") else x


def potong(seri: pd.Series, awal: pd.Period, akhir: pd.Period) -> pd.Series:
    return seri.loc[awal:akhir]
