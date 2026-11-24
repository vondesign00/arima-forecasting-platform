"""
Mesin proyeksi SARIMA untuk outflow (dimodelkan dalam log), disusun agar
sedekat mungkin dengan pengaturan EViews:

- Y = log(outflow)
- Konstanta "C" diperlakukan seperti EViews: sebagai RATA-RATA (regresi pada
  konstanta dengan galat ARMA), bukan intercept persamaan AR.
- Estimasi Maximum Likelihood, kovarians koefisien dengan outer product of
  gradients (OPG) seperti keluaran EViews.
- Kriteria informasi (AIC/SC/HQ) dilaporkan per observasi, seperti EViews.

Modul ini tidak bergantung pada Streamlit, jadi bisa dipakai dari skrip biasa.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# 1. Membaca dan memvalidasi data
# ---------------------------------------------------------------------------

_POLA_EVIEWS = re.compile(r"^\s*(\d{4})\s*[Mm]\s*(\d{1,2})\s*$")  # contoh: 2021M01


def baca_file(file) -> pd.DataFrame:
    """Baca CSV atau Excel. `file` boleh path (str) atau objek unggahan Streamlit."""
    nama = getattr(file, "name", str(file)).lower()
    if nama.endswith((".xlsx", ".xls")):
        return pd.read_excel(file)
    # CSV: coba deteksi pemisah (koma atau titik koma)
    return pd.read_csv(file, sep=None, engine="python")


def ke_periode_bulanan(kolom: pd.Series) -> pd.PeriodIndex:
    """Ubah kolom tanggal (2021M01, 2021-01, 01/01/2021, tanggal Excel) ke periode bulanan."""
    teks = kolom.astype(str).str.strip()
    if teks.str.match(_POLA_EVIEWS).all():
        bagian = teks.str.extract(_POLA_EVIEWS)
        label = bagian[0] + "-" + bagian[1].str.zfill(2)
        return pd.PeriodIndex(label, freq="M")
    tanggal = pd.to_datetime(kolom, errors="raise")
    return pd.PeriodIndex(tanggal, freq="M")


def _ke_angka(kolom: pd.Series) -> pd.Series:
    """Ubah teks angka ke float. Menangani format Indonesia (1.234.567,89)."""
    if pd.api.types.is_numeric_dtype(kolom):
        return kolom.astype(float)
    teks = kolom.astype(str).str.strip()
    format_id = teks.str.contains(r",\d+$") | teks.str.contains(r"\.\d{3}(?:\.|$)")
    teks = teks.where(~format_id, teks.str.replace(".", "", regex=False).str.replace(",", ".", regex=False))
    return pd.to_numeric(teks, errors="coerce")


@dataclass
class HasilValidasi:
    seri: pd.Series                       # seri bulanan (level, bukan log)
    galat: list[str] = field(default_factory=list)       # menghentikan proses
    peringatan: list[str] = field(default_factory=list)  # boleh lanjut

    @property
    def ok(self) -> bool:
        return not self.galat


def siapkan_seri(df: pd.DataFrame, kolom_tanggal: str, kolom_nilai: str) -> HasilValidasi:
    """Bentuk seri bulanan dan periksa masalah umum pada data."""
    galat, peringatan = [], []
    try:
        periode = ke_periode_bulanan(df[kolom_tanggal])
    except Exception as e:  # format tanggal tidak dikenali
        return HasilValidasi(pd.Series(dtype=float), [f"Kolom tanggal tidak terbaca: {e}"])

    nilai = _ke_angka(df[kolom_nilai])
    seri = pd.Series(nilai.to_numpy(), index=periode, name=kolom_nilai).sort_index()

    if seri.index.duplicated().any():
        dup = sorted({str(p) for p in seri.index[seri.index.duplicated()]})
        galat.append(f"Ada bulan ganda: {', '.join(dup)}")
        seri = seri[~seri.index.duplicated(keep="first")]

    lengkap = pd.period_range(seri.index.min(), seri.index.max(), freq="M")
    hilang = lengkap.difference(seri.index)
    if len(hilang):
        peringatan.append(
            f"Ada {len(hilang)} bulan yang tidak ada di data: "
            + ", ".join(str(p) for p in hilang[:12]) + (" ..." if len(hilang) > 12 else "")
            + ". Bulan ini dianggap kosong (missing)."
        )
    seri = seri.reindex(lengkap)

    kosong = seri.isna() & ~seri.index.isin(hilang)
    if kosong.any():
        peringatan.append(f"Ada {int(kosong.sum())} nilai kosong/tidak terbaca sebagai angka.")

    tidak_positif = seri <= 0
    if tidak_positif.any():
        bulan = ", ".join(str(p) for p in seri.index[tidak_positif][:12])
        galat.append(f"Nilai harus > 0 karena akan di-log. Bermasalah di: {bulan}")

    if seri.notna().sum() < 36:
        peringatan.append("Data kurang dari 36 bulan; pola musiman 12 bulan sulit diestimasi dengan andal.")

    return HasilValidasi(seri, galat, peringatan)


# ---------------------------------------------------------------------------
# 2. Spesifikasi dan estimasi model
# ---------------------------------------------------------------------------

@dataclass
class SpesifikasiModel:
    order: tuple[int, int, int] = (0, 0, 0)
    seasonal_order: tuple[int, int, int, int] = (1, 0, 1, 12)
    konstanta: bool = True
    # EViews tidak memaksa stasioner/invertible, jadi default False agar
    # hasilnya bisa dibandingkan. Ubah ke True untuk estimasi yang "aman".
    paksa_stasioner: bool = False
    paksa_invertible: bool = False

    def label(self) -> str:
        p, d, q = self.order
        P, D, Q, s = self.seasonal_order
        return f"SARIMA({p},{d},{q})({P},{D},{Q}){s}" + (" + C" if self.konstanta else "")


def _exog_konstanta(index, pakai: bool):
    if not pakai:
        return None
    return pd.DataFrame({"C": np.ones(len(index))}, index=index)


def estimasi(log_y: pd.Series, spek: SpesifikasiModel):
    """Estimasi SARIMA dengan ML. Mengembalikan objek hasil statsmodels."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX  # impor di sini agar modul tetap bisa dimuat tanpa statsmodels

    model = SARIMAX(
        log_y,
        exog=_exog_konstanta(log_y.index, spek.konstanta),
        order=spek.order,
        seasonal_order=spek.seasonal_order,
        trend="n",
        enforce_stationarity=spek.paksa_stasioner,
        enforce_invertibility=spek.paksa_invertible,
    )
    return model.fit(disp=False, maxiter=2000, cov_type="opg")


_NAMA_EVIEWS = {"sigma2": "SIGMASQ"}


def _nama_parameter(nama: str) -> str:
    if nama in _NAMA_EVIEWS:
        return _NAMA_EVIEWS[nama]
    m = re.match(r"^(ar|ma)\.(S\.)?L(\d+)$", nama)
    if m:
        jenis, musiman, lag = m.groups()
        return f"{'S' if musiman else ''}{jenis.upper()}({lag})"
    return nama


def tabel_koefisien(hasil) -> pd.DataFrame:
    tabel = pd.DataFrame({
        "Koefisien": hasil.params,
        "Std. Error": hasil.bse,
        "z-Statistic": hasil.tvalues,
        "Prob.": hasil.pvalues,
    })
    tabel.index = [_nama_parameter(n) for n in tabel.index]
    tabel.index.name = "Variabel"
    return tabel


def _residual_efektif(hasil) -> pd.Series:
    """Residual tanpa periode awal (burn-in) inisialisasi difus, jika ada."""
    burn = int(getattr(hasil, "loglikelihood_burn", 0) or 0)
    return pd.Series(hasil.resid).iloc[burn:].dropna()


def statistik_ringkas(hasil, log_y: pd.Series) -> pd.Series:
    """Statistik ringkas bergaya EViews (kriteria informasi per observasi)."""
    n = int(hasil.nobs)
    k = len(hasil.params)  # EViews ikut menghitung SIGMASQ
    ll = float(hasil.llf)
    resid = _residual_efektif(hasil)
    y_sejajar = log_y.loc[resid.index] if isinstance(resid.index, pd.PeriodIndex) else log_y.iloc[-len(resid):]
    ssr = float((resid ** 2).sum())
    sst = float(((y_sejajar - y_sejajar.mean()) ** 2).sum())
    return pd.Series({
        "Jumlah observasi": n,
        "Log likelihood": ll,
        "Akaike info criterion": (-2 * ll + 2 * k) / n,
        "Schwarz criterion": (-2 * ll + k * np.log(n)) / n,
        "Hannan-Quinn criter.": (-2 * ll + 2 * k * np.log(np.log(n))) / n,
        "R-squared (perkiraan)": 1 - ssr / sst if sst > 0 else np.nan,
        "Sum squared resid": ssr,
        "Observasi burn-in (dikecualikan dari residual)": int(getattr(hasil, "loglikelihood_burn", 0) or 0),
    })


# ---------------------------------------------------------------------------
# 3. Diagnostik
# ---------------------------------------------------------------------------

def diagnostik(hasil, spek: SpesifikasiModel) -> dict:
    from statsmodels.stats.diagnostic import acorr_ljungbox
    from statsmodels.tsa.stattools import acf

    resid = _residual_efektif(hasil)
    p, _, q = spek.order
    P, _, Q, _ = spek.seasonal_order
    df_model = p + q + P + Q
    lag_uji = [lag for lag in (12, 24) if lag < len(resid)]
    ljung_box = acorr_ljungbox(resid, lags=lag_uji, model_df=df_model) if lag_uji else pd.DataFrame()

    jb = stats.jarque_bera(resid)
    nlags = min(24, len(resid) - 1)
    nilai_acf = acf(resid, nlags=nlags, fft=False)[1:]
    batas_acf = 1.96 / np.sqrt(len(resid))

    akar_ar = 1 / np.abs(hasil.arroots) if len(hasil.arroots) else np.array([])
    akar_ma = 1 / np.abs(hasil.maroots) if len(hasil.maroots) else np.array([])

    return {
        "ljung_box": ljung_box,
        "jarque_bera": {"statistik": float(jb.statistic), "p_value": float(jb.pvalue)},
        "acf": pd.DataFrame({"lag": np.arange(1, nlags + 1), "acf": nilai_acf}),
        "batas_acf": batas_acf,
        "modulus_akar_ar_invers": akar_ar,
        "modulus_akar_ma_invers": akar_ma,
        "peringatan": peringatan_model(hasil, akar_ar, akar_ma),
    }


def peringatan_model(hasil, akar_ar, akar_ma, ambang: float = 0.99) -> list[str]:
    """Deteksi masalah yang membuat model tidak bisa ditafsirkan seperti ARIMA biasa."""
    pesan = []
    if len(akar_ar) and akar_ar.max() >= ambang:
        pesan.append(
            f"Akar AR invers = {akar_ar.max():.4f} (≥ {ambang}). Komponen AR berada di/dekat unit root, "
            "sehingga model tidak stasioner. Nilai C (rata-rata) praktis tidak teridentifikasi."
        )
    if len(akar_ma) and akar_ma.max() >= ambang:
        pesan.append(
            f"Akar MA invers = {akar_ma.max():.4f} (≥ {ambang}). Komponen MA tidak/nyaris tidak invertible."
        )
    params = hasil.params
    for ar, ma in (("ar.S.L12", "ma.S.L12"), ("ar.L1", "ma.L1")):
        if ar in params.index and ma in params.index:
            phi, theta = params[ar], params[ma]
            if abs(phi + theta) < 0.05 and abs(phi) > 0.5:
                pesan.append(
                    f"{_nama_parameter(ar)} = {phi:.4f} dan {_nama_parameter(ma)} = {theta:.4f} hampir saling "
                    "meniadakan (pembatalan akar). Model ini pada dasarnya setara dengan pola musiman "
                    "deterministik (rata-rata per bulan) + white noise; t-statistik yang sangat besar "
                    "tidak bermakna."
                )
    se = hasil.bse
    if (se.drop("sigma2", errors="ignore") < 1e-3).any():
        pesan.append("Ada standard error yang sangat kecil (< 0,001): tanda estimasi berada di batas ruang parameter.")
    return pesan


# ---------------------------------------------------------------------------
# 4. Ukuran akurasi dan baseline
# ---------------------------------------------------------------------------

def ukuran_akurasi(aktual, ramalan) -> dict:
    a = np.asarray(aktual, dtype=float)
    f = np.asarray(ramalan, dtype=float)
    galat = a - f
    hasil = {
        "MAPE (%)": float(np.mean(np.abs(galat / a)) * 100),
        "sMAPE (%)": float(np.mean(np.abs(galat) / ((np.abs(a) + np.abs(f)) / 2)) * 100),
        "MAE": float(np.mean(np.abs(galat))),
        "RMSE": float(np.sqrt(np.mean(galat ** 2))),
    }
    if len(a) > 1:
        pembilang = np.sum(((f[1:] - a[1:]) / a[:-1]) ** 2)
        penyebut = np.sum(((a[1:] - a[:-1]) / a[:-1]) ** 2)
        hasil["Theil U2"] = float(np.sqrt(pembilang / penyebut)) if penyebut > 0 else np.nan
    return hasil


def baseline_naif_musiman(y_latih: pd.Series, periode_uji: pd.PeriodIndex) -> pd.Series:
    """Ramalan = nilai bulan yang sama setahun sebelumnya (hanya dari data latih)."""
    gabung = y_latih.copy()
    for p in periode_uji:
        gabung.loc[p] = gabung.get(p - 12, np.nan)
    return gabung.loc[periode_uji].rename("Naif musiman")


def baseline_rata_bulanan(y_latih: pd.Series, periode_uji: pd.PeriodIndex) -> pd.Series:
    """Ramalan = rata-rata geometrik (rata-rata log) per bulan kalender di data latih."""
    rata = np.log(y_latih.dropna()).groupby(y_latih.dropna().index.month).mean()
    return pd.Series(np.exp(rata.reindex(periode_uji.month).to_numpy()), index=periode_uji,
                     name="Rata-rata per bulan")


# ---------------------------------------------------------------------------
# 5. Proyeksi dan backtest
# ---------------------------------------------------------------------------

def proyeksi(hasil, langkah: int, spek: SpesifikasiModel, alpha: float = 0.05,
             koreksi_bias: bool = False) -> pd.DataFrame:
    """Proyeksi dinamis `langkah` bulan ke depan, dikembalikan ke skala Rupiah (exp)."""
    akhir = pd.Period(hasil.fittedvalues.index[-1], "M")
    index_depan = pd.period_range(akhir + 1, periods=langkah, freq="M")
    ramalan = hasil.get_forecast(steps=langkah, exog=_exog_konstanta(index_depan, spek.konstanta))
    rata_log = np.asarray(ramalan.predicted_mean)
    se_log = np.asarray(ramalan.se_mean)
    ci = np.asarray(ramalan.conf_int(alpha=alpha))
    faktor = np.exp(se_log ** 2 / 2) if koreksi_bias else 1.0
    return pd.DataFrame({
        "log_proyeksi": rata_log,
        "se_log": se_log,
        "proyeksi": np.exp(rata_log) * faktor,
        "batas_bawah": np.exp(ci[:, 0]),
        "batas_atas": np.exp(ci[:, 1]),
    }, index=index_depan)


def potong(seri: pd.Series, awal, akhir) -> pd.Series:
    return seri.loc[pd.Period(awal, "M"):pd.Period(akhir, "M")]


def backtest(y: pd.Series, spek: SpesifikasiModel, awal_latih, akhir_latih, akhir_uji,
             alpha: float = 0.05, koreksi_bias: bool = False) -> dict:
    """Estimasi di data latih, ramal dinamis sepanjang periode uji, bandingkan dengan aktual."""
    y_latih = potong(y, awal_latih, akhir_latih)
    y_uji = potong(y, pd.Period(akhir_latih, "M") + 1, akhir_uji)
    hasil = estimasi(np.log(y_latih), spek)
    ramal = proyeksi(hasil, len(y_uji), spek, alpha, koreksi_bias)

    tabel = pd.DataFrame({
        "aktual": y_uji,
        spek.label(): ramal["proyeksi"],
        "batas_bawah": ramal["batas_bawah"],
        "batas_atas": ramal["batas_atas"],
        "Naif musiman": baseline_naif_musiman(y_latih, y_uji.index),
        "Rata-rata per bulan": baseline_rata_bulanan(y_latih, y_uji.index),
    })
    sah = tabel["aktual"].notna()
    akurasi = pd.DataFrame({
        nama: ukuran_akurasi(tabel.loc[sah, "aktual"], tabel.loc[sah, nama])
        for nama in (spek.label(), "Naif musiman", "Rata-rata per bulan")
    }).T
    return {"hasil": hasil, "tabel": tabel, "akurasi": akurasi}
