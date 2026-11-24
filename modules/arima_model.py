"""
Estimasi (S)ARIMA dan diagnostiknya — dipindahkan dari reference/arima_engine.py
tanpa mengubah metodologi:

- Konstanta C diestimasi sebagai RATA-RATA (regresi pada konstanta dengan galat
  ARMA), seperti EViews; bukan intercept `trend='c'` statsmodels.
- Maximum Likelihood, kovarians koefisien outer product of gradients (OPG).
- Stasioneritas/invertibilitas tidak dipaksa secara default (seperti EViews).
- Kriteria informasi juga dilaporkan per observasi, seperti EViews.

statsmodels diimpor di dalam fungsi agar modul lain tetap bisa diuji tanpa statsmodels.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class SpesifikasiModel:
    order: tuple[int, int, int] = (0, 0, 0)
    seasonal_order: tuple[int, int, int, int] = (1, 0, 0, 12)
    konstanta: bool = True
    paksa_stasioner: bool = False
    paksa_invertible: bool = False

    def label(self) -> str:
        p, d, q = self.order
        P, D, Q, s = self.seasonal_order
        if P == D == Q == 0:
            teks = f"ARIMA({p},{d},{q})"
        else:
            teks = f"SARIMA({p},{d},{q})({P},{D},{Q}){s}"
        return teks + (" + C" if self.konstanta else "")

    def jumlah_parameter_arma(self) -> int:
        p, _, q = self.order
        P, _, Q, _ = self.seasonal_order
        return p + q + P + Q

    def minimal_observasi(self) -> int:
        """Batas bawah kasar agar model bisa diestimasi."""
        p, d, q = self.order
        P, D, Q, s = self.seasonal_order
        lag_maks = max(p + d, (P + D) * s if (P or D) else 0, q, Q * s if Q else 0)
        return max(10, lag_maks + d + D * s + self.jumlah_parameter_arma() + 5)


def exog_konstanta(index, pakai: bool):
    if not pakai:
        return None
    return pd.DataFrame({"C": np.ones(len(index))}, index=index)


def estimasi(y_model: pd.Series, spek: SpesifikasiModel):
    """Estimasi model pada data skala model (mis. log). Mengembalikan hasil statsmodels."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    model = SARIMAX(
        y_model,
        exog=exog_konstanta(y_model.index, spek.konstanta),
        order=spek.order,
        seasonal_order=spek.seasonal_order,
        trend="n",
        enforce_stationarity=spek.paksa_stasioner,
        enforce_invertibility=spek.paksa_invertible,
    )
    return model.fit(disp=False, maxiter=2000, cov_type="opg")


def nama_parameter(nama: str) -> str:
    """Nama parameter statsmodels -> gaya EViews (ar.S.L12 -> SAR(12))."""
    if nama == "sigma2":
        return "SIGMASQ"
    m = re.match(r"^(ar|ma)\.(S\.)?L(\d+)$", nama)
    if m:
        jenis, musiman, lag = m.groups()
        return f"{'S' if musiman else ''}{jenis.upper()}({lag})"
    return nama


def tabel_koefisien(hasil) -> pd.DataFrame:
    tabel = pd.DataFrame({
        "Koefisien": hasil.params,
        "Std. Error": hasil.bse,
        "t/z-Statistic": hasil.tvalues,
        "Prob.": hasil.pvalues,
    })
    tabel.index = [nama_parameter(n) for n in tabel.index]
    tabel.index.name = "Variabel"
    return tabel


def residual_efektif(hasil) -> pd.Series:
    """Residual tanpa periode awal (burn-in) inisialisasi difus, bila ada."""
    burn = int(getattr(hasil, "loglikelihood_burn", 0) or 0)
    return pd.Series(hasil.resid).iloc[burn:].dropna()


def statistik_ringkas(hasil, y_model: pd.Series) -> pd.Series:
    n = int(hasil.nobs)
    k = len(hasil.params)  # EViews ikut menghitung SIGMASQ
    ll = float(hasil.llf)
    resid = residual_efektif(hasil)
    y_sejajar = y_model.reindex(resid.index)
    ssr = float((resid ** 2).sum())
    sst = float(((y_sejajar - y_sejajar.mean()) ** 2).sum())
    return pd.Series({
        "Jumlah observasi": n,
        "Log likelihood": ll,
        "AIC (statsmodels, total)": float(hasil.aic),
        "BIC (statsmodels, total)": float(hasil.bic),
        "Akaike info criterion (per obs., EViews)": (-2 * ll + 2 * k) / n,
        "Schwarz criterion (per obs., EViews)": (-2 * ll + k * np.log(n)) / n,
        "Hannan-Quinn criter. (per obs., EViews)": (-2 * ll + 2 * k * np.log(np.log(n))) / n,
        "R-squared (perkiraan)": 1 - ssr / sst if sst > 0 else np.nan,
        "Sum squared resid": ssr,
        "Observasi burn-in (tidak dihitung di residual)": int(getattr(hasil, "loglikelihood_burn", 0) or 0),
    })


def diagnostik(hasil, spek: SpesifikasiModel) -> dict:
    from statsmodels.stats.diagnostic import acorr_ljungbox
    from statsmodels.tsa.stattools import acf

    resid = residual_efektif(hasil)
    s = spek.seasonal_order[3]
    df_model = spek.jumlah_parameter_arma()
    lag_uji = sorted({lag for lag in (s, 2 * s, 12, 24) if df_model < lag < len(resid)})
    ljung_box = acorr_ljungbox(resid, lags=lag_uji, model_df=df_model) if lag_uji else pd.DataFrame()

    jb = stats.jarque_bera(resid)
    nlags = max(1, min(2 * s, 24, len(resid) - 1))
    nilai_acf = acf(resid, nlags=nlags, fft=False, missing="drop")[1:]

    akar_ar = 1 / np.abs(hasil.arroots) if len(hasil.arroots) else np.array([])
    akar_ma = 1 / np.abs(hasil.maroots) if len(hasil.maroots) else np.array([])

    return {
        "ljung_box": ljung_box,
        "jarque_bera": {"statistik": float(jb.statistic), "p_value": float(jb.pvalue)},
        "acf": pd.DataFrame({"lag": np.arange(1, nlags + 1), "acf": nilai_acf}),
        "batas_acf": 1.96 / np.sqrt(len(resid)),
        "akar_ar_invers_maks": float(akar_ar.max()) if len(akar_ar) else None,
        "akar_ma_invers_maks": float(akar_ma.max()) if len(akar_ma) else None,
        "peringatan": peringatan_model(hasil.params, hasil.bse, akar_ar, akar_ma, s),
    }


def peringatan_model(params: pd.Series, bse: pd.Series, akar_ar, akar_ma, s: int,
                     ambang: float = 0.99) -> list[str]:
    """Deteksi model yang tidak bisa ditafsirkan seperti ARIMA biasa."""
    pesan = []
    if len(akar_ar) and np.max(akar_ar) >= ambang:
        pesan.append(f"Akar AR invers = {np.max(akar_ar):.4f} (≥ {ambang}): komponen AR di/dekat unit root, "
                     "model tidak stasioner.")
    if len(akar_ma) and np.max(akar_ma) >= ambang:
        pesan.append(f"Akar MA invers = {np.max(akar_ma):.4f} (≥ {ambang}): komponen MA tidak/nyaris tidak invertible.")
    for ar, ma in ((f"ar.S.L{s}", f"ma.S.L{s}"), ("ar.L1", "ma.L1")):
        if ar in params.index and ma in params.index:
            phi, theta = params[ar], params[ma]
            if abs(phi + theta) < 0.05 and abs(phi) > 0.5:
                pesan.append(f"{nama_parameter(ar)} = {phi:.4f} dan {nama_parameter(ma)} = {theta:.4f} hampir saling "
                             "meniadakan (pembatalan akar); t-statistik yang sangat besar tidak bermakna.")
    if (bse.drop("sigma2", errors="ignore") < 1e-3).any():
        pesan.append("Ada standard error sangat kecil (< 0,001): estimasi berada di batas ruang parameter.")
    return pesan


def ringkasan_teks(hasil, spek: SpesifikasiModel, stat: pd.Series) -> str:
    """Ringkasan model sebagai teks (untuk diunduh)."""
    garis = [f"Model: {spek.label()}", ""]
    garis.append(tabel_koefisien(hasil).to_string(float_format=lambda v: f"{v:.6f}"))
    garis += ["", stat.to_string(float_format=lambda v: f"{v:.6f}"), "", "--- statsmodels summary ---",
              str(hasil.summary())]
    return "\n".join(garis)
