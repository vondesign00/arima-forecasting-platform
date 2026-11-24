"""
Uji modul. Jalankan dari folder proyek:  python -m pytest -q
Tes yang membutuhkan statsmodels otomatis dilewati bila statsmodels belum terpasang.
"""
import io
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

AKAR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(AKAR))

from modules import data_loader as dl  # noqa: E402
from modules import data_preprocessing as dp  # noqa: E402
from modules import evaluation as ev  # noqa: E402
from modules import exploration as ex  # noqa: E402
from modules import forecasting as fc  # noqa: E402
from modules.arima_model import SpesifikasiModel, nama_parameter, peringatan_model  # noqa: E402

CONTOH_XLSX = AKAR / "sample_data" / "sample_dataset.xlsx"
CONTOH_CSV = AKAR / "sample_data" / "sample_dataset.csv"


# ------------------------------------------------------------------ 1-2. file valid
@pytest.mark.parametrize("path", [CONTOH_XLSX, CONTOH_CSV])
def test_file_contoh_terbaca(path):
    df = dl.baca_file(str(path))
    assert list(df.columns) == ["tanggal", "outflow"]
    v = dp.siapkan_seri(df, *dl.tebak_kolom(df), "Bulanan")
    assert v.ok and not v.peringatan
    assert len(v.seri) == 66 and str(v.seri.index.freq) in ("<MonthEnd>", "M")


def test_csv_titik_koma_dan_format_indonesia():
    teks = "periode;nilai\n2021M01;1.234.567,5\n2021M02;200.000\n2021M03;300000\n"
    f = io.StringIO(teks)
    f.name = "x.csv"
    df = dl.baca_file(f)
    v = dp.siapkan_seri(df, "periode", "nilai", "Bulanan")
    assert v.seri.tolist() == [1234567.5, 200000.0, 300000.0]


def test_format_file_salah():
    with pytest.raises(dl.GalatData):
        dl.baca_file("data.txt")


# ------------------------------------------------------------------ 3. nilai kosong
def _df(tanggal, nilai):
    return pd.DataFrame({"t": tanggal, "v": nilai})


def test_periode_hilang_dilaporkan_tidak_dihapus():
    per = [f"2021M{m:02d}" for m in range(1, 13) if m != 5]
    v = dp.siapkan_seri(_df(per, np.arange(1.0, 12.0)), "t", "v", "Bulanan")
    assert v.ok
    assert len(v.seri) == 12 and np.isnan(v.seri.loc["2021-05"])
    assert any("tidak ada di data" in p for p in v.peringatan)


def test_nilai_kosong_dilaporkan():
    per = [f"2021M{m:02d}" for m in range(1, 13)]
    nilai = [1.0] * 12
    nilai[3] = None
    v = dp.siapkan_seri(_df(per, nilai), "t", "v", "Bulanan")
    assert v.ok and v.seri.isna().sum() == 1
    assert any("kosong" in p for p in v.peringatan)


# ------------------------------------------------------------------ 4. tanggal tidak valid
def test_tanggal_tidak_valid():
    v = dp.siapkan_seri(_df(["abc", "def"] * 6, range(12)), "t", "v", "Bulanan")
    assert not v.ok and "tidak bisa dibaca" in v.galat[0]


def test_periode_ganda_salah_frekuensi():
    harian = pd.date_range("2021-01-01", periods=40, freq="D")
    v = dp.siapkan_seri(_df(harian, range(40)), "t", "v", "Bulanan")
    assert not v.ok and "periode ganda" in v.galat[0]


def test_tanggal_format_indonesia_hari_dulu():
    v = dp.siapkan_seri(_df([f"01/{m:02d}/2021" for m in range(2, 13)], range(1, 12)), "t", "v", "Bulanan")
    assert v.ok, v.galat
    assert dp.label_periode(v.seri.index[0]) == "2021-02" and len(v.seri) == 11


# ------------------------------------------------------------------ 5. target bukan angka
def test_target_bukan_angka():
    per = [f"2021M{m:02d}" for m in range(1, 13)]
    v = dp.siapkan_seri(_df(per, ["tinggi", "rendah"] * 6), "t", "v", "Bulanan")
    assert not v.ok and "bukan angka" in v.galat[0]


def test_kolom_sama():
    v = dp.siapkan_seri(_df(["2021M01"], [1]), "t", "t", "Bulanan")
    assert not v.ok


# ------------------------------------------------------------------ 6. observasi kurang
def test_observasi_terlalu_sedikit():
    per = [f"2021M{m:02d}" for m in range(1, 6)]
    v = dp.siapkan_seri(_df(per, range(1, 6)), "t", "v", "Bulanan")
    assert not v.ok and "terlalu sedikit" in v.galat[0]


def test_minimal_observasi_model_musiman():
    assert SpesifikasiModel(seasonal_order=(1, 0, 0, 12)).minimal_observasi() >= 18
    assert SpesifikasiModel(order=(1, 0, 0), seasonal_order=(0, 0, 0, 12)).minimal_observasi() == 10


# ------------------------------------------------------------------ 7. nol/negatif dengan log
def test_log_menolak_nol_negatif():
    s = pd.Series([1.0, 0.0, -2.0, 4.0], index=pd.period_range("2021-01", periods=4, freq="M"))
    assert dp.cek_transformasi(s, "log")
    assert not dp.cek_transformasi(s, "tanpa")


def test_log_dan_kebalikannya():
    s = pd.Series([100.0, 200.0, 50.0], index=pd.period_range("2021-01", periods=3, freq="M"))
    for mode in dp.TRANSFORMASI:
        balik = dp.ke_skala_asli(dp.ke_skala_model(s, mode), mode)
        harapan = np.exp(s) if mode == "sudah_log" else s
        np.testing.assert_allclose(balik, harapan)


def test_petunjuk_log_ganda():
    s = pd.Series(np.linspace(11.5, 13.5, 24), index=pd.period_range("2021-01", periods=24, freq="M"))
    assert dp.petunjuk_log_ganda(s, "log")
    assert dp.petunjuk_log_ganda(s * 1e5, "log") is None


# ------------------------------------------------------------------ frekuensi lain
@pytest.mark.parametrize("frek,tanggal,n", [
    ("Triwulanan", ["2020Q1", "2020Q2", "2020Q3", "2020Q4"] * 1 + ["2021Q1", "2021Q2", "2021Q3", "2021Q4",
                                                                     "2022Q1", "2022Q2", "2022Q3"], 11),
    ("Mingguan", pd.date_range("2024-01-01", periods=30, freq="W-MON"), 30),
    ("Harian", pd.date_range("2024-01-01", periods=30, freq="D"), 30),
])
def test_frekuensi_lain(frek, tanggal, n):
    v = dp.siapkan_seri(_df(tanggal, np.arange(1.0, n + 1)), "t", "v", frek)
    assert v.ok, v.galat
    assert len(v.seri) == n
    assert dp.label_periode(v.seri.index[0])


# ------------------------------------------------------------------ evaluasi
def test_metrik_definisi():
    a, f = [100, 200, 150], [110, 180, 150]
    m = ev.ukuran_akurasi(a, f)
    assert m["MAE"] == pytest.approx(10.0)
    assert m["RMSE"] == pytest.approx(np.sqrt((100 + 400 + 0) / 3))
    assert m["MAPE (%)"] == pytest.approx((10 / 100 + 20 / 200 + 0) / 3 * 100)
    assert m["Theil U2"] == pytest.approx(0.1940285, rel=1e-6)


def test_metrik_sama_dengan_kode_acuan():
    sys.path.insert(0, str(AKAR / "reference"))
    import arima_engine as ref
    rng = np.random.default_rng(1)
    a, f = rng.uniform(50, 500, 12), rng.uniform(50, 500, 12)
    baru, lama = ev.ukuran_akurasi(a, f), ref.ukuran_akurasi(a, f)
    for k in lama:
        assert baru[k] == pytest.approx(lama[k], rel=1e-12)


def test_mape_mengecualikan_nol():
    m = ev.ukuran_akurasi([0, 100], [10, 110])
    assert m["MAPE (%)"] == pytest.approx(10.0)
    assert m["Aktual = 0 (dikecualikan dari MAPE)"] == 1
    assert m["MAE"] == pytest.approx(10.0)


# ------------------------------------------------------------------ baseline (sama dengan kode acuan)
def test_baseline_sama_dengan_kode_acuan():
    sys.path.insert(0, str(AKAR / "reference"))
    import arima_engine as ref
    df = dl.baca_file(str(CONTOH_XLSX))
    y = dp.siapkan_seri(df, "tanggal", "outflow").seri
    latih, uji = y.loc[:"2025-06"], y.loc["2025-07":].index
    np.testing.assert_allclose(fc.baseline_naif_musiman(latih, uji, 12), ref.baseline_naif_musiman(latih, uji))
    baru = np.exp(fc.baseline_rata_musiman(np.log(latih), uji, 12))
    np.testing.assert_allclose(baru, ref.baseline_rata_bulanan(latih, uji))


# ------------------------------------------------------------------ peringatan model
def test_peringatan_model_degeneratif():
    params = pd.Series({"C": 12.68, "ar.S.L12": 1.0, "ma.S.L12": -0.999895, "sigma2": 0.4})
    bse = pd.Series({"C": 0.22, "ar.S.L12": 7.39e-5, "ma.S.L12": 9.3e-5, "sigma2": 0.09})
    pesan = peringatan_model(params, bse, np.ones(12), np.ones(12) * 0.99999, 12)
    assert len(pesan) == 4
    params2 = pd.Series({"C": 12.68, "ar.S.L12": 0.448, "sigma2": 0.4})
    assert peringatan_model(params2, params2 * 0 + 0.1, np.full(12, 0.448 ** (1 / 12)), np.array([]), 12) == []


def test_nama_parameter():
    assert [nama_parameter(n) for n in ["ar.S.L12", "ma.L1", "sigma2", "C"]] == ["SAR(12)", "MA(1)", "SIGMASQ", "C"]


# ------------------------------------------------------------------ eksplorasi & PNG
def test_statistik_deskriptif():
    s = pd.Series([1.0, 2.0, None, 4.0], index=pd.period_range("2021-01", periods=4, freq="M"))
    st = ex.statistik_deskriptif(s)
    assert st["Observasi terisi"] == 3 and st["Median"] == 2.0
    assert ex.daftar_kosong(s) == ["2021-03"]


def test_png_dari_hasil_tiruan():
    from modules.visualization import png_hasil
    idx = pd.period_range("2021-01", periods=24, freq="M")
    y = pd.Series(np.arange(24.0) + 10, index=idx)
    uji = idx[18:]
    depan = pd.period_range("2023-01", periods=6, freq="M")
    h = types.SimpleNamespace(
        pengaturan=types.SimpleNamespace(awal_latih=idx[0], akhir_latih=idx[17], akhir_uji=idx[-1],
                                         tingkat_kepercayaan=95, spek=SpesifikasiModel()),
        y_asli=y, akhir_data_proyeksi=idx[-1],
        backtest={"tabel": pd.DataFrame({"aktual": y[uji], "ramalan": y[uji] + 1, "batas_bawah": y[uji] - 2,
                                         "batas_atas": y[uji] + 3}, index=uji)},
        ramalan=pd.DataFrame({"ramalan": np.ones(6) * 30, "batas_bawah": np.ones(6) * 25,
                              "batas_atas": np.ones(6) * 35}, index=depan),
    )
    png = png_hasil(h, "outflow")
    assert png[:8] == b"\x89PNG\r\n\x1a\n" and len(png) > 10_000


# ------------------------------------------------------------------ 8-10. butuh statsmodels
@pytest.fixture
def seri_sar12():
    """Proses SAR(12) sintetis: log y = 12,6 + u, u_t = 0,5 u_{t-12} + e."""
    rng = np.random.default_rng(42)
    n = 240
    u = np.zeros(n + 120)
    e = rng.normal(0, 0.3, n + 120)
    for t in range(12, n + 120):
        u[t] = 0.5 * u[t - 12] + e[t]
    idx = pd.period_range("2000-01", periods=n, freq="M")
    return pd.Series(np.exp(12.6 + u[120:]), index=idx, name="outflow")


def _pengaturan(seri, horizon=12, spek=None, uji=True):
    from modules.pipeline import Pengaturan
    return Pengaturan(spek=spek or SpesifikasiModel(), transformasi="log", awal_latih=seri.index[0],
                      akhir_latih=seri.index[-25], akhir_uji=seri.index[-1] if uji else None, horizon=horizon)


def test_estimasi_memulihkan_parameter(seri_sar12):
    pytest.importorskip("statsmodels")
    from modules.pipeline import jalankan
    h = jalankan(seri_sar12, _pengaturan(seri_sar12))
    k = h.koefisien_akhir["Koefisien"]
    assert k["C"] == pytest.approx(12.6, abs=0.1)
    assert k["SAR(12)"] == pytest.approx(0.5, abs=0.15)
    assert h.akurasi is not None and len(h.backtest["tabel"]) == 24


@pytest.mark.parametrize("horizon", [1, 12, 36])
def test_horizon(seri_sar12, horizon):
    pytest.importorskip("statsmodels")
    from modules.pipeline import jalankan
    h = jalankan(seri_sar12, _pengaturan(seri_sar12, horizon, uji=False))
    assert len(h.ramalan) == horizon and h.backtest is None
    assert h.ramalan.index[0] == seri_sar12.index[-1] + 1
    assert (h.ramalan["batas_bawah"] < h.ramalan["ramalan"]).all()
    assert (h.ramalan["ramalan"] < h.ramalan["batas_atas"]).all()


def test_backtest_tanpa_kebocoran(seri_sar12):
    """Mengubah nilai periode uji tidak boleh mengubah ramalan backtest."""
    pytest.importorskip("statsmodels")
    from modules.pipeline import jalankan
    peng = _pengaturan(seri_sar12)
    a = jalankan(seri_sar12, peng).backtest["tabel"]["ramalan"]
    rusak = seri_sar12.copy()
    rusak.iloc[-24:] *= 5
    b = jalankan(rusak, peng).backtest["tabel"]["ramalan"]
    np.testing.assert_allclose(a, b)


def test_sama_dengan_kode_acuan(seri_sar12):
    pytest.importorskip("statsmodels")
    sys.path.insert(0, str(AKAR / "reference"))
    import arima_engine as ref
    from modules.pipeline import jalankan
    peng = _pengaturan(seri_sar12)
    h = jalankan(seri_sar12, peng)
    spek_ref = ref.SpesifikasiModel(order=(0, 0, 0), seasonal_order=(1, 0, 0, 12))
    bt = ref.backtest(seri_sar12, spek_ref, str(peng.awal_latih), str(peng.akhir_latih), str(peng.akhir_uji))
    np.testing.assert_allclose(h.backtest["tabel"]["ramalan"], bt["tabel"][spek_ref.label()], rtol=1e-8)


def test_model_manual_lain_dan_ekspor(seri_sar12):
    pytest.importorskip("statsmodels")
    from modules import export_results as exr
    from modules.pipeline import jalankan, tabel_hasil
    h = jalankan(seri_sar12, _pengaturan(seri_sar12, spek=SpesifikasiModel(order=(1, 0, 1),
                                                                         seasonal_order=(1, 0, 0, 12))))
    assert {"AR(1)", "MA(1)", "SAR(12)", "C", "SIGMASQ"} <= set(h.koefisien_akhir.index)
    assert len(tabel_hasil(h)) == 24 + 12
    xls = pd.read_excel(io.BytesIO(exr.excel_laporan(h)), sheet_name=None)
    assert {"Pengaturan", "Hasil", "Evaluasi", "Koefisien", "Statistik model"} <= set(xls)
    assert exr.csv_hasil(h).startswith("﻿Periode".encode("utf-8"))
    assert b"MODEL UNTUK PROYEKSI" in exr.teks_ringkasan(h)
