# ARIMA Forecasting Platform

*Platform interaktif untuk proyeksi data deret waktu.*

Aplikasi web (Streamlit) untuk membuat proyeksi dengan model ARIMA/SARIMA tanpa menulis kode: unggah data → eksplorasi → atur model → lihat hasil → evaluasi → unduh.

Model bawaan mengikuti hasil identifikasi EViews: **log(outflow), SARIMA(0,0,0)(1,0,0)12 + C**, dengan C diestimasi sebagai rata-rata, sama seperti di EViews.

> **Status validasi (24 Nov 2026):** 34 uji otomatis lulus di Google Colab. Hasil **belum sama dengan EViews**
> (SAR(12) cocok, tetapi C, AIC/SC, dan metrik akurasi berbeda). Dugaan penyebab: 12 observasi awal tidak
> dihitung di likelihood saat opsi *Paksa stasioner* mati. Angka dari aplikasi ini belum untuk dikutip sebagai hasil resmi.

Versi Google Colab (tanpa instalasi): `notebook/SARIMA_Forecasting_Colab.ipynb`.

---

## A. Menjalankan di komputer sendiri (localhost)

### 1. Instal Python (sekali saja)
1. Buka <https://www.python.org/downloads/> dan unduh **Python 3.12**.
2. Saat instalasi di Windows, **centang "Add python.exe to PATH"**, lalu klik *Install Now*.
3. Periksa instalasi: buka **Command Prompt** (tekan tombol Windows, ketik `cmd`), lalu ketik:
   ```
   python --version
   ```
   Hasilnya harus seperti `Python 3.12.x`.

*(Opsional)* Instal **Visual Studio Code** dari <https://code.visualstudio.com> agar lebih mudah membuka folder dan terminal.

### 2. Siapkan folder proyek
Ekstrak file zip ke folder yang mudah ditemukan, misalnya `C:\Proyek\arima-forecasting-platform`.

Lalu masuk ke folder itu dari Command Prompt:
```
cd C:\Proyek\arima-forecasting-platform
```
*(`cd` = pindah folder. Di VS Code: File → Open Folder, lalu Terminal → New Terminal; terminal sudah otomatis berada di folder proyek.)*

### 3. Buat lingkungan virtual (sekali saja)
Lingkungan virtual adalah "ruang" khusus untuk paket-paket proyek ini, supaya tidak tercampur dengan proyek lain.
```
python -m venv .venv
```
Aktifkan (lakukan **setiap kali** membuka terminal baru):
- Windows: `.venv\Scripts\activate`
- Mac/Linux: `source .venv/bin/activate`

Kalau berhasil, muncul tulisan `(.venv)` di awal baris terminal.

### 4. Instal paket yang dibutuhkan (sekali saja)
```
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```
*(Mengunduh Streamlit, statsmodels, pandas, dan lain-lain. Butuh internet, sekitar 2–5 menit.)*

### 5. Jalankan aplikasi
```
python -m streamlit run app.py
```
Browser akan terbuka otomatis di **http://localhost:8501**. Kalau tidak, buka alamat itu secara manual.

Untuk menghentikan aplikasi: kembali ke terminal lalu tekan **Ctrl + C**.

### 6. Pemakaian berikutnya
Cukup tiga langkah: `cd` ke folder proyek → aktifkan `.venv` → `python -m streamlit run app.py`.

---

## B. Validasi terhadap EViews dan kode acuan

Jalankan dengan **data asli** yang dipakai di EViews:
```
python validasi/cek_kesesuaian.py "C:\Proyek\data\data_outflow.xlsx" --tanggal tanggal --target outflow
```
Ganti `tanggal` dan `outflow` dengan nama kolom di file Anda. Skrip ini menampilkan dua perbandingan:
1. **Aplikasi vs EViews** untuk C, SAR(12), AIC, SC, MAPE, sMAPE, MAE, RMSE, dan Theil U2 (status SESUAI/BEDA).
2. **Aplikasi vs kode acuan** (`reference/arima_engine.py`, mesin di notebook Colab sebelumnya) dengan data dan pengaturan yang sama.

Uji otomatis modul:
```
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

---

## C. Mengatasi masalah umum

| Pesan / gejala | Penyebab & solusi |
|---|---|
| `'python' is not recognized` | Python belum masuk PATH. Instal ulang dan centang **Add python.exe to PATH**, atau coba perintah `py` sebagai ganti `python`. |
| `running scripts is disabled` saat aktivasi di PowerShell | Jalankan sekali: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, atau pakai Command Prompt. |
| `No module named streamlit` / `statsmodels` | Lingkungan virtual belum aktif (tidak ada tulisan `(.venv)`), atau langkah 4 belum dijalankan. |
| Port 8501 sudah dipakai | Aplikasi lain masih berjalan. Tutup terminal lama atau jalankan `python -m streamlit run app.py --server.port 8502`. |
| Gagal instal paket di Python versi terbaru | Gunakan Python 3.11 atau 3.12. |
| Error `statsmodels` terkait pandas | Coba `python -m pip install "pandas<3"`. |
| Kolom tanggal tidak terbaca | Gunakan satu format dalam satu kolom, misalnya `2021-01-31`, `31/01/2021`, `2021M01`, atau tanggal Excel. |
| "Estimasi gagal" | Sederhanakan model, periksa nilai kosong, atau aktifkan *Paksa model stasioner* di Opsi lanjutan. Detail teknis tersedia di bawah pesan galat. |

---

## D. Struktur proyek

```
arima-forecasting-platform/
├── app.py                  # navigasi halaman
├── halaman/                # tampilan (UI) setiap halaman
├── modules/                # perhitungan, terpisah dari UI
│   ├── data_loader.py      # baca CSV/Excel, ringkasan
│   ├── data_preprocessing.py  # tanggal & frekuensi, validasi, log
│   ├── exploration.py      # statistik deskriptif, dekomposisi
│   ├── arima_model.py      # estimasi, koefisien, statistik, diagnostik
│   ├── forecasting.py      # backtest & proyeksi
│   ├── evaluation.py       # MAE, RMSE, MAPE, sMAPE, Theil U2
│   ├── pipeline.py         # menjalankan seluruh alur
│   ├── visualization.py    # grafik Plotly & PNG Matplotlib
│   └── export_results.py   # Excel, CSV, TXT
├── tests/                  # uji otomatis
├── validasi/               # perbandingan dengan EViews & kode acuan
├── reference/              # kode/notebook acuan (tidak dipakai aplikasi)
├── sample_data/            # data contoh SINTETIS
└── data/                   # tempat data asli Anda (tidak ikut ke GitHub)
```

## E. Metodologi (tidak diubah dari kode acuan)

- Transformasi log opsional. Hasil selalu dikembalikan ke skala asli dengan `exp()`. Opsi "data sudah dalam log" mencegah log ganda.
- C diestimasi sebagai rata-rata (konstanta sebagai regresor dengan galat ARMA), Maximum Likelihood, kovarians OPG, dan model tidak dipaksa stasioner/invertible (seperti EViews).
- Pembagian latih/uji kronologis. Model backtest hanya memakai data latih dan meramal dinamis sepanjang periode uji.
- Proyeksi ke depan memakai model yang diestimasi ulang dengan seluruh data sampai observasi terakhir.
- MAPE mengecualikan periode dengan aktual = 0, dan jumlahnya dilaporkan.
- Belum tersedia: pemilihan model otomatis (auto-ARIMA), karena tidak ada di kode acuan.
