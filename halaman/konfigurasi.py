import traceback

import streamlit as st

from halaman import komponen as kp
from modules.arima_model import SpesifikasiModel
from modules.data_preprocessing import FREKUENSI, TRANSFORMASI, cek_transformasi, label_periode, petunjuk_log_ganda
from modules.pipeline import GalatPengaturan, Pengaturan, jalankan

seri = kp.perlu_seri()
s_bawaan = FREKUENSI[st.session_state[kp.FREKUENSI]][1]
periode = list(seri.dropna().index)
n = len(periode)

st.title("3. Konfigurasi & Jalankan")

# ---------------------------------------------------------------- transformasi
st.subheader("Transformasi data")
transformasi = st.radio(
    "Bagaimana data diolah sebelum dimodelkan?", list(TRANSFORMASI), format_func=TRANSFORMASI.get,
    help="Log membuat pola musiman yang membesar-mengecil menjadi lebih stabil. Hasil ramalan selalu "
         "dikembalikan ke skala asli (dengan exp) secara otomatis.")
if (petunjuk := petunjuk_log_ganda(seri, transformasi)):
    st.warning(petunjuk)
for g in cek_transformasi(seri, transformasi):
    st.error(g)
if transformasi == "sudah_log":
    st.info("Data dianggap sudah berupa log. Model memakai data apa adanya; hasil ditampilkan dalam skala asli (exp).")

# ---------------------------------------------------------------- periode
st.subheader("Periode latih dan uji")
st.caption("Data dibagi menurut urutan waktu (tidak diacak). Model hanya belajar dari data latih; "
           "data uji dipakai untuk mengukur akurasi ramalan.")
pakai_uji = st.checkbox("Sisihkan periode uji (disarankan)", value=n > 2 * s_bawaan + 10)
panjang_uji = s_bawaan if n > 3 * s_bawaan else max(1, n // 5)
c1, c2, c3 = st.columns(3)
awal_latih = c1.selectbox("Awal data latih", periode, index=0, format_func=label_periode)
opsi_akhir = [p for p in periode if p > awal_latih]
idx_akhir = max(0, len(opsi_akhir) - 1 - (panjang_uji if pakai_uji else 0))
akhir_latih = c2.selectbox("Akhir data latih", opsi_akhir, index=idx_akhir, format_func=label_periode)
akhir_uji = None
if pakai_uji:
    opsi_uji = [p for p in periode if p > akhir_latih]
    if opsi_uji:
        akhir_uji = c3.selectbox("Akhir data uji", opsi_uji, index=len(opsi_uji) - 1, format_func=label_periode)
    else:
        c3.warning("Tidak ada data setelah akhir data latih.")

# ---------------------------------------------------------------- model
st.subheader("Spesifikasi model")
st.info("**Spesifikasi manual.** Pemilihan model otomatis (auto-ARIMA) belum tersedia karena tidak ada di kode "
        "acuan; orde diisi dari hasil identifikasi Anda (mis. EViews). Nilai awal: (0,0,0)(1,0,0) — model SAR musiman.")
with st.expander("Penjelasan parameter"):
    st.markdown(
        """
| Parameter | Arti |
|---|---|
| **p** | Jumlah lag AR: pengaruh nilai periode-periode sebelumnya |
| **d** | Berapa kali data dibedakan (differencing). 0 bila data sudah stasioner |
| **q** | Jumlah lag MA: pengaruh kesalahan ramalan sebelumnya |
| **P, D, Q** | Sama seperti p, d, q tetapi pada lag musiman (mis. 12 bulan lalu) |
| **s** | Panjang satu musim: 12 bulanan, 4 triwulanan, 52 mingguan, 7 harian |
| **Konstanta C** | Rata-rata deret (seperti C di EViews) |
"""
    )
c = st.columns(7)
p = c[0].number_input("p", 0, 5, 0)
d = c[1].number_input("d", 0, 2, 0)
q = c[2].number_input("q", 0, 5, 0)
P = c[3].number_input("P", 0, 3, 1)
D = c[4].number_input("D", 0, 1, 0)
Q = c[5].number_input("Q", 0, 3, 0)
s = c[6].number_input("s", 1, 366, s_bawaan)
konstanta = st.checkbox("Sertakan konstanta C (rata-rata)", value=True)
if konstanta and (d or D):
    st.warning("Dengan differencing (d atau D > 0), konstanta berarti tren/drift. Pastikan ini memang dimaksud.")

with st.expander("Opsi lanjutan"):
    paksa_stasioner = st.checkbox("Paksa model stasioner", value=False,
                                  help="EViews tidak memaksa. Aktifkan bila estimasi lari ke koefisien AR = 1.")
    paksa_invertible = st.checkbox("Paksa model invertible", value=False)
    koreksi_bias = st.checkbox("Koreksi bias saat mengembalikan dari log", value=False,
                               help="exp(ramalan log) adalah median, bukan rata-rata. Koreksi mengalikan "
                                    "dengan exp(σ²/2). Kode acuan/EViews tidak memakainya.")

# ---------------------------------------------------------------- proyeksi
st.subheader("Proyeksi")
c1, c2 = st.columns(2)
horizon = c1.number_input("Horizon (jumlah periode ke depan)", 1, 5 * s_bawaan if s_bawaan > 1 else 60, s_bawaan)
tingkat = c2.select_slider("Tingkat kepercayaan selang ramalan", [80, 90, 95, 99], value=95)

spek = SpesifikasiModel(order=(int(p), int(d), int(q)), seasonal_order=(int(P), int(D), int(Q), int(s)),
                        konstanta=konstanta, paksa_stasioner=paksa_stasioner, paksa_invertible=paksa_invertible)
peng = Pengaturan(spek=spek, transformasi=transformasi, awal_latih=awal_latih, akhir_latih=akhir_latih,
                  akhir_uji=akhir_uji, horizon=int(horizon), tingkat_kepercayaan=int(tingkat),
                  koreksi_bias=koreksi_bias)

st.divider()
st.markdown(f"**Model:** {spek.label()} · **Latih:** {label_periode(awal_latih)}–{label_periode(akhir_latih)}"
            + (f" · **Uji:** {label_periode(akhir_latih + 1)}–{label_periode(akhir_uji)}" if akhir_uji else " · tanpa uji")
            + f" · **Horizon:** {horizon}")

if st.button("Jalankan Proyeksi", type="primary", icon=":material/play_arrow:"):
    with st.status("Menjalankan model...", expanded=True) as status:
        try:
            st.write("Memeriksa data dan pengaturan, mengestimasi model, membuat ramalan...")
            hasil = jalankan(seri, peng)
            st.session_state[kp.HASIL] = hasil
            status.update(label="Selesai", state="complete")
        except GalatPengaturan as e:
            status.update(label="Pengaturan perlu diperbaiki", state="error")
            for baris in str(e).splitlines():
                st.error(baris)
        except Exception as e:
            status.update(label="Estimasi gagal", state="error")
            st.error("Model tidak berhasil diestimasi dengan pengaturan ini. Coba model yang lebih sederhana, "
                     "periksa nilai kosong, atau aktifkan opsi 'Paksa model stasioner'.")
            with st.expander("Detail teknis (untuk pengembang)"):
                st.code(f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}")

if st.session_state.get(kp.HASIL) is not None:
    st.success(f"Hasil tersedia untuk {st.session_state[kp.HASIL].pengaturan.spek.label()}.")
    if st.button("Lihat hasil", icon=":material/show_chart:"):
        st.switch_page("halaman/hasil.py")
