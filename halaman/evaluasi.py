import streamlit as st

from halaman import komponen as kp
from modules.data_preprocessing import label_periode
from modules.evaluation import PENJELASAN

h = kp.perlu_hasil()
peng = h.pengaturan

st.title("5. Evaluasi Model")
if h.akurasi is None:
    st.info("Tidak ada periode uji, jadi akurasi ramalan tidak bisa diukur. Aktifkan "
            "'Sisihkan periode uji' di halaman Konfigurasi.")
    st.stop()

st.caption(f"Akurasi **di luar sampel**: model diestimasi dengan data {label_periode(peng.awal_latih)}–"
           f"{label_periode(peng.akhir_latih)}, lalu meramal {label_periode(peng.akhir_latih + 1)}–"
           f"{label_periode(peng.akhir_uji)} tanpa melihat nilai aktualnya.")

utama = h.akurasi.loc[peng.spek.label()]
m = st.columns(5)
m[0].metric("MAE", kp.fmt_angka(utama["MAE"]))
m[1].metric("RMSE", kp.fmt_angka(utama["RMSE"]))
m[2].metric("MAPE", f"{kp.fmt_angka(utama['MAPE (%)'])}%")
m[3].metric("sMAPE", f"{kp.fmt_angka(utama['sMAPE (%)'])}%")
m[4].metric("Theil U2", kp.fmt_angka(utama.get("Theil U2"), 3))

nol = int(utama.get("Aktual = 0 (dikecualikan dari MAPE)", 0))
if nol:
    st.warning(f"{nol} periode uji bernilai aktual 0 dan dikecualikan dari MAPE (pembagian dengan nol). "
               "MAE, RMSE, dan sMAPE tetap menghitungnya.")

st.subheader("Dibandingkan dengan ramalan sederhana")
st.caption("**Naif musiman**: nilai periode yang sama satu musim sebelumnya. **Rata-rata musiman**: rata-rata "
           "tiap posisi musim (mis. tiap bulan) di data latih. Model yang baik seharusnya lebih akurat dari keduanya.")
st.dataframe(h.akurasi.style.format(kp.fmt_angka, subset=[c for c in h.akurasi.columns
                                                           if c not in ("n", "Aktual = 0 (dikecualikan dari MAPE)")]))

st.subheader("Arti setiap ukuran")
for nama, isi in PENJELASAN.items():
    st.markdown(f"- **{nama}**: {isi}")
st.caption("Periode uji yang pendek membuat ukuran-ukuran ini sensitif terhadap satu-dua periode yang meleset jauh. "
           "Lihat juga kolom galat per periode di halaman Hasil.")
