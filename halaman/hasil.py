import streamlit as st

from halaman import komponen as kp
from modules.arima_model import tabel_koefisien
from modules.data_preprocessing import label_periode
from modules.pipeline import tabel_hasil
from modules.visualization import grafik_acf, grafik_hasil

h = kp.perlu_hasil()
peng = h.pengaturan
target = st.session_state.get(kp.TARGET, "Nilai")

st.title("4. Hasil Proyeksi")
for p in h.diagnostik_akhir["peringatan"]:
    st.warning(p)
for p in h.peringatan:
    st.info(p)

m = st.columns(4)
m[0].metric("Model", peng.spek.label())
m[1].metric("Data latih", f"{label_periode(peng.awal_latih)} – {label_periode(peng.akhir_latih)}")
m[2].metric("Periode uji", f"{label_periode(peng.akhir_latih + 1)} – {label_periode(peng.akhir_uji)}"
            if peng.akhir_uji else "Tidak ada")
m[3].metric("Horizon proyeksi", f"{peng.horizon} periode")

st.plotly_chart(grafik_hasil(h, target))

st.subheader("Tabel aktual dan ramalan")
st.caption("Baris 'Uji (backtest)': ramalan dari model yang hanya memakai data latih. "
           "Baris 'Proyeksi ke depan': aktual belum ada. Semua nilai dalam skala asli data.")
tabel = tabel_hasil(h)
st.dataframe(tabel.style.format({c: kp.fmt_angka for c in tabel.columns if c != "Jenis"}, na_rep="—"))

st.subheader("Estimasi model")
tab_akhir, tab_bt, tab_diag, tab_ringkas = st.tabs(
    ["Model untuk proyeksi", "Model backtest", "Diagnostik residual", "Ringkasan statsmodels"])

with tab_akhir:
    st.caption(f"Diestimasi dengan data {label_periode(peng.awal_latih)} s.d. {label_periode(h.akhir_data_proyeksi)}.")
    st.dataframe(h.koefisien_akhir.style.format("{:.6f}"))
    st.dataframe(h.statistik_akhir.to_frame("Nilai").style.format("{:.6f}"))
    st.caption("AIC/BIC statsmodels berupa nilai total; versi 'per obs.' sebanding dengan angka EViews.")

with tab_bt:
    if h.backtest is None:
        st.info("Tidak ada periode uji.")
    else:
        st.caption(f"Diestimasi hanya dengan data latih {label_periode(peng.awal_latih)} s.d. "
                   f"{label_periode(peng.akhir_latih)}; dipakai untuk ramalan periode uji.")
        st.dataframe(tabel_koefisien(h.backtest["hasil"]).style.format("{:.6f}"))

with tab_diag:
    diag = h.diagnostik_akhir
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Ljung-Box**: p > 0,05 berarti tidak ada autokorelasi residual yang tersisa.")
        st.dataframe(diag["ljung_box"].rename(columns={"lb_stat": "Statistik Q", "lb_pvalue": "p-value"}))
        jb = diag["jarque_bera"]
        st.markdown(f"**Jarque-Bera**: {jb['statistik']:.3f}, p = {jb['p_value']:.4f} "
                    f"→ residual {'normal' if jb['p_value'] > 0.05 else 'tidak normal'}.")
        ar, ma = diag["akar_ar_invers_maks"], diag["akar_ma_invers_maks"]
        st.markdown(f"**Akar invers maksimum** (harus < 1): AR = {ar:.4f}" if ar is not None else "AR: –")
        if ma is not None:
            st.markdown(f"MA = {ma:.4f}")
    with c2:
        st.plotly_chart(grafik_acf(diag))

with tab_ringkas:
    st.code(str(h.hasil_akhir.summary()), language="text")
