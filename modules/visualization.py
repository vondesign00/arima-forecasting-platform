"""Grafik: Plotly untuk layar (interaktif), Matplotlib untuk file PNG unduhan.

Plotly diimpor di dalam fungsi agar fungsi PNG bisa diuji tanpa Plotly."""
from __future__ import annotations

import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

WARNA = {
    "latih": "#1f2937",
    "uji": "#0f766e",
    "backtest": "#d97706",
    "proyeksi": "#2563eb",
    "pita_backtest": "rgba(217,119,6,0.15)",
    "pita_proyeksi": "rgba(37,99,235,0.15)",
}
TATA_LETAK = dict(template="plotly_white", height=440, margin=dict(t=40, b=10, l=10, r=10),
                  legend=dict(orientation="h", y=-0.15), hovermode="x unified")


def _x(index: pd.PeriodIndex):
    return index.to_timestamp()


def grafik_seri(seri: pd.Series, judul: str = "", label_y: str = "Nilai"):
    import plotly.graph_objects as go
    fig = go.Figure(go.Scatter(x=_x(seri.index), y=seri, mode="lines+markers", name=seri.name or "Data",
                               line=dict(color=WARNA["latih"]), marker=dict(size=4)))
    fig.update_layout(title=judul, yaxis_title=label_y, **TATA_LETAK)
    return fig


def _pita(fig, index, bawah, atas, warna, nama):
    fig.add_scatter(x=_x(index), y=atas, line=dict(width=0), showlegend=False, hoverinfo="skip")
    fig.add_scatter(x=_x(index), y=bawah, fill="tonexty", fillcolor=warna, line=dict(width=0), name=nama,
                    hoverinfo="skip")


def grafik_hasil(h, label_y: str = "Nilai"):
    import plotly.graph_objects as go
    """Data latih, data uji, ramalan backtest, proyeksi ke depan, dan selang kepercayaan."""
    peng = h.pengaturan
    tingkat = peng.tingkat_kepercayaan
    fig = go.Figure()
    latih = h.y_asli.loc[peng.awal_latih:peng.akhir_latih]
    fig.add_scatter(x=_x(latih.index), y=latih, name="Aktual — data latih", mode="lines",
                    line=dict(color=WARNA["latih"]))
    if h.backtest is not None:
        t = h.backtest["tabel"]
        _pita(fig, t.index, t["batas_bawah"], t["batas_atas"], WARNA["pita_backtest"], f"Selang {tingkat}% (uji)")
        fig.add_scatter(x=_x(t.index), y=t["aktual"], name="Aktual — data uji", mode="lines+markers",
                        line=dict(color=WARNA["uji"]))
        fig.add_scatter(x=_x(t.index), y=t["ramalan"], name="Ramalan pada periode uji", mode="lines+markers",
                        line=dict(color=WARNA["backtest"], dash="dash"))
    sesudah = h.y_asli.loc[(peng.akhir_uji or peng.akhir_latih) + 1:h.akhir_data_proyeksi]
    if len(sesudah.dropna()):
        fig.add_scatter(x=_x(sesudah.index), y=sesudah, name="Aktual — setelah periode uji", mode="lines",
                        line=dict(color=WARNA["latih"], dash="dot"))
    r = h.ramalan
    _pita(fig, r.index, r["batas_bawah"], r["batas_atas"], WARNA["pita_proyeksi"], f"Selang {tingkat}% (proyeksi)")
    fig.add_scatter(x=_x(r.index), y=r["ramalan"], name="Proyeksi ke depan", mode="lines+markers",
                    line=dict(color=WARNA["proyeksi"]))
    fig.update_layout(title=f"{peng.spek.label()} — aktual dan ramalan", yaxis_title=label_y, **TATA_LETAK)
    return fig


def grafik_acf(diag: dict):
    import plotly.graph_objects as go
    a = diag["acf"]
    fig = go.Figure(go.Bar(x=a["lag"], y=a["acf"], marker_color=WARNA["proyeksi"], name="ACF"))
    for tanda in (1, -1):
        fig.add_hline(y=tanda * diag["batas_acf"], line_dash="dash", line_color="gray")
    fig.update_layout(title="ACF residual (garis putus = batas signifikansi 95%)", xaxis_title="Lag",
                      **{**TATA_LETAK, "height": 320, "hovermode": "closest"})
    return fig


def png_hasil(h, label_y: str = "Nilai") -> bytes:
    """Versi Matplotlib dari grafik_hasil, untuk diunduh sebagai PNG."""
    peng = h.pengaturan
    fig, ax = plt.subplots(figsize=(11, 5), dpi=150)
    latih = h.y_asli.loc[peng.awal_latih:peng.akhir_latih]
    ax.plot(_x(latih.index), latih, color=WARNA["latih"], label="Aktual — data latih")
    if h.backtest is not None:
        t = h.backtest["tabel"]
        ax.fill_between(_x(t.index), t["batas_bawah"], t["batas_atas"], color=WARNA["backtest"], alpha=0.15)
        ax.plot(_x(t.index), t["aktual"], color=WARNA["uji"], marker="o", ms=3, label="Aktual — data uji")
        ax.plot(_x(t.index), t["ramalan"], color=WARNA["backtest"], ls="--", marker="o", ms=3,
                label="Ramalan pada periode uji")
    sesudah = h.y_asli.loc[(peng.akhir_uji or peng.akhir_latih) + 1:h.akhir_data_proyeksi]
    if len(sesudah.dropna()):
        ax.plot(_x(sesudah.index), sesudah, color=WARNA["latih"], ls=":", label="Aktual — setelah periode uji")
    r = h.ramalan
    ax.fill_between(_x(r.index), r["batas_bawah"], r["batas_atas"], color=WARNA["proyeksi"], alpha=0.15,
                    label=f"Selang {peng.tingkat_kepercayaan}%")
    ax.plot(_x(r.index), r["ramalan"], color=WARNA["proyeksi"], marker="o", ms=3, label="Proyeksi ke depan")
    ax.set_title(f"{peng.spek.label()} — aktual dan ramalan")
    ax.set_ylabel(label_y)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)
    ax.ticklabel_format(axis="y", style="plain")
    fig.autofmt_xdate()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()
