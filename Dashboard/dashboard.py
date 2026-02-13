import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

st.set_page_config(
    page_title="Dashboard Analisis Bike Sharing",
    page_icon="🚴",
    layout="wide"
)

# ================= LOAD DATA =================
@st.cache_data
def load_data():
    df = pd.read_csv("Dataset/processed_hour.csv")

    df["dteday"] = pd.to_datetime(df["dteday"])
    df["year"] = df["dteday"].dt.year

    season_labels = {
        1: "Semi",
        2: "Panas",
        3: "Gugur",
        4: "Dingin"
    }

    weather_labels = {
        1: "Cerah",
        2: "Berkabut",
        3: "Hujan Ringan/Salju",
        4: "Hujan Lebat"
    }

    weekday_labels = {
        0: "Senin", 1: "Selasa", 2: "Rabu", 3: "Kamis",
        4: "Jumat", 5: "Sabtu", 6: "Minggu"
    }

    df["season_label"] = df["season"].map(season_labels)
    df["weather_label"] = df["weathersit"].map(weather_labels)
    df["weekday_label"] = df["weekday"].map(weekday_labels)

    return df

df = load_data()

# ================= SIDEBAR =================
with st.sidebar:
    st.title("🚴 Dashboard Bike Sharing")

    tahun = st.multiselect("Pilih Tahun", sorted(df["year"].unique()), default=sorted(df["year"].unique()))
    musim = st.multiselect("Pilih Musim", df["season_label"].unique(), default=df["season_label"].unique())
    cuaca = st.multiselect("Pilih Cuaca", df["weather_label"].unique(), default=df["weather_label"].unique())
    hari = st.radio("Tipe Hari", ["Semua", "Hari Kerja", "Hari Libur"])

# ================= FILTER =================
filtered_df = df.copy()

filtered_df = filtered_df[filtered_df["year"].isin(tahun)]
filtered_df = filtered_df[filtered_df["season_label"].isin(musim)]
filtered_df = filtered_df[filtered_df["weather_label"].isin(cuaca)]

if hari == "Hari Kerja":
    filtered_df = filtered_df[filtered_df["workingday"] == 1]
elif hari == "Hari Libur":
    filtered_df = filtered_df[filtered_df["workingday"] == 0]

# ================= METRIC =================
st.title("📊 Dashboard Analisis Bike Sharing")

col1, col2, col3, col4, col5 = st.columns(5)

total = filtered_df["cnt"].sum()
rata_harian = filtered_df.groupby("dteday")["cnt"].sum().mean()
kasual = filtered_df["casual"].sum()
terdaftar = filtered_df["registered"].sum()
jam_puncak = filtered_df.groupby("hr")["cnt"].mean().idxmax()

col1.metric("Total Peminjaman", f"{total:,}")
col2.metric("Rata-rata Harian", f"{rata_harian:,.0f}")
col3.metric("Pengguna Kasual %", f"{kasual/total*100:.1f}%")
col4.metric("Pengguna Terdaftar %", f"{terdaftar/total*100:.1f}%")
col5.metric("Jam Puncak", f"{jam_puncak}:00")

# ================= TAB =================
tab1, tab2, tab3 = st.tabs([
    "📊 Ringkasan",
    "🌤️ Dampak Cuaca",
    "👥 Segmentasi Pengguna"
])

# ================= RINGKASAN =================
with tab1:
    st.subheader("Tren Peminjaman Harian")

    daily = filtered_df.groupby("dteday")["cnt"].sum().reset_index()

    fig = px.line(daily, x="dteday", y="cnt", labels={
        "dteday": "Tanggal",
        "cnt": "Total Peminjaman"
    })
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Insight Utama")

    musim_terbaik = filtered_df.groupby("season_label")["cnt"].mean().idxmax()
    cuaca_terbaik = filtered_df.groupby("weather_label")["cnt"].mean().idxmax()

    st.write(f"✅ Musim terbaik: **{musim_terbaik}**")
    st.write(f"🌤️ Cuaca terbaik: **{cuaca_terbaik}**")

# ================= CUACA =================
with tab2:
    st.subheader("Rata-rata Peminjaman Berdasarkan Cuaca")

    cuaca_avg = filtered_df.groupby("weather_label")[["casual","registered"]].mean().reset_index()

    fig = go.Figure()
    fig.add_bar(x=cuaca_avg["weather_label"], y=cuaca_avg["casual"], name="Kasual")
    fig.add_bar(x=cuaca_avg["weather_label"], y=cuaca_avg["registered"], name="Terdaftar")

    fig.update_layout(
        xaxis_title="Kondisi Cuaca",
        yaxis_title="Rata-rata Peminjaman"
    )

    st.plotly_chart(fig, use_container_width=True)

# ================= USER =================
with tab3:
    st.subheader("Distribusi Pengguna")

    fig = go.Figure(data=[go.Pie(
        labels=["Kasual", "Terdaftar"],
        values=[kasual, terdaftar],
        hole=.4
    )])

    st.plotly_chart(fig, use_container_width=True)
