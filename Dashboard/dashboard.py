# ===============================
# IMPORT LIBRARY
# ===============================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(layout="wide")

# ===============================
# LOAD DATA
# ===============================
@st.cache_data
def load_data():
    df = pd.read_csv("Dataset/processed_hour.csv")
    df["dteday"] = pd.to_datetime(df["dteday"])
    return df

df = load_data()

# ===============================
# SIDEBAR
# ===============================
st.sidebar.title("🚴 Analisis Bike Sharing")

st.sidebar.header("Filter")

# Tahun
tahun = st.sidebar.multiselect(
    "Pilih Tahun",
    options=df["yr"].unique(),
    default=df["yr"].unique()
)

# Musim
musim = st.sidebar.multiselect(
    "Pilih Musim",
    options=df["season_label"].unique(),
    default=df["season_label"].unique()
)

# Cuaca
cuaca = st.sidebar.multiselect(
    "Pilih Cuaca",
    options=df["weather_label"].unique(),
    default=df["weather_label"].unique()
)

# Tipe hari
tipe_hari = st.sidebar.radio(
    "Tipe Hari",
    ["Semua", "Hari Kerja", "Hari Libur"]
)

# ===============================
# FILTER DATA
# ===============================
filtered_df = df.copy()

filtered_df = filtered_df[filtered_df["yr"].isin(tahun)]
filtered_df = filtered_df[filtered_df["season_label"].isin(musim)]
filtered_df = filtered_df[filtered_df["weather_label"].isin(cuaca)]

if tipe_hari == "Hari Kerja":
    filtered_df = filtered_df[filtered_df["workingday"] == 1]
elif tipe_hari == "Hari Libur":
    filtered_df = filtered_df[filtered_df["workingday"] == 0]

# ===============================
# METRIC UTAMA
# ===============================
st.title("🚴 Dashboard Analisis Bike Sharing")

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Total Peminjaman", f"{filtered_df['cnt'].sum():,}")
col2.metric("Rata-rata Harian", f"{filtered_df.groupby('dteday')['cnt'].sum().mean():.0f}")
col3.metric("Pengguna Kasual %", f"{filtered_df['casual'].sum()/filtered_df['cnt'].sum()*100:.1f}%")
col4.metric("Pengguna Terdaftar %", f"{filtered_df['registered'].sum()/filtered_df['cnt'].sum()*100:.1f}%")
col5.metric("Jam Puncak", f"{filtered_df.groupby('hr')['cnt'].mean().idxmax():02d}:00")

# ===============================
# TABS
# ===============================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Ringkasan",
    "⏰ Analisis Waktu",
    "🌤️ Dampak Cuaca",
    "👥 Segmentasi Pengguna",
    "🎯 Klasterisasi"
])

# =========================================================
# 📊 RINGKASAN
# =========================================================
with tab1:

    st.subheader("Tren Peminjaman Harian")

    daily = filtered_df.groupby("dteday")["cnt"].sum().reset_index()
    fig = px.line(daily, x="dteday", y="cnt")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Distribusi Peminjaman per Tipe Pengguna")

    fig = px.box(filtered_df, y=["cnt", "casual", "registered"])
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Insight Bisnis Utama")

    col1, col2, col3 = st.columns(3)

    col1.markdown(f"""
    **Performa Terbaik**
    - Musim Terbaik: **{filtered_df.groupby('season_label')['cnt'].mean().idxmax()}**
    - Rata-rata Peminjaman: **{filtered_df.groupby('season_label')['cnt'].mean().max():.0f}**
    """)

    col2.markdown(f"""
    **Pola Penggunaan**
    - Jam Tertinggi: **{filtered_df.groupby('hr')['cnt'].mean().nlargest(3).index.tolist()}**
    """)

    col3.markdown(f"""
    **Dampak Cuaca**
    - Cuaca Terbaik: **{filtered_df.groupby('weather_label')['cnt'].mean().idxmax()}**
    """)

# =========================================================
# ⏰ ANALISIS WAKTU
# =========================================================
with tab2:

    st.subheader("Rata-rata Peminjaman per Jam")

    hourly = filtered_df.groupby("hr")[["casual","registered","cnt"]].mean().reset_index()
    fig = px.line(hourly, x="hr", y=["casual","registered","cnt"])
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Rata-rata Peminjaman per Hari")

    by_day = filtered_df.groupby("weekday")[["casual","registered"]].mean().reset_index()
    fig = px.bar(by_day, x="weekday", y=["casual","registered"], barmode="group")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Heatmap Pola Jam")

    heat = filtered_df.pivot_table(values="cnt", index="hr", columns="weekday", aggfunc="mean")
    fig = px.imshow(heat)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Tren Bulanan")

    monthly = filtered_df.groupby(filtered_df["dteday"].dt.to_period("M"))["cnt"].sum().reset_index()
    monthly["dteday"] = monthly["dteday"].astype(str)

    fig = px.line(monthly, x="dteday", y="cnt")
    st.plotly_chart(fig, use_container_width=True)

# =========================================================
# 🌤️ DAMPAK CUACA
# =========================================================
with tab3:

    st.subheader("Rata-rata Peminjaman Berdasarkan Cuaca")

    weather_avg = filtered_df.groupby("weather_label")[["casual","registered"]].mean().reset_index()
    fig = px.bar(weather_avg, x="weather_label", y=["casual","registered"], barmode="group")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Distribusi Musim")
    season_dist = filtered_df["season_label"].value_counts()
    fig = px.pie(values=season_dist.values, names=season_dist.index, hole=0.4)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Dampak Suhu & Kelembapan")

    fig = px.scatter(filtered_df, x="temp", y="cnt", color="season_label")
    st.plotly_chart(fig, use_container_width=True)

    fig = px.scatter(filtered_df, x="hum", y="cnt", color="weather_label")
    st.plotly_chart(fig, use_container_width=True)

# =========================================================
# 👥 SEGMENTASI PENGGUNA
# =========================================================
with tab4:

    st.subheader("Distribusi Pengguna per Jam")

    user_hour = filtered_df.groupby("hr")[["casual","registered"]].mean().reset_index()
    fig = px.area(user_hour, x="hr", y=["casual","registered"])
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Distribusi Total Pengguna")
    user_total = filtered_df[["casual","registered"]].sum()

    fig = px.pie(values=user_total.values, names=user_total.index, hole=0.4)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Perbandingan Hari Kerja vs Libur")

    working = filtered_df.groupby(["workingday","hr"])["cnt"].mean().reset_index()
    fig = px.bar(working, x="hr", y="cnt", color="workingday", barmode="group")
    st.plotly_chart(fig, use_container_width=True)

# =========================================================
# 🎯 KLASTERISASI
# =========================================================
with tab5:

    st.subheader("Analisis Klaster Lanjutan")

    k = st.slider("Jumlah Klaster", 2, 6, 4)

    cluster_df = pd.read_csv("Dataset/cluster_analysis.csv")

    fig = px.scatter(cluster_df, x="pca1", y="pca2", color="cluster")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Perbandingan Klaster")
    st.dataframe(cluster_df.groupby("cluster").mean())

    st.subheader("Rekomendasi Strategis")
    st.markdown("""
    1. Tambahkan sepeda saat jam klaster permintaan tinggi  
    2. Gunakan dynamic pricing saat peak time  
    3. Jadwalkan maintenance saat demand rendah  
    4. Targetkan pengguna kasual di jam santai  
    5. Optimalkan redistribusi armada  
    """)

# ===============================
# FOOTER
# ===============================
st.markdown("🚴 Dashboard Analisis Bike Sharing | Dibuat dengan Streamlit & Plotly")
