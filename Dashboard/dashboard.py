# =========================================================
# IMPORT LIBRARY
# =========================================================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from datetime import datetime

# =========================================================
# KONFIGURASI HALAMAN
# =========================================================
st.set_page_config(
    page_title="Dashboard Analisis Bike Sharing",
    page_icon="🚴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# LOAD DATA
# =========================================================
@st.cache_data
def load_data():
    df = pd.read_csv("Dataset/processed_hour.csv")

    # Konversi tanggal
    df["dteday"] = pd.to_datetime(df["dteday"])
    df["datetime"] = df["dteday"] + pd.to_timedelta(df["hr"], unit="h")

    # Feature tambahan
    df["year"] = df["dteday"].dt.year
    df["month"] = df["dteday"].dt.month

    # Label kategori
    season_labels = {1: "Spring", 2: "Summer", 3: "Fall", 4: "Winter"}
    weather_labels = {1: "Clear", 2: "Mist", 3: "Light Snow/Rain", 4: "Heavy Rain/Snow"}
    weekday_labels = {0: "Senin", 1: "Selasa", 2: "Rabu", 3: "Kamis",
                      4: "Jumat", 5: "Sabtu", 6: "Minggu"}

    df["season_label"] = df["season"].map(season_labels)
    df["weather_label"] = df["weathersit"].map(weather_labels)
    df["weekday_label"] = df["weekday"].map(weekday_labels)

    # Rush hour
    df["is_rush_hour"] = df["hr"].apply(lambda x: 1 if x in [7, 8, 17, 18] else 0)

    # Suhu ke Celsius
    df["temp_celsius"] = df["temp"] * 41 - 8

    return df


df = load_data()

# =========================================================
# SIDEBAR FILTER
# =========================================================
with st.sidebar:
    st.title("🚴 Dashboard Bike Sharing")

    years = sorted(df["year"].unique())
    selected_year = st.multiselect("Pilih Tahun", years, default=years)

    seasons = df["season_label"].unique()
    selected_season = st.multiselect("Pilih Musim", seasons, default=seasons)

    weather = df["weather_label"].unique()
    selected_weather = st.multiselect("Pilih Kondisi Cuaca", weather, default=weather)

    working_day_option = st.radio("Tipe Hari", ["Semua", "Hari Kerja", "Hari Libur"])

# =========================================================
# PROSES FILTER DATA
# =========================================================
filtered_df = df.copy()

if selected_year:
    filtered_df = filtered_df[filtered_df["year"].isin(selected_year)]

if selected_season:
    filtered_df = filtered_df[filtered_df["season_label"].isin(selected_season)]

if selected_weather:
    filtered_df = filtered_df[filtered_df["weather_label"].isin(selected_weather)]

if working_day_option == "Hari Kerja":
    filtered_df = filtered_df[filtered_df["workingday"] == 1]
elif working_day_option == "Hari Libur":
    filtered_df = filtered_df[filtered_df["workingday"] == 0]

# =========================================================
# HEADER DASHBOARD
# =========================================================
st.title("📊 Dashboard Analisis Bike Sharing")

# =========================================================
# METRIC UTAMA
# =========================================================
col1, col2, col3, col4, col5 = st.columns(5)

total_rentals = filtered_df["cnt"].sum()
avg_daily = filtered_df.groupby("dteday")["cnt"].sum().mean()
casual_pct = filtered_df["casual"].sum() / total_rentals * 100
registered_pct = filtered_df["registered"].sum() / total_rentals * 100
peak_hour = filtered_df.groupby("hr")["cnt"].mean().idxmax()

col1.metric("Total Peminjaman", f"{total_rentals:,}")
col2.metric("Rata-rata Harian", f"{avg_daily:,.0f}")
col3.metric("Pengguna Kasual %", f"{casual_pct:.1f}%")
col4.metric("Pengguna Terdaftar %", f"{registered_pct:.1f}%")
col5.metric("Jam Puncak", f"{peak_hour}:00")

# =========================================================
# TAB NAVIGASI
# =========================================================
tab1, tab2, tab3 = st.tabs([
    "📊 Ringkasan",
    "🌤️ Dampak Cuaca",
    "👥 Segmentasi Pengguna"
])

# =========================================================
# TAB 1 — RINGKASAN
# =========================================================
with tab1:
    st.subheader("Tren Peminjaman Harian")

    daily = filtered_df.groupby("dteday")["cnt"].sum().reset_index()

    fig = px.line(daily, x="dteday", y="cnt",
                  labels={"dteday": "Tanggal", "cnt": "Total Peminjaman"})
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Insight Utama")

    best_season = filtered_df.groupby("season_label")["cnt"].mean().idxmax()
    best_weather = filtered_df.groupby("weather_label")["cnt"].mean().idxmax()

    st.write(f"Musim terbaik: **{best_season}**")
    st.write(f"Cuaca terbaik: **{best_weather}**")

# =========================================================
# TAB 2 — DAMPAK CUACA
# =========================================================
with tab2:
    st.subheader("Rata-rata Peminjaman Berdasarkan Cuaca")

    weather_avg = filtered_df.groupby("weather_label")[["casual", "registered"]].mean().reset_index()

    fig = go.Figure()
    fig.add_bar(x=weather_avg["weather_label"], y=weather_avg["casual"], name="Kasual")
    fig.add_bar(x=weather_avg["weather_label"], y=weather_avg["registered"], name="Terdaftar")

    fig.update_layout(
        xaxis_title="Kondisi Cuaca",
        yaxis_title="Rata-rata Peminjaman"
    )

    st.plotly_chart(fig, use_container_width=True)

# =========================================================
# TAB 3 — SEGMENTASI PENGGUNA
# =========================================================
with tab3:
    st.subheader("Distribusi Pengguna")

    fig = go.Figure(data=[go.Pie(
        labels=["Kasual", "Terdaftar"],
        values=[filtered_df["casual"].sum(), filtered_df["registered"].sum()],
        hole=.4
    )])

    st.plotly_chart(fig, use_container_width=True)

# =========================================================
# FOOTER
# =========================================================
st.markdown("---")
st.markdown("🚴 Dashboard Analisis Bike Sharing — Streamlit")
