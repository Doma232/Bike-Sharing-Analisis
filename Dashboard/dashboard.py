# =========================
# IMPORT LIBRARY
# =========================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Dashboard Analisis Bike Sharing",
    page_icon="🚴",
    layout="wide"
)

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    df = pd.read_csv("Dataset/processed_hour.csv")
    df["dteday"] = pd.to_datetime(df["dteday"])
    df["year"] = df["dteday"].dt.year
    return df

df = load_data()

# =========================
# SIDEBAR
# =========================
st.sidebar.title("🚴 Analisis Bike Sharing")

tahun = st.sidebar.multiselect(
    "Pilih Tahun",
    sorted(df.year.unique()),
    default=sorted(df.year.unique())
)

musim = st.sidebar.multiselect(
    "Pilih Musim",
    df.season_label.unique(),
    default=df.season_label.unique()
)

cuaca = st.sidebar.multiselect(
    "Pilih Cuaca",
    df.weather_label.unique(),
    default=df.weather_label.unique()
)

hari = st.sidebar.radio(
    "Tipe Hari",
    ["Semua", "Hari Kerja", "Hari Libur"]
)

# =========================
# FILTER DATA
# =========================
filtered_df = df.copy()

filtered_df = filtered_df[filtered_df.year.isin(tahun)]
filtered_df = filtered_df[filtered_df.season_label.isin(musim)]
filtered_df = filtered_df[filtered_df.weather_label.isin(cuaca)]

if hari == "Hari Kerja":
    filtered_df = filtered_df[filtered_df.workingday == 1]
elif hari == "Hari Libur":
    filtered_df = filtered_df[filtered_df.workingday == 0]

# =========================
# HEADER
# =========================
st.title("🚴 Dashboard Analisis Bike Sharing")

# =========================
# METRICS
# =========================
col1, col2, col3, col4, col5 = st.columns(5)

total = filtered_df["cnt"].sum()
avg_harian = filtered_df.groupby("dteday")["cnt"].sum().mean()
casual_pct = filtered_df["casual"].sum() / total * 100 if total else 0
reg_pct = filtered_df["registered"].sum() / total * 100 if total else 0
jam_puncak = (
    filtered_df.groupby("hr")["cnt"].mean().idxmax()
    if not filtered_df.empty else 0
)

col1.metric("Total Peminjaman", f"{total:,}")
col2.metric("Rata-rata Harian", f"{avg_harian:.0f}")
col3.metric("Pengguna Kasual %", f"{casual_pct:.1f}%")
col4.metric("Pengguna Terdaftar %", f"{reg_pct:.1f}%")
col5.metric("Jam Puncak", f"{jam_puncak}:00")

# =========================
# TABS
# =========================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Ringkasan",
    "Analisis Waktu",
    "Dampak Cuaca",
    "Segmentasi Pengguna",
    "Klasterisasi"
])

# =========================
# RINGKASAN
# =========================
with tab1:

    st.subheader("Tren Peminjaman Harian")

    daily = filtered_df.groupby("dteday")["cnt"].sum().reset_index()
    fig = px.line(daily, x="dteday", y="cnt")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Insight Bisnis Utama")

    col1, col2, col3 = st.columns(3)

    best_season = filtered_df.groupby("season_label")["cnt"].mean().idxmax()
    best_weather = filtered_df.groupby("weather_label")["cnt"].mean().idxmax()
    top_hours = filtered_df.groupby("hr")["cnt"].mean().nlargest(3).index.tolist()

    col1.write(f"Musim Terbaik: {best_season}")
    col2.write(f"Jam Tertinggi: {top_hours}")
    col3.write(f"Cuaca Terbaik: {best_weather}")

# =========================
# ANALISIS WAKTU
# =========================
with tab2:

    hourly = filtered_df.groupby("hr")[["casual", "registered", "cnt"]].mean().reset_index()
    fig = px.line(hourly, x="hr", y=["casual", "registered", "cnt"])
    st.plotly_chart(fig, use_container_width=True)

# =========================
# DAMPAK CUACA
# =========================
with tab3:

    cuaca_avg = filtered_df.groupby("weather_label")[["casual", "registered"]].mean().reset_index()
    fig = px.bar(cuaca_avg, x='weather_label', y=['casual', 'registered'], barmode='group')
    st.plotly_chart(fig, use_container_width=True)

# =========================
# SEGMENTASI PENGGUNA
# =========================
with tab4:

    user_hour = filtered_df.groupby("hr")[["casual", "registered"]].mean().reset_index()
    fig = px.area(user_hour, x="hr", y=["casual", "registered"])
    st.plotly_chart(fig, use_container_width=True)

# =========================
# KLASTERISASI
# =========================
with tab5:

    st.subheader("Analisis Klaster Lanjutan")

    if filtered_df.shape[0] < 10:
        st.warning("Data tidak cukup untuk clustering berdasarkan filter yang dipilih.")
    else:

        cluster_data = filtered_df.groupby("hr").mean(numeric_only=True).reset_index()

        scaler = StandardScaler()
        X = scaler.fit_transform(cluster_data.drop("hr", axis=1))

        k = st.slider("Jumlah Klaster", 2, 8, 4)

        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_data["cluster"] = kmeans.fit_predict(X)

        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(X)

        cluster_data["pca1"] = pca_result[:, 0]
        cluster_data["pca2"] = pca_result[:, 1]

        fig = px.scatter(
            cluster_data,
            x="pca1",
            y="pca2",
            color="cluster",
            size="cnt",
            text="hr"
        )

        st.plotly_chart(fig, use_container_width=True)

# =========================
# FOOTER
# =========================
st.markdown("---")
st.markdown("🚴 Dashboard Analisis Bike Sharing | Streamlit & Plotly")
