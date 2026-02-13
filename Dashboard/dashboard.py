import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Dashboard Analisis Bike Sharing",
    page_icon="🚴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================= CSS =================
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    font-weight: bold;
    text-align: center;
    padding: 1rem 0;
}
.insight-box {
    background-color: #e8f4f8;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #2ecc71;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

# ================= LOAD DATA =================
@st.cache_data
def load_data():
    df = pd.read_csv('Dataset/processed_hour.csv')

    df['dteday'] = pd.to_datetime(df['dteday'])
    df['datetime'] = df['dteday'] + pd.to_timedelta(df['hr'], unit='h')

    df['year'] = df['dteday'].dt.year
    df['month'] = df['dteday'].dt.month

    season_labels = {1: 'Spring', 2: 'Summer', 3: 'Fall', 4: 'Winter'}
    weather_labels = {1: 'Clear', 2: 'Mist', 3: 'Light Snow/Rain', 4: 'Heavy Rain/Snow'}
    weekday_labels = {0: 'Senin', 1: 'Selasa', 2: 'Rabu', 3: 'Kamis',
                      4: 'Jumat', 5: 'Sabtu', 6: 'Minggu'}

    df['season_label'] = df['season'].map(season_labels)
    df['weather_label'] = df['weathersit'].map(weather_labels)
    df['weekday_label'] = df['weekday'].map(weekday_labels)

    df['is_rush_hour'] = df['hr'].apply(lambda x: 1 if x in [7, 8, 17, 18] else 0)

    df['temp_celsius'] = df['temp'] * 41 - 8
    df['atemp_celsius'] = df['atemp'] * 50 - 16

    return df

df = load_data()

# ================= SIDEBAR =================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/bicycle.png", width=80)
    st.title("🚴 Analisis Bike Sharing")

    st.header("Filter")

    years = sorted(df['year'].unique())
    selected_year = st.multiselect("Pilih Tahun", years, default=years)

    seasons = df['season_label'].unique()
    selected_season = st.multiselect("Pilih Musim", seasons, default=seasons)

    weather_conditions = df['weather_label'].unique()
    selected_weather = st.multiselect("Pilih Cuaca", weather_conditions, default=weather_conditions)

    working_day_option = st.radio("Tipe Hari", ["Semua", "Hari Kerja", "Hari Libur"])

# ================= FILTER DATA =================
filtered_df = df.copy()

if selected_year:
    filtered_df = filtered_df[filtered_df['year'].isin(selected_year)]

if selected_season:
    filtered_df = filtered_df[filtered_df['season_label'].isin(selected_season)]

if selected_weather:
    filtered_df = filtered_df[filtered_df['weather_label'].isin(selected_weather)]

if working_day_option == "Hari Kerja":
    filtered_df = filtered_df[filtered_df['workingday'] == 1]
elif working_day_option == "Hari Libur":
    filtered_df = filtered_df[filtered_df['workingday'] == 0]

# ================= HEADER =================
st.markdown('<h1 class="main-header">🚴 Dashboard Analisis Bike Sharing</h1>', unsafe_allow_html=True)
st.markdown("---")

# ================= METRIC =================
col1, col2, col3, col4, col5 = st.columns(5)

total_rentals = filtered_df['cnt'].sum()
avg_daily = filtered_df.groupby('dteday')['cnt'].sum().mean()
casual_pct = (filtered_df['casual'].sum() / total_rentals * 100)
registered_pct = (filtered_df['registered'].sum() / total_rentals * 100)
peak_hour = filtered_df.groupby('hr')['cnt'].mean().idxmax()

col1.metric("Total Peminjaman", f"{total_rentals:,}")
col2.metric("Rata-rata Harian", f"{avg_daily:,.0f}")
col3.metric("Pengguna Kasual %", f"{casual_pct:.1f}%")
col4.metric("Pengguna Terdaftar %", f"{registered_pct:.1f}%")
col5.metric("Jam Puncak", f"{peak_hour}:00")

st.markdown("---")

# ================= TABS =================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Ringkasan",
    "⏰ Analisis Waktu",
    "🌤️ Dampak Cuaca",
    "👥 Segmentasi Pengguna",
    "🎯 Klasterisasi"
])

# ================= TAB 1 =================
with tab1:
    st.header("📊 Ringkasan & Insight Utama")

    col1, col2 = st.columns(2)

    with col1:
        daily_data = filtered_df.groupby('dteday')['cnt'].sum().reset_index()
        fig = px.line(daily_data, x='dteday', y='cnt',
                      title='Tren Peminjaman Harian',
                      labels={'dteday': 'Tanggal', 'cnt': 'Total Peminjaman'})
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = go.Figure()
        fig.add_trace(go.Box(y=filtered_df['cnt'], name='Total'))
        fig.add_trace(go.Box(y=filtered_df['casual'], name='Kasual'))
        fig.add_trace(go.Box(y=filtered_df['registered'], name='Terdaftar'))
        fig.update_layout(title='Distribusi Peminjaman Berdasarkan Tipe Pengguna')
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="insight-box">', unsafe_allow_html=True)
    st.subheader("🔍 Insight Bisnis Utama")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**🎯 Performa Musim**")
        best_season = filtered_df.groupby('season_label')['cnt'].mean().idxmax()
        best_season_avg = filtered_df.groupby('season_label')['cnt'].mean().max()
        st.write(f"• Musim Terbaik: {best_season}")
        st.write(f"• Rata-rata Peminjaman: {best_season_avg:.0f}")

    with col2:
        st.markdown("**⚡ Pola Penggunaan**")
        peak_hours = filtered_df.groupby('hr')['cnt'].mean().nlargest(3)
        st.write(f"• Jam Tertinggi: {', '.join([f'{h}:00' for h in peak_hours.index])}")

    with col3:
        st.markdown("**🌤️ Pengaruh Cuaca**")
        best_weather = filtered_df.groupby('weather_label')['cnt'].mean().idxmax()
        st.write(f"• Cuaca Terbaik: {best_weather}")

    st.markdown('</div>', unsafe_allow_html=True)
