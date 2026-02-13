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
    df = pd.read_csv('Dataset/processed_hour.csv')

    df['dteday'] = pd.to_datetime(df['dteday'])
    df['datetime'] = df['dteday'] + pd.to_timedelta(df['hr'], unit='h')

    df['year'] = df['dteday'].dt.year
    df['month'] = df['dteday'].dt.month
    df['day_of_week'] = df['dteday'].dt.dayofweek

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

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/bicycle.png", width=80)
    st.title("🚴 Analisis Bike Sharing")

    st.markdown("---")
    st.header("Filter")

    years = sorted(df['year'].unique())
    selected_year = st.multiselect("Pilih Tahun", years, default=years)

    seasons = df['season_label'].unique()
    selected_season = st.multiselect("Pilih Musim", seasons, default=seasons)

    weather_conditions = df['weather_label'].unique()
    selected_weather = st.multiselect("Pilih Kondisi Cuaca", weather_conditions, default=weather_conditions)

    working_day_option = st.radio("Tipe Hari", ["Semua", "Hari Kerja", "Hari Libur"])

# =========================================================
# FILTER DATA
# =========================================================
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

# =========================================================
# HEADER
# =========================================================
st.markdown('<h1 style="text-align:center;">🚴 Dashboard Analisis Bike Sharing</h1>', unsafe_allow_html=True)

# =========================================================
# METRIC
# =========================================================
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

# =========================================================
# TAB
# =========================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Ringkasan",
    "⏰ Analisis Waktu",
    "🌤️ Dampak Cuaca",
    "👥 Segmentasi Pengguna",
    "🎯 Klasterisasi"
])

# =========================================================
# TAB 1 — RINGKASAN
# =========================================================
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

        fig.update_layout(
            title='Distribusi Peminjaman Berdasarkan Tipe Pengguna',
            yaxis_title='Jumlah Peminjaman'
        )
        st.plotly_chart(fig, use_container_width=True)

# =========================================================
# TAB 2 — ANALISIS WAKTU
# =========================================================
with tab2:
    st.header("⏰ Analisis Waktu")

    hourly_avg = filtered_df.groupby('hr')[['casual', 'registered', 'cnt']].mean().reset_index()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hourly_avg['hr'], y=hourly_avg['casual'], name='Kasual'))
    fig.add_trace(go.Scatter(x=hourly_avg['hr'], y=hourly_avg['registered'], name='Terdaftar'))
    fig.add_trace(go.Scatter(x=hourly_avg['hr'], y=hourly_avg['cnt'], name='Total'))

    fig.update_layout(
        title='Rata-rata Peminjaman per Jam',
        xaxis_title='Jam',
        yaxis_title='Rata-rata Peminjaman'
    )
    st.plotly_chart(fig, use_container_width=True)

# =========================================================
# TAB 3 — DAMPAK CUACA
# =========================================================
with tab3:
    st.header("🌤️ Analisis Dampak Cuaca")

    weather_avg = filtered_df.groupby('weather_label')[['casual', 'registered']].mean().reset_index()

    fig = go.Figure()
    fig.add_bar(x=weather_avg['weather_label'], y=weather_avg['casual'], name='Kasual')
    fig.add_bar(x=weather_avg['weather_label'], y=weather_avg['registered'], name='Terdaftar')

    fig.update_layout(
        title='Rata-rata Peminjaman Berdasarkan Cuaca',
        xaxis_title='Kondisi Cuaca',
        yaxis_title='Rata-rata Peminjaman'
    )
    st.plotly_chart(fig, use_container_width=True)

# =========================================================
# TAB 4 — SEGMENTASI
# =========================================================
with tab4:
    st.header("👥 Segmentasi Pengguna")

    total_casual = filtered_df['casual'].sum()
    total_registered = filtered_df['registered'].sum()

    fig = go.Figure(data=[go.Pie(
        labels=['Kasual', 'Terdaftar'],
        values=[total_casual, total_registered],
        hole=.4
    )])

    fig.update_layout(title='Proporsi Total Pengguna')
    st.plotly_chart(fig, use_container_width=True)

# =========================================================
# FOOTER
# =========================================================
st.markdown("---")
st.markdown(
    "<div style='text-align:center;'>🚴 Dashboard Analisis Bike Sharing — Streamlit</div>",
    unsafe_allow_html=True
)
