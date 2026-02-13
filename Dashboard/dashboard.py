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

# Konfigurasi halaman
st.set_page_config(
    page_title="Dasbor Analitik Bike Sharing",
    page_icon="🚴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #667eea;
    }
    .insight-box {
        background-color: #e8f4f8;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Muat data
@st.cache_data
def load_data():
    df = pd.read_csv('Dataset/processed_hour.csv')
    
    # Preprocessing data
    df['dteday'] = pd.to_datetime(df['dteday'])
    df['datetime'] = df['dteday'] + pd.to_timedelta(df['hr'], unit='h')
    
    # Feature engineering
    df['year'] = df['dteday'].dt.year
    df['month'] = df['dteday'].dt.month
    df['day_of_week'] = df['dteday'].dt.dayofweek
    
    # Label dalam Bahasa Indonesia
    season_labels = {1: 'Semi', 2: 'Panas', 3: 'Gugur', 4: 'Dingin'}
    weather_labels = {1: 'Cerah', 2: 'Berawan', 3: 'Hujan/Salju Ringan', 4: 'Hujan/Salju Lebat'}
    weekday_labels = {0: 'Senin', 1: 'Selasa', 2: 'Rabu', 3: 'Kamis', 4: 'Jumat', 5: 'Sabtu', 6: 'Minggu'}
    
    df['season_label'] = df['season'].map(season_labels)
    df['weather_label'] = df['weathersit'].map(weather_labels)
    df['weekday_label'] = df['weekday'].map(weekday_labels)
    
    # Waktu dalam sehari
    def categorize_hour(hour):
        if 6 <= hour < 12:
            return 'Pagi'
        elif 12 <= hour < 18:
            return 'Siang'
        elif 18 <= hour < 24:
            return 'Sore'
        else:
            return 'Malam'
    
    df['time_of_day'] = df['hr'].apply(categorize_hour)
    df['is_rush_hour'] = df['hr'].apply(lambda x: 1 if x in [7, 8, 17, 18] else 0)
    
    # Suhu dalam Celsius
    df['temp_celsius'] = df['temp'] * 41 - 8
    df['atemp_celsius'] = df['atemp'] * 50 - 16
    
    return df

# Muat data
with st.spinner('Memuat data...'):
    df = load_data()

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/bicycle.png", width=80)
    st.title("🚴 Analitik Bike Sharing")
    st.markdown("---")
    
    # Filter
    st.header("Filter Data")
    
    # Filter tahun
    years = sorted(df['year'].unique())
    selected_year = st.multiselect("Pilih Tahun", years, default=years)
    
    # Filter musim
    seasons = df['season_label'].unique()
    selected_season = st.multiselect("Pilih Musim", seasons, default=seasons)
    
    # Filter cuaca
    weather_conditions = df['weather_label'].unique()
    selected_weather = st.multiselect("Pilih Kondisi Cuaca", weather_conditions, default=weather_conditions)
    
    # Filter hari kerja
    working_day_option = st.radio("Tipe Hari", ["Semua", "Hari Kerja", "Hari Libur"])
    
    st.markdown("---")
    
    # Info
    st.info("""
    **Tentang Dasbor Ini**
    
    Dasbor interaktif ini menyediakan analisis mendalam tentang pola penggunaan Bike Sharing, meliputi:
    - Tren temporal
    - Dampak cuaca
    - Segmentasi pengguna
    - Analisis clustering
    """)
    
    st.markdown("---")
    st.caption("Dibuat dengan ❤️ menggunakan Streamlit")

# Terapkan filter
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

# Konten utama
st.markdown('<div class="main-header">🚴 Dasbor Analitik Bike Sharing</div>', unsafe_allow_html=True)
st.markdown("---")

# Metrik utama
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    total_rentals = filtered_df['cnt'].sum()
    st.metric("Total Penyewaan", f"{total_rentals:,}")

with col2:
    avg_daily = filtered_df.groupby('dteday')['cnt'].sum().mean()
    st.metric("Rata-rata Harian", f"{avg_daily:,.0f}")

with col3:
    casual_pct = (filtered_df['casual'].sum() / total_rentals * 100)
    st.metric("Pengguna Kasual %", f"{casual_pct:.1f}%")

with col4:
    registered_pct = (filtered_df['registered'].sum() / total_rentals * 100)
    st.metric("Pengguna Terdaftar %", f"{registered_pct:.1f}%")

with col5:
    peak_hour = filtered_df.groupby('hr')['cnt'].mean().idxmax()
    st.metric("Jam Puncak", f"{peak_hour}:00")

st.markdown("---")

# Tab
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Ikhtisar", 
    "⏰ Analisis Temporal", 
    "🌤️ Dampak Cuaca", 
    "👥 Segmentasi Pengguna", 
    "🎯 Clustering"
])

# TAB 1: Ikhtisar
with tab1:
    st.header("📊 Ikhtisar & Wawasan Utama")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Tren harian
        daily_data = filtered_df.groupby('dteday')['cnt'].sum().reset_index()
        fig = px.line(daily_data, x='dteday', y='cnt', 
                     title='Tren Penyewaan Harian',
                     labels={'dteday': 'Tanggal', 'cnt': 'Total Penyewaan'})
        fig.update_traces(line_color='#1f77b4', line_width=2)
        fig.update_layout(height=400, hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Distribusi penyewaan
        fig = go.Figure()
        fig.add_trace(go.Box(y=filtered_df['cnt'], name='Total', marker_color='lightblue'))
        fig.add_trace(go.Box(y=filtered_df['casual'], name='Kasual', marker_color='lightcoral'))
        fig.add_trace(go.Box(y=filtered_df['registered'], name='Terdaftar', marker_color='lightgreen'))
        fig.update_layout(
            title='Distribusi Penyewaan berdasarkan Tipe Pengguna',
            yaxis_title='Jumlah Penyewaan',
            height=400,
            showlegend=True
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Wawasan bisnis utama
    st.markdown('<div class="insight-box">', unsafe_allow_html=True)
    st.subheader("🔍 Wawasan Bisnis Utama")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**🎯 Kinerja Puncak**")
        best_season = filtered_df.groupby('season_label')['cnt'].mean().idxmax()
        best_season_avg = filtered_df.groupby('season_label')['cnt'].mean().max()
        st.write(f"• Musim Terbaik: {best_season}")
        st.write(f"• Rata-rata Penyewaan: {best_season_avg:.0f}")
    
    with col2:
        st.markdown("**⚡ Pola Penggunaan**")
        peak_hours = filtered_df.groupby('hr')['cnt'].mean().nlargest(3)
        st.write(f"• Jam Teratas: {', '.join([f'{h}:00' for h in peak_hours.index])}")
        rush_impact = filtered_df[filtered_df['is_rush_hour']==1]['cnt'].mean() / filtered_df[filtered_df['is_rush_hour']==0]['cnt'].mean()
        st.write(f"• Dampak Jam Sibuk: +{rush_impact:.1%}")
    
    with col3:
        st.markdown("**🌤️ Dampak Cuaca**")
        best_weather = filtered_df.groupby('weather_label')['cnt'].mean().idxmax()
        clear_avg = filtered_df[filtered_df['weather_label']=='Cerah']['cnt'].mean()
        rain_avg = filtered_df[filtered_df['weather_label']=='Hujan/Salju Ringan']['cnt'].mean()
        if rain_avg > 0:
            weather_impact = clear_avg / rain_avg
            st.write(f"• Cuaca Terbaik: {best_weather}")
            st.write(f"• Cerah vs Hujan: {weather_impact:.1f}x lebih banyak")
        else:
            st.write(f"• Cuaca Terbaik: {best_weather}")
    
    st.markdown('</div>', unsafe_allow_html=True)

# TAB 2: Analisis Temporal
with tab2:
    st.header("⏰ Analisis Temporal")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Pola per jam
        hourly_avg = filtered_df.groupby('hr')[['casual', 'registered', 'cnt']].mean().reset_index()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hourly_avg['hr'], y=hourly_avg['casual'], 
                                name='Kasual', mode='lines+markers',
                                line=dict(color='coral', width=3)))
        fig.add_trace(go.Scatter(x=hourly_avg['hr'], y=hourly_avg['registered'], 
                                name='Terdaftar', mode='lines+markers',
                                line=dict(color='skyblue', width=3)))
        fig.add_trace(go.Scatter(x=hourly_avg['hr'], y=hourly_avg['cnt'], 
                                name='Total', mode='lines',
                                line=dict(color='green', width=2, dash='dash')))
        fig.update_layout(
            title='Rata-rata Penyewaan per Jam',
            xaxis_title='Jam dalam Sehari',
            yaxis_title='Rata-rata Penyewaan',
            hovermode='x unified',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Pola per hari
        day_order = ['Senin', 'Selasa', 'Rabu', 'Kamis', 'Jumat', 'Sabtu', 'Minggu']
        daily_avg = filtered_df.groupby('weekday_label')[['casual', 'registered', 'cnt']].mean().reindex(day_order)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(x=daily_avg.index, y=daily_avg['casual'], 
                            name='Kasual', marker_color='coral'))
        fig.add_trace(go.Bar(x=daily_avg.index, y=daily_avg['registered'], 
                            name='Terdaftar', marker_color='skyblue'))
        fig.update_layout(
            title='Rata-rata Penyewaan per Hari dalam Seminggu',
            xaxis_title='Hari',
            yaxis_title='Rata-rata Penyewaan',
            barmode='stack',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Heatmap
    st.subheader("📅 Peta Panas Pola Per Jam")
    pivot_data = filtered_df.pivot_table(values='cnt', index='hr', columns='weekday_label', aggfunc='mean')
    pivot_data = pivot_data[day_order] if all(day in pivot_data.columns for day in day_order) else pivot_data
    
    fig = px.imshow(pivot_data,
                    labels=dict(x="Hari dalam Seminggu", y="Jam dalam Sehari", color="Rata-rata Penyewaan"),
                    x=pivot_data.columns,
                    y=pivot_data.index,
                    color_continuous_scale='YlOrRd',
                    aspect="auto")
    fig.update_layout(height=500, title='Peta Intensitas Penyewaan')
    st.plotly_chart(fig, use_container_width=True)
    
    # Tren bulanan
    st.subheader("📈 Analisis Tren Bulanan")
    monthly_data = filtered_df.groupby(['year', 'month']).agg({
        'cnt': 'sum',
        'casual': 'sum',
        'registered': 'sum'
    }).reset_index()
    monthly_data['year_month'] = monthly_data['year'].astype(str) + '-' + monthly_data['month'].astype(str).str.zfill(2)
    
    fig = px.line(monthly_data, x='year_month', y=['casual', 'registered', 'cnt'],
                 title='Tren Penyewaan Bulanan berdasarkan Tipe Pengguna',
                 labels={'value': 'Total Penyewaan', 'year_month': 'Tahun-Bulan', 'variable': 'Tipe Pengguna'})
    fig.update_layout(height=400, hovermode='x unified')
    st.plotly_chart(fig, use_container_width=True)

# TAB 3: Dampak Cuaca
with tab3:
    st.header("🌤️ Analisis Dampak Cuaca")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Perbandingan kondisi cuaca
        weather_avg = filtered_df.groupby('weather_label')[['casual', 'registered', 'cnt']].mean().reset_index()
        fig = go.Figure()
        fig.add_trace(go.Bar(x=weather_avg['weather_label'], y=weather_avg['casual'],
                            name='Kasual', marker_color='coral'))
        fig.add_trace(go.Bar(x=weather_avg['weather_label'], y=weather_avg['registered'],
                            name='Terdaftar', marker_color='skyblue'))
        fig.update_layout(
            title='Rata-rata Penyewaan berdasarkan Kondisi Cuaca',
            xaxis_title='Cuaca',
            yaxis_title='Rata-rata Penyewaan',
            barmode='group',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Perbandingan musim
        season_order = ['Semi', 'Panas', 'Gugur', 'Dingin']
        season_avg = filtered_df.groupby('season_label')['cnt'].mean().reindex(season_order)
        
        fig = go.Figure(data=[go.Pie(labels=season_avg.index, values=season_avg.values,
                                     hole=.3, marker_colors=['#ff9999', '#66b3ff', '#99ff99', '#ffcc99'])])
        fig.update_layout(
            title='Distribusi Rata-rata Penyewaan berdasarkan Musim',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Analisis suhu
    st.subheader("🌡️ Dampak Suhu")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.scatter(filtered_df, x='temp_celsius', y='cnt', color='season_label',
                        title='Suhu vs Penyewaan',
                        labels={'temp_celsius': 'Suhu (°C)', 'cnt': 'Total Penyewaan'},
                        opacity=0.5)
        
        # Tambahkan garis tren
        z = np.polyfit(filtered_df['temp_celsius'], filtered_df['cnt'], 2)
        p = np.poly1d(z)
        temp_range = np.linspace(filtered_df['temp_celsius'].min(), filtered_df['temp_celsius'].max(), 100)
        fig.add_trace(go.Scatter(x=temp_range, y=p(temp_range),
                                mode='lines', name='Tren',
                                line=dict(color='red', width=3, dash='dash')))
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Dampak kelembaban
        fig = px.scatter(filtered_df, x='hum', y='cnt', color='weather_label',
                        title='Kelembaban vs Penyewaan',
                        labels={'hum': 'Kelembaban (ternormalisasi)', 'cnt': 'Total Penyewaan'},
                        opacity=0.5)
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Statistik cuaca
    st.subheader("📊 Statistik Cuaca")
    weather_stats = filtered_df.groupby('weather_label').agg({
        'cnt': ['mean', 'min', 'max', 'std'],
        'temp_celsius': 'mean',
        'hum': 'mean',
        'windspeed': 'mean'
    }).round(2)
    weather_stats.columns = ['Rata-rata Penyewaan', 'Min Penyewaan', 'Max Penyewaan', 
                             'Std Dev', 'Rata-rata Suhu (°C)', 'Rata-rata Kelembaban', 'Rata-rata Kecepatan Angin']
    st.dataframe(weather_stats, use_container_width=True)

# TAB 4: Segmentasi Pengguna
with tab4:
    st.header("👥 Analisis Segmentasi Pengguna")
    
    # Perbandingan Kasual vs Terdaftar
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Tipe pengguna per jam
        hourly_users = filtered_df.groupby('hr')[['casual', 'registered']].mean().reset_index()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hourly_users['hr'], y=hourly_users['casual'],
                                fill='tozeroy', name='Kasual',
                                line=dict(color='coral')))
        fig.add_trace(go.Scatter(x=hourly_users['hr'], y=hourly_users['registered'],
                                fill='tozeroy', name='Terdaftar',
                                line=dict(color='skyblue')))
        fig.update_layout(
            title='Distribusi Tipe Pengguna per Jam',
            xaxis_title='Jam dalam Sehari',
            yaxis_title='Rata-rata Penyewaan',
            hovermode='x unified',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Diagram lingkaran distribusi total
        total_casual = filtered_df['casual'].sum()
        total_registered = filtered_df['registered'].sum()
        
        fig = go.Figure(data=[go.Pie(
            labels=['Kasual', 'Terdaftar'],
            values=[total_casual, total_registered],
            hole=.4,
            marker_colors=['coral', 'skyblue']
        )])
        fig.update_layout(
            title='Distribusi Total Pengguna',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Hari kerja vs Akhir pekan
    st.subheader("📅 Perilaku Hari Kerja vs Akhir Pekan/Libur")
    
    col1, col2 = st.columns(2)
    
    with col1:
        workday_hourly = filtered_df[filtered_df['workingday']==1].groupby('hr')[['casual', 'registered']].mean()
        fig = go.Figure()
        fig.add_trace(go.Bar(x=workday_hourly.index, y=workday_hourly['casual'],
                            name='Kasual', marker_color='coral'))
        fig.add_trace(go.Bar(x=workday_hourly.index, y=workday_hourly['registered'],
                            name='Terdaftar', marker_color='skyblue'))
        fig.update_layout(
            title='Pola Hari Kerja',
            xaxis_title='Jam',
            yaxis_title='Rata-rata Penyewaan',
            barmode='stack',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        holiday_hourly = filtered_df[filtered_df['workingday']==0].groupby('hr')[['casual', 'registered']].mean()
        fig = go.Figure()
        fig.add_trace(go.Bar(x=holiday_hourly.index, y=holiday_hourly['casual'],
                            name='Kasual', marker_color='coral'))
        fig.add_trace(go.Bar(x=holiday_hourly.index, y=holiday_hourly['registered'],
                            name='Terdaftar', marker_color='skyblue'))
        fig.update_layout(
            title='Pola Akhir Pekan/Libur',
            xaxis_title='Jam',
            yaxis_title='Rata-rata Penyewaan',
            barmode='stack',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Wawasan perilaku pengguna
    st.markdown('<div class="insight-box">', unsafe_allow_html=True)
    st.subheader("💡 Wawasan Perilaku Pengguna")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Pengguna Terdaftar (Komuter)**")
        st.write("• Puncak pada pukul 7-9 pagi dan 5-7 sore")
        st.write("• Penggunaan lebih tinggi pada hari kerja")
        st.write("• Pola harian yang konsisten")
        st.write("• Perilaku tahan terhadap cuaca")
    
    with col2:
        st.markdown("**Pengguna Kasual (Rekreasi)**")
        st.write("• Penggunaan lebih tinggi di akhir pekan")
        st.write("• Puncak pada siang hari pukul 12-4 sore")
        st.write("• Lebih sensitif terhadap cuaca")
        st.write("• Variasi musiman")
    
    st.markdown('</div>', unsafe_allow_html=True)

# TAB 5: Analisis Clustering
with tab5:
    st.header("🎯 Analisis Clustering Lanjutan")
    
    # Persiapkan data clustering
    @st.cache_data
    def perform_clustering(data, n_clusters=4):
        cluster_features = data.groupby('hr').agg({
            'cnt': 'mean',
            'casual': 'mean',
            'registered': 'mean',
            'temp': 'mean',
            'hum': 'mean',
            'windspeed': 'mean'
        }).reset_index()
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(cluster_features.drop('hr', axis=1))
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_features['cluster'] = kmeans.fit_predict(X_scaled)
        
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        cluster_features['pca1'] = X_pca[:, 0]
        cluster_features['pca2'] = X_pca[:, 1]
        
        return cluster_features, pca.explained_variance_ratio_
    
    # Pemilih jumlah cluster
    n_clusters = st.slider("Pilih jumlah cluster", min_value=2, max_value=8, value=4)
    
    cluster_df, variance_ratio = perform_clustering(filtered_df, n_clusters)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Visualisasi PCA
        fig = px.scatter(cluster_df, x='pca1', y='pca2', color='cluster',
                        text='hr', size='cnt',
                        title=f'K-Means Clustering (k={n_clusters})',
                        labels={'pca1': f'PC1 ({variance_ratio[0]:.1%})',
                               'pca2': f'PC2 ({variance_ratio[1]:.1%})'},
                        color_continuous_scale='viridis')
        fig.update_traces(textposition='top center')
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        st.info(f"Total varians dijelaskan: {sum(variance_ratio):.1%}")
    
    with col2:
        # Karakteristik cluster
        st.subheader("Karakteristik Cluster")
        
        for cluster_id in sorted(cluster_df['cluster'].unique()):
            cluster_data = cluster_df[cluster_df['cluster'] == cluster_id]
            hours = sorted(cluster_data['hr'].tolist())
            
            with st.expander(f"📊 Cluster {cluster_id} (Jam: {hours})"):
                st.write(f"**Jam:** {', '.join(map(str, hours))}")
                st.write(f"**Rata-rata Penyewaan:** {cluster_data['cnt'].mean():.0f}")
                st.write(f"**Rata-rata Kasual:** {cluster_data['casual'].mean():.0f}")
                st.write(f"**Rata-rata Terdaftar:** {cluster_data['registered'].mean():.0f}")
                st.write(f"**Rata-rata Suhu:** {cluster_data['temp'].mean():.2f}")
                
                # Karakteristik
                avg_demand = filtered_df.groupby('hr')['cnt'].mean().mean()
                if cluster_data['cnt'].mean() > avg_demand * 1.5:
                    st.success("🔥 **Periode Permintaan Tinggi**")
                elif cluster_data['cnt'].mean() < avg_demand * 0.5:
                    st.warning("❄️ **Periode Permintaan Rendah**")
                else:
                    st.info("⚖️ **Periode Permintaan Sedang**")
    
    # Perbandingan cluster
    st.subheader("📊 Perbandingan Cluster")
    cluster_summary = cluster_df.groupby('cluster').agg({
        'cnt': 'mean',
        'casual': 'mean',
        'registered': 'mean',
        'temp': 'mean',
        'hum': 'mean'
    }).round(2)
    cluster_summary.columns = ['Rata-rata Total', 'Rata-rata Kasual', 'Rata-rata Terdaftar', 
                               'Rata-rata Suhu', 'Rata-rata Kelembaban']
    st.dataframe(cluster_summary, use_container_width=True)
    
    # Rekomendasi berdasarkan cluster
    st.markdown('<div class="insight-box">', unsafe_allow_html=True)
    st.subheader("🎯 Rekomendasi Strategis")
    
    st.markdown("""
    Berdasarkan analisis clustering:
    
    1. **Alokasi Sumber Daya**: Kerahkan lebih banyak sepeda selama jam-jam dengan permintaan tinggi yang teridentifikasi di cluster
    2. **Penetapan Harga Dinamis**: Terapkan harga premium selama periode permintaan puncak
    3. **Penjadwalan Pemeliharaan**: Rencanakan pemeliharaan selama jam-jam permintaan rendah
    4. **Strategi Pemasaran**: Targetkan pengguna kasual selama periode permintaan sedang dengan promosi khusus
    5. **Manajemen Armada**: Gunakan wawasan cluster untuk redistribusi sepeda yang optimal antar lokasi
    6. **Prediksi Permintaan**: Manfaatkan pola cluster untuk perencanaan kapasitas yang lebih baik
    7. **Optimasi Operasional**: Sesuaikan tingkat staf berdasarkan karakteristik cluster periode waktu
    """)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <p style='font-size: 1.2rem;'>🚴 Dasbor Analitik Bike Sharing | Dibuat dengan Streamlit & Plotly</p>
        <p style='font-size: 0.9rem;'>Data diperbarui secara berkala untuk memberikan wawasan terkini</p>
    </div>
""", unsafe_allow_html=True)
