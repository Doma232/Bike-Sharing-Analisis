## 🤝 Kontribusi Kelompok <br>
10124011 - Amhar Razka Ariyanto " Project Coordinator & Quality Control "<br> 
10124025 - Rafli Zeirofi Dokumentasi Notebook<br>
10124041 - Ryan Setiawan " Dashboard Streamlit " <br>
10124042 - Jaifar Husnayan Hakim " Analis Data Notebook "<br>
10124043 - Aditia Pramudia Video youtube<br>
10124475 - Daffa Rizky Ramadhan Github & Testing<br>

# 🚴 Bike Sharing Analytics Dashboard

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://bike-sharing-kelompok-04.streamlit.app/)
[URL-Youtube] (https://youtu.be/gzmOVKpgduE)

## 📋 Project Overview

Dashboard analisis data bike sharing yang komprehensif dengan visualisasi interaktif dan insights strategis untuk bisnis bike sharing. Project ini merupakan submission untuk UAS dengan implementasi teknik analisis data lanjutan.

### ✨Fitur Utama
- **Dashboard Interaktif**: Dashboard Streamlit dengan filter dinamis dan visualisasi real-time
- **Analisis Lanjutan**: 
  - Dekomposisi Time Series
  - K-Means Clustering
  - Principal Component Analysis (PCA)
  - Analisis Korelasi Statistik
  - Pattern Mining
- **Insight Komprehensif**: Insight bisnis dan rekomendasi strategis
- **Desain Responsif**: Tampilan ramah mobile
- **Filter Real-time**: Penyaringan data multi-dimensi

## 📊 Dataset

Dataset yang digunakan adalah **Bike Sharing Dataset (Hourly)** dengan 17,379 records dan 17 variabel:

**Variabel Waktu (Temporal):**
- `dteday`: Tanggal
- `season`: Musim (1-4)
- `yr`: Tahun (0: 2011, 1: 2012)
- `mnth`: Bulan (1-12)
- `hr`: Jam (0-23)
- `holiday`: Indikator hari libur
- `weekday`: Hari dalam minggu
- `workingday`: Indikator hari kerja

**Variabel Cuaca:**
- `weathersit`: Kondisi cuaca (1-4)
- `temp`: Suhu ternormalisasi
- `atemp`: Suhu terasa (feels-like temperature)
- `hum`: Kelembapan ternormalisasi
- `windspeed`: Kecepatan angin ternormalisasi

**Variabel Target:**
- `casual`: Jumlah pengguna casual
- `registered`: Jumlah pengguna terdaftar
- `cnt`: Total penyewaan sepeda

## 🛠️ Teknologi yang Digunakan

- **Python 3.8+**
- **Data Analysis**: Pandas, NumPy
- **Visualization**: Plotly, Matplotlib, Seaborn
- **Machine Learning**: Scikit-learn
- **Statistical Analysis**: SciPy, Statsmodels
- **Dashboard**: Streamlit
- **Deployment**: Streamlit Cloud

## 📁 Struktur Proyek

```
bike-sharing-analysis/
│
├── Dashboard/
│   └── dashboard.py                 # Streamlit dashboard application
│
├── Dataset/
│   ├── cluster_analysis.csv         # Clustering results
│   ├── hour.csv                     # Raw dataset
│   ├── processed_daily.csv          # Processed daily data
│   └── processed_hour.csv           # Processed hourly data
│
├── Notebook/
│   └── bike_sharing_analysis.ipynb  # Jupyter notebook analysis
│
├── requirements.txt                 # Python dependencies
└── README.md                        # Project documentation

```

## 📈 Metodologi Analisis

### 1. Data Preprocessing & Feature Engineering
- Pembersihan dan validasi data
- Penanganan missing value
- Pembuatan fitur baru (time of day, indikator rush hour, kategori suhu)
- Ekstraksi fitur tanggal dan waktu

### 2. Exploratory Data Analysis (EDA)
- Distribusi variabel target
- Analisis pola waktu
- Analisis musiman
- Analisis pengaruh cuaca
- Analisis korelasi

### 3. Analisis Lanjutan

#### Analisis Time Series
- Identifikasi tren
- Dekomposisi musiman
- Moving averages
- Deteksi pola

#### Analisis Clustering (K-Means)
- Penentuan jumlah cluster optimal (Metode Elbow)
- Segmentasi pengguna
- PCA untuk reduksi dimensi
- Profiling cluster

#### Analisis Statistik
- Korelasi Pearson dengan uji signifikansi
- Pengujian hipotesis
- Analisis distribusi

### 4. Insight Bisnis & Rekomendasi
- Identifikasi jam puncak penggunaan
- Analisis pola perilaku pengguna
- Strategi berbasis kondisi cuaca
- Rekomendasi optimasi penggunaan sumber daya

## 📊 Temuan Utama

### 🎯 Pola Penggunaan
- **Jam Puncak**: 07–09 pagi dan 17–19 sore (jam kerja)
- **Musim Terbaik**: Musim gugur memiliki rata-rata penyewaan tertinggi
- **Distribusi Pengguna**: ±80% pengguna registered, ±20% pengguna casual

### 🌤️ Pengaruh Cuaca
- Korelasi positif kuat dengan suhu (r > 0.4)
- Cuaca cerah menghasilkan penyewaan 3–4x lebih banyak dibanding hujan
- Rentang suhu optimal: 15–25°C

### 👥 Segmentasi Pengguna
- **Pengguna Registered**: Pola commuting konsisten, lebih tahan terhadap cuaca
- **Pengguna Casual**: Dominan akhir pekan, sensitif terhadap cuaca

### 🎯 Insight Clustering
- Teridentifikasi 4 cluster jam penggunaan
- Permintaan tinggi: (07–09 & 17–19)
- Permintaan rendah: (00–05)
- Permintaan sedang: siang hingga malam

## 💡 Rekomendasi Bisnis

1. **Optimasi Sumber Daya**
   - Tambah sepeda saat jam puncak (07-09, 17-19)
   - Hindari maintenance saat demand tinggi

2. **Dynamic Pricing**
   - Harga dinamis saat rush hour
   - Diskon saat off-peak

3. **Manajemen Armada**
   - Konversi pengguna kasual menjadi pengguna tetap
   - Promosi berbasis cuaca
   - Kampanye musiman

4. **Fleet Management**
   - Redistribusi sepeda berdasarkan cluster
   - Penjadwalan maintenance prediktif

5. **User Experience**
   - Notifikasi cuaca untuk pengguna
   - Program loyalitas untuk pengguna registered 
   - Paket akhir pekan untuk penguna casual 

## 📸 Tampilan Dashboard

*Dashboard akan menampilkan:*
- Key Performance Metrics
- Grafik time series interaktif
- Heatmap analisis pola
- Visualisasi pengaruh cuaca
- Analisis segmentasi pengguna
- Visualisasi clustering berbasis PCA

  
