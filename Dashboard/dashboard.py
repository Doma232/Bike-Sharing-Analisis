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

# Page configuration
st.set_page_config(
    page_title="Bike Sharing Analytics Dashboard",
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
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
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

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('Dataset/processed_hour.csv')
    
    # Data preprocessing
    df['dteday'] = pd.to_datetime(df['dteday'])
    df['datetime'] = df['dteday'] + pd.to_timedelta(df['hr'], unit='h')
    
    # Feature engineering
    df['year'] = df['dteday'].dt.year
    df['month'] = df['dteday'].dt.month
    df['day_of_week'] = df['dteday'].dt.dayofweek
    
    # Labels
    season_labels = {1: 'Semi', 2: 'Hujan', 3: 'Panas', 4: 'Salju'}
    weather_labels = {1: 'Ekstrem', 2: 'Kabut', 3: 'Salju Ringan/Hujan', 4: 'Dingin'}
    weekday_labels = {0: 'Senin', 1: 'Selasa', 2: 'Rabu', 3: 'Kamis', 
                      4: 'Jumat', 5: 'Sabtu', 6: 'Minggu'}
    
    df['season_label'] = df['season'].map(season_labels)
    df['weather_label'] = df['weathersit'].map(weather_labels)
    df['weekday_label'] = df['weekday'].map(weekday_labels)
    
    # Time of day
    def categorize_hour(hour):
        if 6 <= hour < 12:
            return 'Morning'
        elif 12 <= hour < 18:
            return 'Afternoon'
        elif 18 <= hour < 24:
            return 'Evening'
        else:
            return 'Night'
    
    df['time_of_day'] = df['hr'].apply(categorize_hour)
    df['is_rush_hour'] = df['hr'].apply(lambda x: 1 if x in [7, 8, 17, 18] else 0)
    
    # Temperature in Celsius
    df['temp_celsius'] = df['temp'] * 41 - 8
    df['atemp_celsius'] = df['atemp'] * 50 - 16
    
    return df

# Load data
with st.spinner('Loading data...'):
    df = load_data()

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/bicycle.png", width=80)
    st.title("🚴 Bike Sharing Analytics")
    
    st.markdown("---")
    
    # Filters
    st.header("Filters")
    
    # Year filter
    years = sorted(df['year'].unique())
    selected_year = st.multiselect("Pilih Tahun", years, default=years)
    
    # Season filter
    seasons = df['season_label'].unique()
    selected_season = st.multiselect("Pilih Musim", seasons, default=seasons)
    
    # Weather filter
    weather_conditions = df['weather_label'].unique()
    selected_weather = st.multiselect("Pilih Cuaca", weather_conditions, default=weather_conditions)
    
    # Working day filter
    working_day_option = st.radio("Pilih Hari", ["Semua", "Hari Kerja", "Hari Libur"])
    
    st.markdown("---")
    
    # Info
    st.info("""
    **Tentang Dasbor Ini**

    Dasbor interaktif ini menyediakan analisis tentang pola Bike Sharing, termasuk:
    - Temporal trends
    - Weather impact
    - User segmentation
    - Clustering analysis
    """)
    
    st.markdown("---")
    st.caption("Created with ❤️ using Streamlit")

# Apply filters
filtered_df = df.copy()

if selected_year:
    filtered_df = filtered_df[filtered_df['year'].isin(selected_year)]

if selected_season:
    filtered_df = filtered_df[filtered_df['season_label'].isin(selected_season)]

if selected_weather:
    filtered_df = filtered_df[filtered_df['weather_label'].isin(selected_weather)]

if working_day_option == "Working Day":
    filtered_df = filtered_df[filtered_df['workingday'] == 1]
elif working_day_option == "Holiday":
    filtered_df = filtered_df[filtered_df['workingday'] == 0]

# Main content
st.markdown('<h1 class="main-header">🚴 Bike Sharing Analytics Dashboard</h1>', unsafe_allow_html=True)

st.markdown("---")

# Key Metrics
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    total_rentals = filtered_df['cnt'].sum()
    st.metric("Total Rentals", f"{total_rentals:,}")

with col2:
    avg_daily = filtered_df.groupby('dteday')['cnt'].sum().mean()
    st.metric("Avg Daily Rentals", f"{avg_daily:,.0f}")

with col3:
    casual_pct = (filtered_df['casual'].sum() / total_rentals * 100)
    st.metric("Casual Users %", f"{casual_pct:.1f}%")

with col4:
    registered_pct = (filtered_df['registered'].sum() / total_rentals * 100)
    st.metric("Registered Users %", f"{registered_pct:.1f}%")

with col5:
    peak_hour = filtered_df.groupby('hr')['cnt'].mean().idxmax()
    st.metric("Peak Hour", f"{peak_hour}:00")

st.markdown("---")

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Overview", "⏰ Temporal Analysis", "🌤️ Weather Impact", "👥 User Segmentation", "🎯 Clustering"])

# TAB 1: Overview
with tab1:
    st.header("📊 Overview & Key Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Time series
        daily_data = filtered_df.groupby('dteday')['cnt'].sum().reset_index()
        
        fig = px.line(daily_data, x='dteday', y='cnt', 
                     title='Daily Rental Trend',
                     labels={'dteday': 'Date', 'cnt': 'Total Rentals'})
        fig.update_traces(line_color='#1f77b4', line_width=2)
        fig.update_layout(height=400, hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Rental distribution
        fig = go.Figure()
        fig.add_trace(go.Box(y=filtered_df['cnt'], name='Total', marker_color='lightblue'))
        fig.add_trace(go.Box(y=filtered_df['casual'], name='Casual', marker_color='lightcoral'))
        fig.add_trace(go.Box(y=filtered_df['registered'], name='Registered', marker_color='lightgreen'))
        
        fig.update_layout(
            title='Rental Distribution by User Type',
            yaxis_title='Number of Rentals',
            height=400,
            showlegend=True
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Key Insights
    st.markdown('<div class="insight-box">', unsafe_allow_html=True)
    st.subheader("🔍 Key Business Insights")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**🎯 Peak Performance**")
        best_season = filtered_df.groupby('season_label')['cnt'].mean().idxmax()
        best_season_avg = filtered_df.groupby('season_label')['cnt'].mean().max()
        st.write(f"• Best Season: {best_season}")
        st.write(f"• Avg Rentals: {best_season_avg:.0f}")
    
    with col2:
        st.markdown("**⚡ Usage Patterns**")
        peak_hours = filtered_df.groupby('hr')['cnt'].mean().nlargest(3)
        st.write(f"• Top Hours: {', '.join([f'{h}:00' for h in peak_hours.index])}")
        st.write(f"• Rush Hour Impact: +{filtered_df[filtered_df['is_rush_hour']==1]['cnt'].mean() / filtered_df[filtered_df['is_rush_hour']==0]['cnt'].mean():.1%}")
    
    with col3:
        st.markdown("**🌤️ Weather Impact**")
        best_weather = filtered_df.groupby('weather_label')['cnt'].mean().idxmax()
        weather_impact = (filtered_df[filtered_df['weather_label']=='Clear']['cnt'].mean() / 
                         filtered_df[filtered_df['weather_label']=='Light Snow/Rain']['cnt'].mean())
        st.write(f"• Best Weather: {best_weather}")
        st.write(f"• Clear vs Rain: {weather_impact:.1f}x more")
    
    st.markdown('</div>', unsafe_allow_html=True)

# TAB 2: Temporal Analysis
with tab2:
    st.header("⏰ Temporal Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Hourly pattern
        hourly_avg = filtered_df.groupby('hr')[['casual', 'registered', 'cnt']].mean().reset_index()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hourly_avg['hr'], y=hourly_avg['casual'], 
                                name='Casual', mode='lines+markers', line=dict(color='coral', width=3)))
        fig.add_trace(go.Scatter(x=hourly_avg['hr'], y=hourly_avg['registered'], 
                                name='Registered', mode='lines+markers', line=dict(color='skyblue', width=3)))
        fig.add_trace(go.Scatter(x=hourly_avg['hr'], y=hourly_avg['cnt'], 
                                name='Total', mode='lines', line=dict(color='green', width=2, dash='dash')))
        
        fig.update_layout(
            title='Average Rentals by Hour',
            xaxis_title='Hour of Day',
            yaxis_title='Average Rentals',
            hovermode='x unified',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Day of week pattern
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        daily_avg = filtered_df.groupby('weekday_label')[['casual', 'registered', 'cnt']].mean().reindex(day_order)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(x=daily_avg.index, y=daily_avg['casual'], 
                            name='Casual', marker_color='coral'))
        fig.add_trace(go.Bar(x=daily_avg.index, y=daily_avg['registered'], 
                            name='Registered', marker_color='skyblue'))
        
        fig.update_layout(
            title='Average Rentals by Day of Week',
            xaxis_title='Day',
            yaxis_title='Average Rentals',
            barmode='stack',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Heatmap
    st.subheader("📅 Hourly Pattern Heatmap")
    
    pivot_data = filtered_df.pivot_table(values='cnt', index='hr', columns='weekday_label', aggfunc='mean')
    pivot_data = pivot_data[day_order] if all(day in pivot_data.columns for day in day_order) else pivot_data
    
    fig = px.imshow(pivot_data,
                    labels=dict(x="Day of Week", y="Hour of Day", color="Avg Rentals"),
                    x=pivot_data.columns,
                    y=pivot_data.index,
                    color_continuous_scale='YlOrRd',
                    aspect="auto")
    
    fig.update_layout(height=500, title='Rental Intensity Heatmap')
    st.plotly_chart(fig, use_container_width=True)
    
    # Monthly trend
    st.subheader("📈 Monthly Trend Analysis")
    
    monthly_data = filtered_df.groupby(['year', 'month']).agg({
        'cnt': 'sum',
        'casual': 'sum',
        'registered': 'sum'
    }).reset_index()
    
    monthly_data['year_month'] = monthly_data['year'].astype(str) + '-' + monthly_data['month'].astype(str).str.zfill(2)
    
    fig = px.line(monthly_data, x='year_month', y=['casual', 'registered', 'cnt'],
                 title='Monthly Rental Trend by User Type',
                 labels={'value': 'Total Rentals', 'year_month': 'Year-Month', 'variable': 'User Type'})
    
    fig.update_layout(height=400, hovermode='x unified')
    st.plotly_chart(fig, use_container_width=True)

# TAB 3: Weather Impact
with tab3:
    st.header("🌤️ Weather Impact Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Weather situation comparison
        weather_avg = filtered_df.groupby('weather_label')[['casual', 'registered', 'cnt']].mean().reset_index()
        
        fig = go.Figure()
        fig.add_trace(go.Bar(x=weather_avg['weather_label'], y=weather_avg['casual'],
                            name='Casual', marker_color='coral'))
        fig.add_trace(go.Bar(x=weather_avg['weather_label'], y=weather_avg['registered'],
                            name='Registered', marker_color='skyblue'))
        
        fig.update_layout(
            title='Average Rentals by Weather Condition',
            xaxis_title='Weather',
            yaxis_title='Average Rentals',
            barmode='group',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Season comparison
        season_order = ['Spring', 'Summer', 'Fall', 'Winter']
        season_avg = filtered_df.groupby('season_label')['cnt'].mean().reindex(season_order)
        
        fig = go.Figure(data=[go.Pie(labels=season_avg.index, values=season_avg.values,
                                     hole=.3, marker_colors=['#ff9999', '#66b3ff', '#99ff99', '#ffcc99'])])
        
        fig.update_layout(
            title='Average Rentals Distribution by Season',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Temperature analysis
    st.subheader("🌡️ Temperature Impact")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.scatter(filtered_df, x='temp_celsius', y='cnt', 
                        color='season_label',
                        title='Temperature vs Rentals',
                        labels={'temp_celsius': 'Temperature (°C)', 'cnt': 'Total Rentals'},
                        opacity=0.5)
        
        # Add trendline
        z = np.polyfit(filtered_df['temp_celsius'], filtered_df['cnt'], 2)
        p = np.poly1d(z)
        temp_range = np.linspace(filtered_df['temp_celsius'].min(), filtered_df['temp_celsius'].max(), 100)
        
        fig.add_trace(go.Scatter(x=temp_range, y=p(temp_range),
                                mode='lines', name='Trend',
                                line=dict(color='red', width=3, dash='dash')))
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Humidity impact
        fig = px.scatter(filtered_df, x='hum', y='cnt',
                        color='weather_label',
                        title='Humidity vs Rentals',
                        labels={'hum': 'Humidity (normalized)', 'cnt': 'Total Rentals'},
                        opacity=0.5)
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Weather statistics
    st.subheader("📊 Weather Statistics")
    
    weather_stats = filtered_df.groupby('weather_label').agg({
        'cnt': ['mean', 'min', 'max', 'std'],
        'temp_celsius': 'mean',
        'hum': 'mean',
        'windspeed': 'mean'
    }).round(2)
    
    weather_stats.columns = ['Avg Rentals', 'Min Rentals', 'Max Rentals', 'Std Dev', 'Avg Temp (°C)', 'Avg Humidity', 'Avg Windspeed']
    st.dataframe(weather_stats, use_container_width=True)

# TAB 4: User Segmentation
with tab4:
    st.header("👥 User Segmentation Analysis")
    
    # Casual vs Registered comparison
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # User type by hour
        hourly_users = filtered_df.groupby('hr')[['casual', 'registered']].mean().reset_index()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hourly_users['hr'], y=hourly_users['casual'],
                                fill='tozeroy', name='Casual',
                                line=dict(color='coral')))
        fig.add_trace(go.Scatter(x=hourly_users['hr'], y=hourly_users['registered'],
                                fill='tozeroy', name='Registered',
                                line=dict(color='skyblue')))
        
        fig.update_layout(
            title='User Type Distribution by Hour',
            xaxis_title='Hour of Day',
            yaxis_title='Average Rentals',
            hovermode='x unified',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Pie chart for total distribution
        total_casual = filtered_df['casual'].sum()
        total_registered = filtered_df['registered'].sum()
        
        fig = go.Figure(data=[go.Pie(
            labels=['Casual', 'Registered'],
            values=[total_casual, total_registered],
            hole=.4,
            marker_colors=['coral', 'skyblue']
        )])
        
        fig.update_layout(
            title='Total User Distribution',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Working day vs Weekend
    st.subheader("📅 Working Day vs Weekend/Holiday Behavior")
    
    col1, col2 = st.columns(2)
    
    with col1:
        workday_hourly = filtered_df[filtered_df['workingday']==1].groupby('hr')[['casual', 'registered']].mean()
        
        fig = go.Figure()
        fig.add_trace(go.Bar(x=workday_hourly.index, y=workday_hourly['casual'],
                            name='Casual', marker_color='coral'))
        fig.add_trace(go.Bar(x=workday_hourly.index, y=workday_hourly['registered'],
                            name='Registered', marker_color='skyblue'))
        
        fig.update_layout(
            title='Working Day Pattern',
            xaxis_title='Hour',
            yaxis_title='Average Rentals',
            barmode='stack',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        holiday_hourly = filtered_df[filtered_df['workingday']==0].groupby('hr')[['casual', 'registered']].mean()
        
        fig = go.Figure()
        fig.add_trace(go.Bar(x=holiday_hourly.index, y=holiday_hourly['casual'],
                            name='Casual', marker_color='coral'))
        fig.add_trace(go.Bar(x=holiday_hourly.index, y=holiday_hourly['registered'],
                            name='Registered', marker_color='skyblue'))
        
        fig.update_layout(
            title='Weekend/Holiday Pattern',
            xaxis_title='Hour',
            yaxis_title='Average Rentals',
            barmode='stack',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # User behavior insights
    st.markdown('<div class="insight-box">', unsafe_allow_html=True)
    st.subheader("💡 User Behavior Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Registered Users (Commuters)**")
        st.write("• Peak at 7-9 AM and 5-7 PM")
        st.write("• Higher usage on working days")
        st.write("• Consistent daily patterns")
        st.write("• Weather-resistant behavior")
    
    with col2:
        st.markdown("**Casual Users (Leisure)**")
        st.write("• Higher usage on weekends")
        st.write("• Peak at midday 12-4 PM")
        st.write("• More weather-sensitive")
        st.write("• Seasonal variations")
    
    st.markdown('</div>', unsafe_allow_html=True)

# TAB 5: Clustering Analysis
with tab5:
    st.header("🎯 Advanced Clustering Analysis")
    
    # Prepare clustering data
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
    
    # Number of clusters selector
    n_clusters = st.slider("Select number of clusters", min_value=2, max_value=8, value=4)
    
    cluster_df, variance_ratio = perform_clustering(filtered_df, n_clusters)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # PCA visualization
        fig = px.scatter(cluster_df, x='pca1', y='pca2', color='cluster',
                        text='hr', size='cnt',
                        title=f'K-Means Clustering (k={n_clusters})',
                        labels={'pca1': f'PC1 ({variance_ratio[0]:.1%})',
                               'pca2': f'PC2 ({variance_ratio[1]:.1%})'},
                        color_continuous_scale='viridis')
        
        fig.update_traces(textposition='top center')
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        st.info(f"Total variance explained: {sum(variance_ratio):.1%}")
    
    with col2:
        # Cluster characteristics
        st.subheader("Cluster Characteristics")
        
        for cluster_id in sorted(cluster_df['cluster'].unique()):
            cluster_data = cluster_df[cluster_df['cluster'] == cluster_id]
            hours = sorted(cluster_data['hr'].tolist())
            
            with st.expander(f"📊 Cluster {cluster_id} (Hours: {hours})"):
                st.write(f"**Hours:** {', '.join(map(str, hours))}")
                st.write(f"**Avg Rentals:** {cluster_data['cnt'].mean():.0f}")
                st.write(f"**Avg Casual:** {cluster_data['casual'].mean():.0f}")
                st.write(f"**Avg Registered:** {cluster_data['registered'].mean():.0f}")
                st.write(f"**Avg Temperature:** {cluster_data['temp'].mean():.2f}")
                
                # Characteristic
                if cluster_data['cnt'].mean() > filtered_df.groupby('hr')['cnt'].mean().mean() * 1.5:
                    st.success("🔥 **High Demand Period**")
                elif cluster_data['cnt'].mean() < filtered_df.groupby('hr')['cnt'].mean().mean() * 0.5:
                    st.warning("❄️ **Low Demand Period**")
                else:
                    st.info("⚖️ **Moderate Demand Period**")
    
    # Cluster comparison
    st.subheader("📊 Cluster Comparison")
    
    cluster_summary = cluster_df.groupby('cluster').agg({
        'cnt': 'mean',
        'casual': 'mean',
        'registered': 'mean',
        'temp': 'mean',
        'hum': 'mean'
    }).round(2)
    
    cluster_summary.columns = ['Avg Total', 'Avg Casual', 'Avg Registered', 'Avg Temp', 'Avg Humidity']
    
    st.dataframe(cluster_summary, use_container_width=True)
    
    # Recommendations based on clusters
    st.markdown('<div class="insight-box">', unsafe_allow_html=True)
    st.subheader("🎯 Strategic Recommendations")
    st.markdown("""
    Based on clustering analysis:
    
    1. **Resource Allocation**: Deploy more bikes during high-demand cluster hours
    2. **Dynamic Pricing**: Implement surge pricing during peak clusters
    3. **Maintenance Scheduling**: Plan maintenance during low-demand clusters
    4. **Marketing Strategy**: Target casual users during moderate-demand periods
    5. **Fleet Management**: Use cluster insights for optimal bike redistribution
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🚴 Bike Sharing Analytics Dashboard | Built with Streamlit & Plotly</p>
    <p>Data-driven insights for smart bike sharing operations</p>
</div>
""", unsafe_allow_html=True)
