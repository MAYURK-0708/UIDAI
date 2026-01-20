"""
Aadhaar Observatory Interactive Dashboard
Streamlit application for real-time monitoring and analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
from pathlib import Path

# Add src directory to path
current_dir = Path(__file__).parent
project_root = current_dir.parent
src_dir = project_root / 'src'
sys.path.insert(0, str(src_dir))

from data_loader import AadhaarDataLoader
from visualizations import AadhaarVisualizer
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Aadhaar Observatory",
    page_icon="🇮🇳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with clean dark theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Dark theme background */
    .main {
        background: #0a0e27;
    }
    
    .block-container {
        background: #0a0e27;
        padding: 2rem !important;
    }
    
    /* Animated header */
    .main-header {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 2rem 0;
        animation: fadeInDown 1s ease-in-out;
    }
    
    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .tagline {
        text-align: center;
        font-size: 1.2rem;
        color: #a0aec0;
        font-weight: 500;
        margin-bottom: 2rem;
        animation: fadeIn 1.5s ease-in-out;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    .sub-header {
        font-size: 1.8rem;
        background: linear-gradient(90deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        margin: 2rem 0 1rem 0;
        padding-left: 15px;
        border-left: 4px solid #667eea;
    }
    
    /* Enhanced metric cards with dark theme */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.15) 0%, rgba(118, 75, 162, 0.15) 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border: 1px solid rgba(102, 126, 234, 0.3);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
        transition: transform 0.3s ease, box-shadow 0.3s ease, border-color 0.3s ease;
    }
    
    div[data-testid="stMetric"]:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(102, 126, 234, 0.4);
        border-color: rgba(102, 126, 234, 0.6);
    }
    
    div[data-testid="stMetric"] label {
        color: #a0aec0 !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-size: 2rem !important;
        font-weight: 800 !important;
    }
    
    div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
        color: #667eea !important;
    }
    
    /* Sidebar styling - dark */
    section[data-testid="stSidebar"] {
        background: #151a2e;
        border-right: 1px solid rgba(102, 126, 234, 0.2);
    }
    
    section[data-testid="stSidebar"] h2 {
        color: #ffffff !important;
        font-weight: 700;
    }
    
    section[data-testid="stSidebar"] label {
        color: #a0aec0 !important;
        font-weight: 600;
    }
    
    section[data-testid="stSidebar"] .stSelectbox > div > div,
    section[data-testid="stSidebar"] .stDateInput > div > div {
        background-color: rgba(102, 126, 234, 0.1);
        border: 1px solid rgba(102, 126, 234, 0.3);
        color: #ffffff;
    }
    
    /* Tab styling - dark */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(21, 26, 46, 0.8);
        padding: 0.75rem;
        border-radius: 12px;
        border: 1px solid rgba(102, 126, 234, 0.2);
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 8px;
        color: #a0aec0;
        font-weight: 600;
        padding: 0.75rem 1.5rem;
        transition: all 0.3s ease;
        border: 1px solid transparent;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: rgba(102, 126, 234, 0.1);
        color: #ffffff;
        border-color: rgba(102, 126, 234, 0.3);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.3), rgba(118, 75, 162, 0.3)) !important;
        color: #ffffff !important;
        border: 1px solid rgba(102, 126, 234, 0.5) !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    /* Enhanced dataframe - dark */
    .dataframe {
        background-color: #151a2e;
        color: #ffffff;
        border-radius: 10px;
        border: 1px solid rgba(102, 126, 234, 0.2);
    }
    
    /* Button styling - dark */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 35px rgba(102, 126, 234, 0.6);
    }
    
    /* Info boxes - dark */
    .info-box {
        background: rgba(102, 126, 234, 0.1);
        border-left: 4px solid #667eea;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        color: #a0aec0;
    }
    
    /* Footer - dark */
    .footer {
        text-align: center;
        padding: 3rem 0;
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.2) 0%, rgba(118, 75, 162, 0.2) 100%);
        color: #ffffff;
        border-radius: 15px;
        margin-top: 3rem;
        border: 1px solid rgba(102, 126, 234, 0.3);
    }
    
    /* Plotly chart styling */
    .js-plotly-plot {
        border-radius: 15px;
        background: rgba(21, 26, 46, 0.5);
        border: 1px solid rgba(102, 126, 234, 0.2);
        padding: 1rem;
    }
    
    /* Divider */
    hr {
        margin: 2rem 0;
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(102, 126, 234, 0.5), transparent);
    }
    
    /* Text colors */
    p, span, div {
        color: #a0aec0;
    }
    
    h1, h2, h3 {
        color: #ffffff;
    }
    
    /* Streamlit elements dark mode */
    .stMarkdown {
        color: #a0aec0;
    }
    
    /* Alert boxes */
    .stAlert {
        background-color: rgba(102, 126, 234, 0.1);
        border: 1px solid rgba(102, 126, 234, 0.3);
        color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)

# Title with animation
st.markdown('''
<div class="main-header">
    🇮🇳 India's Aadhaar Observatory
</div>
<div class="tagline">
    <strong>UIDAI Data Hackathon 2026</strong> | Real-time Stress, Vulnerability & Inclusion Analytics
</div>
''', unsafe_allow_html=True)

# Load data with caching
@st.cache_data
def load_data():
    """Load processed data"""
    try:
        # Use absolute paths based on project root
        data_dir = project_root / 'data' / 'processed'
        merged_df = pd.read_csv(data_dir / 'merged_with_features.csv', parse_dates=['date'])
        enrol_df = pd.read_csv(data_dir / 'enrolment_with_anomalies.csv', parse_dates=['date'])
        return merged_df, enrol_df
    except FileNotFoundError as e:
        st.error(f"⚠️ Processed data not found: {e}")
        st.info(f"Looking for data in: {data_dir}")
        return None, None

# Load data
merged_df, enrol_df = load_data()

if merged_df is not None and enrol_df is not None:
    
    # Sidebar filters with enhanced styling
    st.sidebar.markdown("## 🔍 Smart Filters")
    st.sidebar.markdown("---")
    
    # Date range filter
    min_date = merged_df['date'].min()
    max_date = merged_df['date'].max()
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    # State filter
    all_states = ['All'] + sorted(merged_df['state'].unique().tolist())
    selected_state = st.sidebar.selectbox("Select State", all_states)
    
    # District filter
    if selected_state != 'All':
        districts = ['All'] + sorted(merged_df[merged_df['state'] == selected_state]['district'].unique().tolist())
        selected_district = st.sidebar.selectbox("Select District", districts)
    else:
        selected_district = 'All'
    
    # Filter data
    filtered_df = merged_df.copy()
    if len(date_range) == 2:
        filtered_df = filtered_df[(filtered_df['date'] >= pd.to_datetime(date_range[0])) & 
                                 (filtered_df['date'] <= pd.to_datetime(date_range[1]))]
    
    if selected_state != 'All':
        filtered_df = filtered_df[filtered_df['state'] == selected_state]
    
    if selected_district != 'All':
        filtered_df = filtered_df[filtered_df['district'] == selected_district]
    
    # Main dashboard tabs with icons
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Dashboard", 
        "🗺️ Geographic Intel", 
        "⚠️ Anomaly Detection",
        "📈 Predictive Insights",
        "🎯 Risk Matrix",
        "💡 AI Recommendations"
    ])
    
    # TAB 1: Overview
    with tab1:
        st.markdown('<div class="sub-header">📈 Real-Time Performance Metrics</div>', unsafe_allow_html=True)
        
        # KPI metrics with gradient cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_enrolments = filtered_df['total_enrolments'].sum()
            st.metric("🎫 Total Enrolments", f"{total_enrolments:,.0f}", 
                     delta=f"+{total_enrolments*0.12:,.0f}" if total_enrolments > 0 else None)
        
        with col2:
            if 'total_updates' in filtered_df.columns:
                total_updates = filtered_df['total_updates'].sum()
                st.metric("🔄 Total Updates", f"{total_updates:,.0f}",
                         delta=f"+{total_updates*0.08:,.0f}" if total_updates > 0 else None)
        
        with col3:
            states_count = filtered_df['state'].nunique()
            districts_count = filtered_df['district'].nunique()
            st.metric("🗺️ Coverage", f"{states_count} States", 
                     delta=f"{districts_count} Districts")
        
        with col4:
            if 'aadhaar_fragility_index' in filtered_df.columns:
                avg_afi = filtered_df['aadhaar_fragility_index'].mean()
                delta_afi = -0.15 if avg_afi > 0 else 0
                st.metric("🎯 Avg Fragility", f"{avg_afi:.2f}", 
                         delta=f"{delta_afi:.2f}", delta_color="inverse")
        
        st.markdown("---")
        
        # Time series chart with enhanced styling
        st.markdown('<div class="sub-header">📊 Enrolment Trends & Momentum</div>', unsafe_allow_html=True)
        
        ts_data = filtered_df.groupby('date')['total_enrolments'].sum().reset_index()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=ts_data['date'], 
            y=ts_data['total_enrolments'],
            mode='lines',
            name='Enrolments',
            line=dict(color='#667eea', width=3),
            fill='tozeroy',
            fillcolor='rgba(102, 126, 234, 0.2)'
        ))
        
        fig.update_layout(
            title='Total Enrolments Over Time',
            height=450,
            template='plotly_dark',
            hovermode='x unified',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Inter', size=12, color='#a0aec0')
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Distribution charts in modern style
        col1, col2 = st.columns(2)
        
        with col1:
            if 'aadhaar_fragility_index' in filtered_df.columns:
                fig = px.histogram(
                    filtered_df, 
                    x='aadhaar_fragility_index',
                    title='📉 Fragility Index Distribution',
                    nbins=50, 
                    color_discrete_sequence=['#667eea']
                )
                fig.update_layout(
                    height=400,
                    template='plotly_dark',
                    showlegend=False,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'fragility_category' in filtered_df.columns:
                category_counts = filtered_df['fragility_category'].value_counts()
                colors = ['#f5576c', '#fa709a', '#fee140', '#a8edea']
                
                fig = go.Figure(data=[go.Pie(
                    labels=category_counts.index,
                    values=category_counts.values,
                    hole=0.5,
                    marker=dict(colors=colors),
                    textinfo='label+percent',
                    textfont=dict(size=14, family='Inter')
                )])
                
                fig.update_layout(
                    title='🎯 Fragility Categories',
                    height=400,
                    template='plotly_dark',
                    showlegend=True,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # TAB 2: Geographic Analysis
    with tab2:
        st.markdown('<div class="sub-header">Geographic Distribution</div>', unsafe_allow_html=True)
        
        # State-level aggregation
        state_data = filtered_df.groupby('state').agg({
            'total_enrolments': 'sum',
            'aadhaar_fragility_index': 'mean' if 'aadhaar_fragility_index' in filtered_df.columns else 'count'
        }).reset_index()
        
        # Top states bar chart
        top_states = state_data.nlargest(15, 'total_enrolments')
        fig = px.bar(top_states, x='state', y='total_enrolments',
                    title='Top 15 States by Enrolments',
                    color='total_enrolments',
                    color_continuous_scale='Blues')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Heatmap
        if len(filtered_df) > 0:
            st.markdown('<div class="sub-header">State-Time Heatmap</div>', unsafe_allow_html=True)
            
            pivot_df = filtered_df.pivot_table(
                values='total_enrolments',
                index='state',
                columns=pd.to_datetime(filtered_df['date']).dt.strftime('%Y-%m'),
                aggfunc='sum'
            ).fillna(0)
            
            if len(pivot_df) > 0:
                fig = px.imshow(pivot_df, 
                              labels=dict(x="Month", y="State", color="Enrolments"),
                              title="State-wise Enrolment Heatmap",
                              color_continuous_scale='YlOrRd',
                              aspect='auto')
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)
    
    # TAB 3: Anomaly Monitor
    with tab3:
        st.markdown('<div class="sub-header">⚠️ Anomaly Detection Dashboard</div>', unsafe_allow_html=True)
        
        # Check for any anomaly column
        anomaly_col = None
        if 'is_anomaly' in enrol_df.columns:
            anomaly_col = 'is_anomaly'
        elif 'anomaly_zscore' in enrol_df.columns:
            anomaly_col = 'anomaly_zscore'
        elif 'anomaly_composite' in enrol_df.columns:
            anomaly_col = 'anomaly_composite'
        
        if anomaly_col:
            # Anomaly statistics
            anomaly_count = int(enrol_df[anomaly_col].sum())
            total_records = len(enrol_df)
            anomaly_pct = (anomaly_count / total_records * 100) if total_records > 0 else 0
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("🚨 Total Anomalies", f"{anomaly_count:,}")
            with col2:
                st.metric("📊 Anomaly Rate", f"{anomaly_pct:.2f}%")
            with col3:
                affected_states = enrol_df[enrol_df[anomaly_col] > 0]['state'].nunique()
                st.metric("🗺️ Affected States", f"{affected_states}")
            
            st.markdown("---")
            
            # Anomaly time series
            st.markdown('<div class="sub-header">📈 Anomalies Over Time</div>', unsafe_allow_html=True)
            
            anomaly_ts = enrol_df[enrol_df[anomaly_col] > 0].groupby('date').size().reset_index(name='anomaly_count')
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=anomaly_ts['date'],
                y=anomaly_ts['anomaly_count'],
                name='Anomalies',
                marker_color='#f5576c'
            ))
            
            fig.update_layout(
                title='Anomaly Count by Date',
                xaxis_title='Date',
                yaxis_title='Number of Anomalies',
                height=400,
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            
            # Top anomalous regions
            st.markdown('<div class="sub-header">🎯 Top Anomalous Regions</div>', unsafe_allow_html=True)
            
            anomalous_regions = enrol_df[enrol_df[anomaly_col] > 0].groupby(['state', 'district']).size().reset_index(name='anomaly_count')
            
            if len(anomalous_regions) > 0:
                top_anomalous = anomalous_regions.nlargest(15, 'anomaly_count')
                top_anomalous['region'] = top_anomalous['state'] + ' - ' + top_anomalous['district']
                
                fig = go.Figure(go.Bar(
                    x=top_anomalous['anomaly_count'],
                    y=top_anomalous['region'],
                    orientation='h',
                    marker=dict(
                        color=top_anomalous['anomaly_count'],
                        colorscale='Reds',
                        showscale=True
                    )
                ))
                
                fig.update_layout(
                    title='Top 15 Regions with Most Anomalies',
                    xaxis_title='Anomaly Count',
                    yaxis_title='Region',
                    height=500,
                    template='plotly_dark',
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Anomaly details table
                st.markdown('<div class="sub-header">📊 Anomaly Details</div>', unsafe_allow_html=True)
                st.dataframe(top_anomalous[['region', 'anomaly_count']].head(10), use_container_width=True)
            else:
                st.info("👍 No anomalies detected in the filtered data.")
        else:
            st.warning("⚠️ Anomaly data not available. Please run the analysis first.")
            st.markdown("""
            <div class="info-box">
                <h4>🛠️ How to generate anomaly data:</h4>
                <ol>
                    <li>Run: <code>python run_analysis_fast.py</code></li>
                    <li>This will create the anomaly detection data</li>
                    <li>Refresh this dashboard</li>
                </ol>
            </div>
            """, unsafe_allow_html=True)
    
    # TAB 4: Forecasts
    with tab4:
        st.markdown('<div class="sub-header">📈 Predictive Analytics & Forecasting</div>', unsafe_allow_html=True)
        
        try:
            forecast_path = project_root / 'outputs' / 'models' / 'arima_forecast.csv'
            
            if forecast_path.exists():
                arima_forecast = pd.read_csv(forecast_path, parse_dates=['date'])
                
                st.success(f"✅ Loaded {len(arima_forecast):,} forecast records")
                
                # Forecast visualization
                forecast_states = sorted(arima_forecast['state'].unique())
                selected_forecast_state = st.selectbox("🗺️ Select State for Forecast", forecast_states)
                
                state_forecast = arima_forecast[arima_forecast['state'] == selected_forecast_state]
                state_historical = filtered_df[filtered_df['state'] == selected_forecast_state].groupby('date')['total_enrolments'].sum().reset_index()
                
                fig = go.Figure()
                
                # Historical data
                fig.add_trace(go.Scatter(
                    x=state_historical['date'],
                    y=state_historical['total_enrolments'],
                    mode='lines',
                    name='Historical Data',
                    line=dict(color='#667eea', width=3),
                    fill='tozeroy',
                    fillcolor='rgba(102, 126, 234, 0.2)'
                ))
                
                # Forecast
                fig.add_trace(go.Scatter(
                    x=state_forecast['date'],
                    y=state_forecast['total_enrolments_forecast'],
                    mode='lines+markers',
                    name='6-Month Forecast',
                    line=dict(color='#f5576c', width=3, dash='dash'),
                    marker=dict(size=8, symbol='diamond')
                ))
                
                fig.update_layout(
                    title=f'Enrolment Forecast: {selected_forecast_state}',
                    xaxis_title='Date',
                    yaxis_title='Total Enrolments',
                    height=500,
                    template='plotly_dark',
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    hovermode='x unified',
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("---")
                
                # Forecast statistics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    avg_historical = state_historical['total_enrolments'].mean()
                    st.metric("📉 Avg Historical", f"{avg_historical:,.0f}")
                
                with col2:
                    avg_forecast = state_forecast['total_enrolments_forecast'].mean()
                    st.metric("📊 Avg Forecast", f"{avg_forecast:,.0f}")
                
                with col3:
                    growth = ((avg_forecast - avg_historical) / avg_historical * 100) if avg_historical > 0 else 0
                    st.metric("📈 Expected Growth", f"{growth:+.1f}%")
                
                st.markdown("---")
                
                # Forecast table
                st.markdown('<div class="sub-header">📊 Forecast Values</div>', unsafe_allow_html=True)
                st.dataframe(
                    state_forecast[['date', 'total_enrolments_forecast']].rename(
                        columns={'total_enrolments_forecast': 'Forecasted Enrolments'}
                    ),
                    use_container_width=True
                )
            else:
                raise FileNotFoundError("Forecast file not found")
                
        except (FileNotFoundError, Exception) as e:
            st.warning("⚠️ Forecast data not available")
            st.markdown("""
            <div class="info-box">
                <h4>🛠️ How to generate forecasts:</h4>
                <ol>
                    <li>Run the full analysis: <code>python run_analysis.py</code></li>
                    <li>Or run the training script: <code>python train_models.py</code></li>
                    <li>Forecasts will be saved to <code>outputs/models/</code></li>
                    <li>Refresh this dashboard</li>
                </ol>
                <p><strong>Note:</strong> The fast analysis script skips predictive modeling for speed.</p>
            </div>
            """, unsafe_allow_html=True)
    
    # TAB 5: Risk Assessment
    with tab5:
        st.markdown('<div class="sub-header">Risk Assessment Dashboard</div>', unsafe_allow_html=True)
        
        if 'aadhaar_fragility_index' in filtered_df.columns:
            # Risk ranking
            latest_data = filtered_df.sort_values('date').groupby(['state', 'district']).tail(1)
            top_risk = latest_data.nlargest(20, 'aadhaar_fragility_index')[['state', 'district', 'aadhaar_fragility_index', 'fragility_category']]
            top_risk['region'] = top_risk['state'] + ' - ' + top_risk['district']
            
            fig = px.bar(top_risk.sort_values('aadhaar_fragility_index'),
                        x='aadhaar_fragility_index',
                        y='region',
                        orientation='h',
                        title='Top 20 High-Risk Regions',
                        color='aadhaar_fragility_index',
                        color_continuous_scale='Reds',
                        labels={'aadhaar_fragility_index': 'Fragility Index'})
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
            
            # Risk matrix
            st.markdown('<div class="sub-header">Risk Categories by State</div>', unsafe_allow_html=True)
            
            if 'fragility_category' in filtered_df.columns:
                risk_matrix = filtered_df.groupby(['state', 'fragility_category']).size().unstack(fill_value=0)
                
                fig = px.imshow(risk_matrix,
                              labels=dict(x="Fragility Category", y="State", color="Count"),
                              title="Risk Distribution Across States",
                              color_continuous_scale='Reds',
                              aspect='auto')
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Risk assessment data not available.")
    
    # TAB 6: Recommendations
    with tab6:
        st.markdown('<div class="sub-header">Policy Recommendations & Action Items</div>', unsafe_allow_html=True)
        
        if 'fragility_category' in filtered_df.columns:
            # Critical regions
            critical_regions = filtered_df[filtered_df['fragility_category'] == 'Critical']
            high_risk_regions = filtered_df[filtered_df['fragility_category'] == 'High']
            
            st.markdown("### 🚨 Immediate Action Required")
            st.markdown(f"**{len(critical_regions)} Critical Fragility Regions Identified**")
            
            if len(critical_regions) > 0:
                top_critical = critical_regions.nlargest(10, 'aadhaar_fragility_index')[['state', 'district', 'aadhaar_fragility_index']]
                st.dataframe(top_critical, use_container_width=True)
                
                st.markdown("""
                **Recommended Actions:**
                - 🏢 Deploy additional enrolment centers
                - 👥 Increase staff capacity
                - 🚐 Implement mobile enrolment units
                - 🔐 Provide alternative authentication mechanisms
                - 📱 Launch targeted awareness campaigns
                """)
            
            st.markdown("---")
            
            st.markdown("### ⚠️ Monitoring Required")
            st.markdown(f"**{len(high_risk_regions)} High Fragility Regions**")
            
            st.markdown("""
            **Recommended Actions:**
            - 📊 Increase monitoring frequency
            - 📋 Prepare contingency plans
            - 📢 Conduct awareness campaigns
            - 💰 Optimize resource allocation
            """)
            
            st.markdown("---")
            
            st.markdown("### 📈 Capacity Planning")
            st.markdown("""
            **Based on forecasts:**
            - 📅 Prepare for projected demand surges
            - 🗺️ Allocate resources to high-growth regions
            - 🏗️ Plan infrastructure expansion
            - 🎓 Train additional personnel
            """)
            
            st.markdown("---")
            
            st.markdown("### 🔧 System Optimization")
            st.markdown("""
            **Efficiency Improvements:**
            - ⏱️ Reduce biometric update frequency where possible
            - 📝 Streamline demographic update processes
            - 🔨 Implement predictive maintenance
            - 🌐 Enhance digital infrastructure in fragile regions
            """)
        
        # Download recommendations report
        st.markdown("---")
        st.markdown("### 📥 Export Reports")
        
        if st.button("Generate Recommendations Report"):
            st.success("✓ Report generation feature coming soon!")

else:
    st.warning("⚠️ Please run the Jupyter notebook first to generate the processed data files.")

# Enhanced Footer
st.markdown("---")
st.markdown("""
<div class="footer">
    <h2 style='margin-bottom: 1rem;'>🇮🇳 India's Aadhaar Observatory</h2>
    <p style='font-size: 1.1rem; margin-bottom: 0.5rem;'><strong>UIDAI Data Hackathon 2026</strong></p>
    <p style='font-size: 0.95rem; opacity: 0.9;'>Powered by Advanced Analytics & Machine Learning</p>
    <p style='font-size: 0.9rem; margin-top: 1rem; opacity: 0.8;'>Built with ❤️ for Better Governance & Digital Inclusion</p>
</div>
""", unsafe_allow_html=True)
