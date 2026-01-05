import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Spain Electricity Market Analysis",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("⚡ Spain Electricity Market Analysis")
st.markdown("""
    This dashboard provides comprehensive analysis of the Spanish electricity market,
    including price trends, demand patterns, and generation sources.
""")

# Sidebar for navigation and filters
st.sidebar.header("Navigation & Filters")
page = st.sidebar.radio(
    "Select a page",
    ["Overview", "Price Analysis", "Demand Patterns", "Generation Mix", "About"]
)

# Date range selector
date_range = st.sidebar.date_input(
    "Select date range",
    value=(datetime.now() - timedelta(days=30), datetime.now()),
    max_value=datetime.now()
)

# Sample data generator (replace with real data source)
@st.cache_data
def load_sample_data(days=30):
    """Generate sample electricity market data"""
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    data = {
        'date': dates,
        'price': np.random.uniform(50, 150, days) + np.sin(np.arange(days) * 2 * np.pi / 7) * 20,
        'demand': np.random.uniform(20000, 35000, days),
        'wind': np.random.uniform(5000, 15000, days),
        'solar': np.random.uniform(2000, 10000, days),
        'hydro': np.random.uniform(3000, 8000, days),
        'nuclear': np.random.uniform(7000, 9000, days),
        'gas': np.random.uniform(5000, 12000, days),
    }
    return pd.DataFrame(data)

# Load data
df = load_sample_data(30)

# PAGE: Overview
if page == "Overview":
    st.header("Market Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_price = df['price'].mean()
        st.metric("Avg Price (€/MWh)", f"{avg_price:.2f}")
    
    with col2:
        avg_demand = df['demand'].mean()
        st.metric("Avg Demand (MWh)", f"{avg_demand:,.0f}")
    
    with col3:
        renewable_pct = ((df['wind'].mean() + df['solar'].mean() + df['hydro'].mean()) / df['demand'].mean()) * 100
        st.metric("Renewable %", f"{renewable_pct:.1f}%")
    
    with col4:
        latest_price = df['price'].iloc[-1]
        price_change = ((latest_price - df['price'].iloc[-7]) / df['price'].iloc[-7]) * 100
        st.metric("Latest Price (€/MWh)", f"{latest_price:.2f}", delta=f"{price_change:.1f}%")
    
    st.divider()
    
    # Price trend chart
    st.subheader("Price Trend (Last 30 Days)")
    fig_price = go.Figure()
    fig_price.add_trace(go.Scatter(
        x=df['date'],
        y=df['price'],
        mode='lines+markers',
        name='Price',
        line=dict(color='#FF6692', width=2)
    ))
    fig_price.update_layout(
        title="Electricity Price Trend",
        xaxis_title="Date",
        yaxis_title="Price (€/MWh)",
        hovermode='x unified',
        height=400
    )
    st.plotly_chart(fig_price, use_container_width=True)

# PAGE: Price Analysis
elif page == "Price Analysis":
    st.header("Price Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Price Distribution")
        fig_dist = px.histogram(df, x='price', nbins=20, title="Price Distribution")
        fig_dist.update_layout(height=400)
        st.plotly_chart(fig_dist, use_container_width=True)
    
    with col2:
        st.subheader("Price Statistics")
        stats = df['price'].describe()
        st.dataframe(stats, use_container_width=True)
    
    st.divider()
    st.subheader("Detailed Price Data")
    st.dataframe(df[['date', 'price', 'demand']], use_container_width=True)

# PAGE: Demand Patterns
elif page == "Demand Patterns":
    st.header("Demand Patterns")
    
    fig_demand = go.Figure()
    fig_demand.add_trace(go.Scatter(
        x=df['date'],
        y=df['demand'],
        mode='lines+markers',
        name='Demand',
        line=dict(color='#1f77b4', width=2),
        fill='tozeroy'
    ))
    fig_demand.update_layout(
        title="Electricity Demand Trend",
        xaxis_title="Date",
        yaxis_title="Demand (MWh)",
        hovermode='x unified',
        height=450
    )
    st.plotly_chart(fig_demand, use_container_width=True)
    
    st.divider()
    st.subheader("Demand vs Price Correlation")
    fig_corr = px.scatter(df, x='demand', y='price', trendline='ols',
                          title="Price vs Demand Relationship")
    fig_corr.update_layout(height=400)
    st.plotly_chart(fig_corr, use_container_width=True)

# PAGE: Generation Mix
elif page == "Generation Mix":
    st.header("Generation Sources Mix")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Average Generation Mix (Last 30 Days)")
        sources = ['wind', 'solar', 'hydro', 'nuclear', 'gas']
        avg_gen = [df[source].mean() for source in sources]
        fig_pie = go.Figure(data=[go.Pie(
            labels=['Wind', 'Solar', 'Hydro', 'Nuclear', 'Gas'],
            values=avg_gen,
            marker=dict(colors=['#74B9FF', '#FFD93D', '#6BCB77', '#FF6B6B', '#C0C0C0'])
        )])
        fig_pie.update_layout(height=400)
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        st.subheader("Generation Source Trends")
        fig_gen = go.Figure()
        for source, label in [('wind', 'Wind'), ('solar', 'Solar'), 
                              ('hydro', 'Hydro'), ('nuclear', 'Nuclear'), ('gas', 'Gas')]:
            fig_gen.add_trace(go.Scatter(
                x=df['date'],
                y=df[source],
                mode='lines',
                name=label
            ))
        fig_gen.update_layout(
            title="Generation Sources Over Time",
            xaxis_title="Date",
            yaxis_title="Generation (MWh)",
            hovermode='x unified',
            height=400
        )
        st.plotly_chart(fig_gen, use_container_width=True)

# PAGE: About
elif page == "About":
    st.header("About This Dashboard")
    
    st.markdown("""
    ### Overview
    This Streamlit application provides analysis and visualization of the Spanish electricity market.
    
    ### Data Sources
    - Real-time market data from Spain's electricity operator
    - Historical price and demand information
    - Generation mix by source
    
    ### Features
    - **Overview**: Key metrics and price trends
    - **Price Analysis**: Detailed price statistics and distribution
    - **Demand Patterns**: Demand trends and correlation analysis
    - **Generation Mix**: Energy source composition and trends
    
    ### Technologies Used
    - **Streamlit**: Web application framework
    - **Pandas**: Data manipulation and analysis
    - **Plotly**: Interactive visualizations
    - **NumPy**: Numerical computations
    
    ### Last Updated
    """)
    st.info(f"Last refresh: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    
    st.markdown("""
    ### Contact & Support
    For questions or feedback, please visit the repository.
    """)

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: gray; font-size: 12px;'>
    Spain Electricity Market Analysis © 2026
</div>
""", unsafe_allow_html=True)
