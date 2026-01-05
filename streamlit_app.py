import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np

# Set page configuration
st.set_page_config(
    page_title="Spain Electricity Market Analysis",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("⚡ Spain Electricity Market Analysis")
st.markdown("""
    This dashboard provides insights into Spain's electricity market, including:
    - Real-time electricity prices
    - Supply and demand analysis
    - Renewable energy penetration
    - Historical trends
""")

# Sidebar configuration
st.sidebar.header("Dashboard Configuration")
analysis_type = st.sidebar.selectbox(
    "Select Analysis Type",
    ["Overview", "Price Analysis", "Supply & Demand", "Renewable Energy", "Historical Trends"]
)

date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(datetime.now() - timedelta(days=30), datetime.now()),
    max_value=datetime.now()
)

# Main content area
if analysis_type == "Overview":
    st.header("Market Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Current Price",
            value="€85.50/MWh",
            delta="+2.5%",
            delta_color="inverse"
        )
    
    with col2:
        st.metric(
            label="Total Demand",
            value="28,500 MW",
            delta="-1.2%"
        )
    
    with col3:
        st.metric(
            label="Renewable Generation",
            value="45.2%",
            delta="+3.1%"
        )
    
    with col4:
        st.metric(
            label="Wind Power",
            value="8,920 MW",
            delta="+5.8%",
            delta_color="inverse"
        )
    
    st.markdown("---")
    
    # Sample data for visualization
    hours = pd.date_range(start=datetime.now() - timedelta(hours=24), periods=24, freq='h')
    sample_data = pd.DataFrame({
        'Time': hours,
        'Price': np.random.uniform(70, 120, 24),
        'Demand': np.random.uniform(25000, 30000, 24),
        'Wind': np.random.uniform(6000, 10000, 24),
        'Solar': np.random.uniform(2000, 8000, 24)
    })
    
    # Price chart
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Hourly Electricity Price (24h)")
        fig_price = px.line(
            sample_data,
            x='Time',
            y='Price',
            title='Electricity Price Trend',
            labels={'Price': 'Price (€/MWh)', 'Time': 'Time'}
        )
        fig_price.update_traces(line=dict(color='#FF6B6B', width=2))
        st.plotly_chart(fig_price, use_container_width=True)
    
    with col2:
        st.subheader("Demand & Generation (24h)")
        fig_demand = px.line(
            sample_data,
            x='Time',
            y=['Demand', 'Wind', 'Solar'],
            title='Demand vs Renewable Generation',
            labels={'value': 'Power (MW)', 'Time': 'Time', 'variable': 'Source'}
        )
        st.plotly_chart(fig_demand, use_container_width=True)

elif analysis_type == "Price Analysis":
    st.header("Price Analysis")
    
    st.subheader("Price Statistics")
    price_data = np.random.uniform(65, 130, 100)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Average Price", f"€{price_data.mean():.2f}/MWh")
    with col2:
        st.metric("Max Price", f"€{price_data.max():.2f}/MWh")
    with col3:
        st.metric("Min Price", f"€{price_data.min():.2f}/MWh")
    with col4:
        st.metric("Std Deviation", f"€{price_data.std():.2f}")
    
    st.markdown("---")
    
    # Price distribution
    fig_dist = px.histogram(
        x=price_data,
        nbins=20,
        title='Price Distribution',
        labels={'x': 'Price (€/MWh)', 'count': 'Frequency'}
    )
    st.plotly_chart(fig_dist, use_container_width=True)

elif analysis_type == "Supply & Demand":
    st.header("Supply & Demand Analysis")
    
    # Create sample data
    hours = pd.date_range(start=datetime.now() - timedelta(days=7), periods=168, freq='h')
    supply_demand_data = pd.DataFrame({
        'Time': hours,
        'Demand': np.random.uniform(22000, 32000, 168),
        'Nuclear': np.random.uniform(8000, 10000, 168),
        'Hydro': np.random.uniform(3000, 6000, 168),
        'Wind': np.random.uniform(5000, 12000, 168),
        'Solar': np.random.uniform(1000, 8000, 168),
        'Gas': np.random.uniform(2000, 8000, 168),
        'Coal': np.random.uniform(1000, 4000, 168)
    })
    
    fig_supply = px.area(
        supply_demand_data,
        x='Time',
        y=['Nuclear', 'Hydro', 'Wind', 'Solar', 'Gas', 'Coal'],
        title='Energy Supply Mix (7 days)',
        labels={'value': 'Power (MW)', 'Time': 'Time', 'variable': 'Source'}
    )
    st.plotly_chart(fig_supply, use_container_width=True)
    
    st.markdown("---")
    
    # Balance table
    latest_balance = supply_demand_data.iloc[-1]
    st.subheader("Current Energy Balance")
    
    balance_df = pd.DataFrame({
        'Source': ['Nuclear', 'Hydro', 'Wind', 'Solar', 'Gas', 'Coal', 'Total Supply', 'Demand', 'Balance'],
        'Power (MW)': [
            f"{latest_balance['Nuclear']:.0f}",
            f"{latest_balance['Hydro']:.0f}",
            f"{latest_balance['Wind']:.0f}",
            f"{latest_balance['Solar']:.0f}",
            f"{latest_balance['Gas']:.0f}",
            f"{latest_balance['Coal']:.0f}",
            f"{sum([latest_balance['Nuclear'], latest_balance['Hydro'], latest_balance['Wind'], latest_balance['Solar'], latest_balance['Gas'], latest_balance['Coal']]):.0f}",
            f"{latest_balance['Demand']:.0f}",
            f"{sum([latest_balance['Nuclear'], latest_balance['Hydro'], latest_balance['Wind'], latest_balance['Solar'], latest_balance['Gas'], latest_balance['Coal']]) - latest_balance['Demand']:.0f}"
        ]
    })
    st.table(balance_df)

elif analysis_type == "Renewable Energy":
    st.header("Renewable Energy Analysis")
    
    # Sample renewable data
    renewable_data = pd.DataFrame({
        'Source': ['Wind', 'Solar', 'Hydro', 'Biomass'],
        'Power (MW)': [8920, 5450, 4200, 820],
        'Percentage': [45.2, 27.6, 21.3, 4.1]
    })
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_pie = px.pie(
            renewable_data,
            values='Power (MW)',
            names='Source',
            title='Renewable Energy Sources Distribution'
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        fig_bar = px.bar(
            renewable_data,
            x='Source',
            y='Power (MW)',
            title='Renewable Power Generation by Source',
            labels={'Power (MW)': 'Power (MW)', 'Source': 'Energy Source'}
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    
    st.markdown("---")
    st.subheader("Renewable Energy Targets")
    
    targets = pd.DataFrame({
        'Year': [2023, 2024, 2025, 2026, 2030],
        'Target (%)': [45, 50, 55, 58, 70]
    })
    
    fig_targets = px.line(
        targets,
        x='Year',
        y='Target (%)',
        title='Spain Renewable Energy Target Trajectory',
        markers=True,
        labels={'Target (%)': 'Renewable Energy (%)', 'Year': 'Year'}
    )
    st.plotly_chart(fig_targets, use_container_width=True)

elif analysis_type == "Historical Trends":
    st.header("Historical Trends")
    
    # Create sample historical data
    dates = pd.date_range(start=datetime.now() - timedelta(days=365), periods=365, freq='d')
    historical_data = pd.DataFrame({
        'Date': dates,
        'Avg_Price': np.cumsum(np.random.normal(0, 1, 365)) + 85,
        'Avg_Demand': np.cumsum(np.random.normal(0, 10, 365)) + 28500,
        'Renewable_Percentage': np.cumsum(np.random.normal(0, 0.1, 365)) + 40
    })
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_price_trend = px.line(
            historical_data,
            x='Date',
            y='Avg_Price',
            title='Historical Average Price Trend (1 year)',
            labels={'Avg_Price': 'Price (€/MWh)', 'Date': 'Date'}
        )
        st.plotly_chart(fig_price_trend, use_container_width=True)
    
    with col2:
        fig_renewable_trend = px.line(
            historical_data,
            x='Date',
            y='Renewable_Percentage',
            title='Renewable Energy Penetration Trend (1 year)',
            labels={'Renewable_Percentage': 'Renewable (%)', 'Date': 'Date'}
        )
        st.plotly_chart(fig_renewable_trend, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: gray; font-size: 0.8rem;'>
        Last updated: {} UTC | Data source: Spain Electricity Market Analysis
    </div>
""".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")), unsafe_allow_html=True)
