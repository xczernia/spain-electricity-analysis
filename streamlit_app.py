import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Spain Electricity Market Analysis",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #FF6B35;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-header">⚡ Spain Electricity Market Analysis</div>', unsafe_allow_html=True)
st.markdown("Real-time insights into Spain's electricity generation, pricing, and demand patterns")
st.divider()

# ============================================================================
# GENERATE SAMPLE DATA
# ============================================================================

@st.cache_data
def generate_sample_data():
    """Generate realistic sample data for Spain's electricity market"""
    
    # Date range: last 90 days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)
    date_range = pd.date_range(start=start_date, end=end_date, freq='H')
    
    np.random.seed(42)
    
    # Price data (€/MWh) - more realistic patterns
    base_price = 45
    seasonal_pattern = 10 * np.sin(np.arange(len(date_range)) * 2 * np.pi / (365 * 24))
    daily_pattern = 15 * np.sin(np.arange(len(date_range)) * 2 * np.pi / 24)
    noise = np.random.normal(0, 5, len(date_range))
    prices = base_price + seasonal_pattern + daily_pattern + noise
    prices = np.maximum(prices, 15)  # Minimum price floor
    
    # Demand data (MW)
    base_demand = 28000
    seasonal_demand = 3000 * np.sin(np.arange(len(date_range)) * 2 * np.pi / (365 * 24))
    daily_demand = 5000 * np.sin(np.arange(len(date_range)) * 2 * np.pi / 24 - np.pi/2)
    demand_noise = np.random.normal(0, 800, len(date_range))
    demand = base_demand + seasonal_demand + daily_demand + demand_noise
    demand = np.maximum(demand, 15000)  # Minimum demand
    
    # Generation mix (%)
    solar_base = 15
    wind_base = 25
    nuclear_base = 20
    gas_base = 20
    hydro_base = 15
    other_base = 5
    
    # Add variability
    solar = np.maximum(solar_base + np.random.normal(0, 5, len(date_range)), 0)
    wind = np.maximum(wind_base + np.random.normal(0, 8, len(date_range)), 0)
    nuclear = np.maximum(nuclear_base + np.random.normal(0, 2, len(date_range)), 15)
    gas = np.maximum(gas_base + np.random.normal(0, 6, len(date_range)), 0)
    hydro = np.maximum(hydro_base + np.random.normal(0, 4, len(date_range)), 0)
    other = np.maximum(other_base + np.random.normal(0, 2, len(date_range)), 0)
    
    # Normalize to 100%
    total = solar + wind + nuclear + gas + hydro + other
    solar = (solar / total) * 100
    wind = (wind / total) * 100
    nuclear = (nuclear / total) * 100
    gas = (gas / total) * 100
    hydro = (hydro / total) * 100
    other = (other / total) * 100
    
    # Renewable percentage
    renewable = solar + wind + hydro
    
    df = pd.DataFrame({
        'datetime': date_range,
        'price': prices,
        'demand': demand,
        'solar': solar,
        'wind': wind,
        'nuclear': nuclear,
        'gas': gas,
        'hydro': hydro,
        'other': other,
        'renewable': renewable
    })
    
    return df

df = generate_sample_data()

# ============================================================================
# SIDEBAR FILTERS
# ============================================================================

with st.sidebar:
    st.header("🔧 Filters & Controls")
    
    # Date range selector
    date_range = st.date_input(
        "Select Date Range",
        value=(df['datetime'].min().date(), df['datetime'].max().date()),
        min_value=df['datetime'].min().date(),
        max_value=df['datetime'].max().date(),
        key="date_range"
    )
    
    if len(date_range) == 2:
        start_date, end_date = date_range
        df_filtered = df[(df['datetime'].date >= start_date) & (df['datetime'].date <= end_date)].copy()
    else:
        df_filtered = df.copy()
    
    st.divider()
    st.subheader("Analysis Options")
    
    show_price = st.checkbox("Price Analysis", value=True)
    show_demand = st.checkbox("Demand Patterns", value=True)
    show_generation = st.checkbox("Generation Mix", value=True)
    show_daily = st.checkbox("Daily Curves", value=True)

# ============================================================================
# KEY METRICS
# ============================================================================

st.subheader("📊 Key Metrics")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    avg_price = df_filtered['price'].mean()
    st.metric(
        "Avg Price",
        f"€{avg_price:.2f}/MWh",
        f"{((avg_price / df['price'].mean()) - 1) * 100:.1f}%",
        delta_color="inverse"
    )

with col2:
    avg_demand = df_filtered['demand'].mean()
    st.metric(
        "Avg Demand",
        f"{avg_demand:,.0f} MW",
        f"{((avg_demand / df['demand'].mean()) - 1) * 100:.1f}%"
    )

with col3:
    renewable_pct = df_filtered['renewable'].mean()
    st.metric(
        "Renewable %",
        f"{renewable_pct:.1f}%",
        f"{((renewable_pct / df['renewable'].mean()) - 1) * 100:.1f}%"
    )

with col4:
    max_price = df_filtered['price'].max()
    st.metric(
        "Max Price",
        f"€{max_price:.2f}/MWh",
        f"€{max_price - df_filtered['price'].min():.2f} range"
    )

with col5:
    peak_demand = df_filtered['demand'].max()
    st.metric(
        "Peak Demand",
        f"{peak_demand:,.0f} MW",
        f"{((peak_demand / df_filtered['demand'].mean()) - 1) * 100:.1f}% above avg"
    )

st.divider()

# ============================================================================
# PRICE ANALYSIS
# ============================================================================

if show_price:
    st.subheader("💰 Price Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Price over time
        fig_price = go.Figure()
        fig_price.add_trace(go.Scatter(
            x=df_filtered['datetime'],
            y=df_filtered['price'],
            mode='lines',
            name='Electricity Price',
            line=dict(color='#FF6B35', width=2),
            fill='tozeroy',
            fillcolor='rgba(255, 107, 53, 0.2)'
        ))
        
        fig_price.update_layout(
            title="Electricity Price Over Time",
            xaxis_title="Date",
            yaxis_title="Price (€/MWh)",
            hovermode='x unified',
            template='plotly_white',
            height=400
        )
        st.plotly_chart(fig_price, use_container_width=True)
    
    with col2:
        # Price distribution
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(
            x=df_filtered['price'],
            nbinsx=30,
            name='Price Distribution',
            marker=dict(color='#FF6B35', opacity=0.7),
            showlegend=False
        ))
        
        fig_dist.update_layout(
            title="Price Distribution",
            xaxis_title="Price (€/MWh)",
            yaxis_title="Frequency (Hours)",
            template='plotly_white',
            height=400
        )
        st.plotly_chart(fig_dist, use_container_width=True)
    
    # Price statistics by hour
    st.subheader("Hourly Price Patterns")
    df_filtered['hour'] = df_filtered['datetime'].dt.hour
    hourly_stats = df_filtered.groupby('hour')['price'].agg(['mean', 'min', 'max']).reset_index()
    
    fig_hourly = go.Figure()
    fig_hourly.add_trace(go.Scatter(
        x=hourly_stats['hour'],
        y=hourly_stats['mean'],
        mode='lines+markers',
        name='Average Price',
        line=dict(color='#FF6B35', width=3),
        marker=dict(size=8)
    ))
    fig_hourly.add_trace(go.Scatter(
        x=hourly_stats['hour'],
        y=hourly_stats['max'],
        fill=None,
        mode='lines',
        line_color='rgba(0,0,0,0)',
        showlegend=False
    ))
    fig_hourly.add_trace(go.Scatter(
        x=hourly_stats['hour'],
        y=hourly_stats['min'],
        fill='tonexty',
        mode='lines',
        line_color='rgba(0,0,0,0)',
        name='Min-Max Range',
        fillcolor='rgba(255, 107, 53, 0.2)'
    ))
    
    fig_hourly.update_layout(
        title="Average Hourly Price Pattern (24h cycle)",
        xaxis_title="Hour of Day",
        yaxis_title="Price (€/MWh)",
        hovermode='x unified',
        template='plotly_white',
        height=400
    )
    st.plotly_chart(fig_hourly, use_container_width=True)

# ============================================================================
# DEMAND PATTERNS
# ============================================================================

if show_demand:
    st.subheader("📈 Demand Patterns")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Demand over time
        fig_demand = go.Figure()
        fig_demand.add_trace(go.Scatter(
            x=df_filtered['datetime'],
            y=df_filtered['demand'],
            mode='lines',
            name='Electricity Demand',
            line=dict(color='#4ECDC4', width=2),
            fill='tozeroy',
            fillcolor='rgba(78, 205, 196, 0.2)'
        ))
        
        fig_demand.update_layout(
            title="Electricity Demand Over Time",
            xaxis_title="Date",
            yaxis_title="Demand (MW)",
            hovermode='x unified',
            template='plotly_white',
            height=400
        )
        st.plotly_chart(fig_demand, use_container_width=True)
    
    with col2:
        # Price vs Demand correlation
        fig_corr = px.scatter(
            df_filtered.sample(min(500, len(df_filtered))),
            x='demand',
            y='price',
            trendline='ols',
            title="Price vs Demand Correlation",
            labels={'demand': 'Demand (MW)', 'price': 'Price (€/MWh)'},
            height=400
        )
        fig_corr.update_traces(marker=dict(color='#4ECDC4', size=6, opacity=0.6))
        st.plotly_chart(fig_corr, use_container_width=True)
    
    # Hourly demand pattern
    st.subheader("Hourly Demand Pattern")
    hourly_demand = df_filtered.groupby('hour')['demand'].agg(['mean', 'min', 'max']).reset_index()
    
    fig_hourly_dem = go.Figure()
    fig_hourly_dem.add_trace(go.Scatter(
        x=hourly_demand['hour'],
        y=hourly_demand['mean'],
        mode='lines+markers',
        name='Average Demand',
        line=dict(color='#4ECDC4', width=3),
        marker=dict(size=8)
    ))
    fig_hourly_dem.add_trace(go.Scatter(
        x=hourly_demand['hour'],
        y=hourly_demand['max'],
        fill=None,
        mode='lines',
        line_color='rgba(0,0,0,0)',
        showlegend=False
    ))
    fig_hourly_dem.add_trace(go.Scatter(
        x=hourly_demand['hour'],
        y=hourly_demand['min'],
        fill='tonexty',
        mode='lines',
        line_color='rgba(0,0,0,0)',
        name='Min-Max Range',
        fillcolor='rgba(78, 205, 196, 0.2)'
    ))
    
    fig_hourly_dem.update_layout(
        title="Average Hourly Demand Pattern (24h cycle)",
        xaxis_title="Hour of Day",
        yaxis_title="Demand (MW)",
        hovermode='x unified',
        template='plotly_white',
        height=400
    )
    st.plotly_chart(fig_hourly_dem, use_container_width=True)

# ============================================================================
# GENERATION MIX
# ============================================================================

if show_generation:
    st.subheader("⚡ Generation Mix Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Current generation mix (average)
        current_mix = df_filtered[['solar', 'wind', 'nuclear', 'gas', 'hydro', 'other']].mean()
        
        colors = ['#FFD166', '#118AB2', '#073B4C', '#EF476F', '#06A77D', '#8E7DBE']
        fig_pie = go.Figure(data=[go.Pie(
            labels=current_mix.index.str.capitalize(),
            values=current_mix.values,
            hole=0.4,
            marker=dict(colors=colors),
            textposition='inside',
            textinfo='label+percent'
        )])
        
        fig_pie.update_layout(
            title="Average Generation Mix",
            height=400
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Renewable vs Non-renewable
        renewable_data = df_filtered.groupby(df_filtered['datetime'].dt.date)[['solar', 'wind', 'hydro']].mean().sum(axis=1)
        non_renewable_data = df_filtered.groupby(df_filtered['datetime'].dt.date)[['nuclear', 'gas', 'other']].mean().sum(axis=1)
        
        fig_stack = go.Figure()
        fig_stack.add_trace(go.Bar(
            x=renewable_data.index,
            y=renewable_data.values,
            name='Renewable',
            marker=dict(color='#06A77D')
        ))
        fig_stack.add_trace(go.Bar(
            x=non_renewable_data.index,
            y=non_renewable_data.values,
            name='Non-Renewable',
            marker=dict(color='#EF476F')
        ))
        
        fig_stack.update_layout(
            title="Renewable vs Non-Renewable Generation",
            xaxis_title="Date",
            yaxis_title="Generation (%)",
            barmode='stack',
            hovermode='x unified',
            template='plotly_white',
            height=400
        )
        st.plotly_chart(fig_stack, use_container_width=True)
    
    # Generation sources over time
    st.subheader("Generation Sources Over Time")
    
    fig_sources = go.Figure()
    
    sources = ['solar', 'wind', 'nuclear', 'gas', 'hydro', 'other']
    colors_dict = {
        'solar': '#FFD166',
        'wind': '#118AB2',
        'nuclear': '#073B4C',
        'gas': '#EF476F',
        'hydro': '#06A77D',
        'other': '#8E7DBE'
    }
    
    for source in sources:
        fig_sources.add_trace(go.Scatter(
            x=df_filtered['datetime'],
            y=df_filtered[source],
            mode='lines',
            name=source.capitalize(),
            line=dict(width=2, color=colors_dict[source]),
            stackgroup='one'
        ))
    
    fig_sources.update_layout(
        title="Stacked Generation Mix Over Time",
        xaxis_title="Date",
        yaxis_title="Generation (%)",
        hovermode='x unified',
        template='plotly_white',
        height=450
    )
    st.plotly_chart(fig_sources, use_container_width=True)

# ============================================================================
# DAILY CURVES VISUALIZATION
# ============================================================================

if show_daily:
    st.subheader("📅 Daily Curves Visualization")
    
    # Select a date for detailed daily curve
    selected_date = st.date_input(
        "Select a date to view detailed daily curve",
        value=df_filtered['datetime'].max().date(),
        min_value=df_filtered['datetime'].min().date(),
        max_value=df_filtered['datetime'].max().date()
    )
    
    daily_data = df_filtered[df_filtered['datetime'].date == selected_date].copy()
    
    if len(daily_data) > 0:
        col1, col2 = st.columns(2)
        
        with col1:
            # Daily price curve
            fig_daily_price = go.Figure()
            fig_daily_price.add_trace(go.Scatter(
                x=daily_data['datetime'].dt.strftime('%H:%M'),
                y=daily_data['price'],
                mode='lines+markers',
                name='Price',
                line=dict(color='#FF6B35', width=3),
                marker=dict(size=8),
                fill='tozeroy',
                fillcolor='rgba(255, 107, 53, 0.2)'
            ))
            
            fig_daily_price.update_layout(
                title=f"Daily Price Curve - {selected_date}",
                xaxis_title="Time",
                yaxis_title="Price (€/MWh)",
                hovermode='x unified',
                template='plotly_white',
                height=400
            )
            st.plotly_chart(fig_daily_price, use_container_width=True)
        
        with col2:
            # Daily demand curve
            fig_daily_demand = go.Figure()
            fig_daily_demand.add_trace(go.Scatter(
                x=daily_data['datetime'].dt.strftime('%H:%M'),
                y=daily_data['demand'],
                mode='lines+markers',
                name='Demand',
                line=dict(color='#4ECDC4', width=3),
                marker=dict(size=8),
                fill='tozeroy',
                fillcolor='rgba(78, 205, 196, 0.2)'
            ))
            
            fig_daily_demand.update_layout(
                title=f"Daily Demand Curve - {selected_date}",
                xaxis_title="Time",
                yaxis_title="Demand (MW)",
                hovermode='x unified',
                template='plotly_white',
                height=400
            )
            st.plotly_chart(fig_daily_demand, use_container_width=True)
        
        # Daily generation mix
        st.subheader(f"Generation Mix Breakdown - {selected_date}")
        
        fig_daily_gen = go.Figure()
        
        for source in sources:
            fig_daily_gen.add_trace(go.Scatter(
                x=daily_data['datetime'].dt.strftime('%H:%M'),
                y=daily_data[source],
                mode='lines',
                name=source.capitalize(),
                line=dict(width=2, color=colors_dict[source]),
                stackgroup='one'
            ))
        
        fig_daily_gen.update_layout(
            title=f"Stacked Generation Mix - {selected_date}",
            xaxis_title="Time",
            yaxis_title="Generation (%)",
            hovermode='x unified',
            template='plotly_white',
            height=400
        )
        st.plotly_chart(fig_daily_gen, use_container_width=True)
        
        # Detailed data table for the day
        st.subheader(f"Hourly Data - {selected_date}")
        
        display_data = daily_data[[
            'datetime', 'price', 'demand', 'solar', 'wind', 'nuclear', 'gas', 'hydro', 'renewable'
        ]].copy()
        
        display_data['datetime'] = display_data['datetime'].dt.strftime('%H:%M')
        display_data = display_data.rename(columns={
            'datetime': 'Time',
            'price': 'Price (€/MWh)',
            'demand': 'Demand (MW)',
            'solar': 'Solar (%)',
            'wind': 'Wind (%)',
            'nuclear': 'Nuclear (%)',
            'gas': 'Gas (%)',
            'hydro': 'Hydro (%)',
            'renewable': 'Renewable (%)'
        })
        
        st.dataframe(
            display_data.style.format({
                'Price (€/MWh)': '{:.2f}',
                'Demand (MW)': '{:,.0f}',
                'Solar (%)': '{:.1f}',
                'Wind (%)': '{:.1f}',
                'Nuclear (%)': '{:.1f}',
                'Gas (%)': '{:.1f}',
                'Hydro (%)': '{:.1f}',
                'Renewable (%)': '{:.1f}'
            }),
            use_container_width=True,
            height=400
        )
    else:
        st.warning("No data available for the selected date")

# ============================================================================
# FOOTER
# ============================================================================

st.divider()
st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
    <p>📊 Spain Electricity Market Analysis Dashboard</p>
    <p style='font-size: 0.85em;'>Data generated from 90-day historical patterns | Last updated: {}</p>
    </div>
""".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')), unsafe_allow_html=True)
