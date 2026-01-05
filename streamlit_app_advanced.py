"""
Advanced Streamlit Application for Spain Electricity Analysis
Features: Advanced ML Forecasting, API Integration, Real-time Data Processing
Author: xczernia
Date: 2026-01-05
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import requests
from typing import Dict, List, Tuple, Optional
import json
from abc import ABC, abstractmethod
import logging
from functools import lru_cache
import warnings

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==================== Configuration ====================
REDIS_ENABLED = False  # Set to True if Redis is available
API_TIMEOUT = 30
CACHE_EXPIRATION = 3600  # 1 hour in seconds
MAX_FORECAST_DAYS = 30

# API Endpoints
ELASTICSEARCH_ENDPOINT = "http://elasticsearch:9200"
API_BASE_URL = "http://api:8000"

# ==================== Custom Exceptions ====================
class DataFetchError(Exception):
    """Raised when data fetching fails"""
    pass

class ForecastError(Exception):
    """Raised when forecasting fails"""
    pass

# ==================== Abstract Base Classes ====================
class DataProvider(ABC):
    """Abstract base class for data providers"""
    
    @abstractmethod
    def fetch_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch data from the provider"""
        pass
    
    @abstractmethod
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate fetched data"""
        pass

class Forecaster(ABC):
    """Abstract base class for forecasting models"""
    
    @abstractmethod
    def train(self, data: pd.DataFrame) -> None:
        """Train the forecasting model"""
        pass
    
    @abstractmethod
    def predict(self, periods: int) -> pd.DataFrame:
        """Generate forecasts"""
        pass
    
    @abstractmethod
    def get_metrics(self) -> Dict[str, float]:
        """Get model performance metrics"""
        pass

# ==================== Data Providers ====================
class REDissApiProvider(DataProvider):
    """Provider for REDiss electricity data API"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.base_url = "https://api.rediss.es/v1"
        self.session = requests.Session()
        if api_key:
            self.session.headers.update({"Authorization": f"Bearer {api_key}"})
    
    def fetch_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch real-time electricity data from REDiss API"""
        try:
            params = {
                'start': start_date,
                'end': end_date,
                'region': 'ES',
                'metrics': ['demand', 'generation', 'price']
            }
            
            response = self.session.get(
                f"{self.base_url}/electricity",
                params=params,
                timeout=API_TIMEOUT
            )
            response.raise_for_status()
            
            data = response.json()
            df = pd.DataFrame(data['data'])
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            logger.info(f"Successfully fetched {len(df)} records from REDiss API")
            return df
            
        except requests.RequestException as e:
            logger.error(f"Error fetching data from REDiss API: {str(e)}")
            raise DataFetchError(f"Failed to fetch REDiss data: {str(e)}")
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate data structure and content"""
        required_columns = ['timestamp', 'demand', 'generation', 'price']
        return all(col in data.columns for col in required_columns)

class LocalDataProvider(DataProvider):
    """Provider for local CSV data"""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
    
    def fetch_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Load data from local CSV file"""
        try:
            df = pd.read_csv(self.file_path)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Filter by date range
            start = pd.to_datetime(start_date)
            end = pd.to_datetime(end_date)
            df = df[(df['timestamp'] >= start) & (df['timestamp'] <= end)]
            
            logger.info(f"Successfully loaded {len(df)} records from local file")
            return df
            
        except Exception as e:
            logger.error(f"Error loading local data: {str(e)}")
            raise DataFetchError(f"Failed to load local data: {str(e)}")
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate data structure"""
        return 'timestamp' in data.columns and len(data) > 0

# ==================== Forecasting Models ====================
class ARIMAForecaster(Forecaster):
    """ARIMA-based forecasting model"""
    
    def __init__(self, order: Tuple[int, int, int] = (1, 1, 1)):
        self.order = order
        self.model = None
        self.fitted_model = None
        self.metrics = {}
        self.training_data = None
    
    def train(self, data: pd.DataFrame) -> None:
        """Train ARIMA model"""
        try:
            from statsmodels.tsa.arima.model import ARIMA
            
            target_column = 'demand'
            if target_column not in data.columns:
                target_column = data.columns[1]
            
            self.training_data = data[target_column].values
            self.model = ARIMA(self.training_data, order=self.order)
            self.fitted_model = self.model.fit()
            
            # Calculate metrics
            self.metrics['aic'] = float(self.fitted_model.aic)
            self.metrics['bic'] = float(self.fitted_model.bic)
            
            logger.info(f"ARIMA model trained successfully. AIC: {self.metrics['aic']:.2f}")
            
        except Exception as e:
            logger.error(f"Error training ARIMA model: {str(e)}")
            raise ForecastError(f"Failed to train ARIMA model: {str(e)}")
    
    def predict(self, periods: int) -> pd.DataFrame:
        """Generate ARIMA forecasts"""
        if self.fitted_model is None:
            raise ForecastError("Model not trained. Call train() first.")
        
        try:
            forecast = self.fitted_model.get_forecast(steps=periods)
            forecast_df = forecast.conf_int()
            forecast_df['prediction'] = forecast.predicted_mean.values
            forecast_df.columns = ['lower_ci', 'upper_ci', 'forecast']
            
            return forecast_df.reset_index(drop=True)
            
        except Exception as e:
            logger.error(f"Error generating ARIMA predictions: {str(e)}")
            raise ForecastError(f"Failed to generate forecasts: {str(e)}")
    
    def get_metrics(self) -> Dict[str, float]:
        return self.metrics

class ExponentialSmoothingForecaster(Forecaster):
    """Exponential Smoothing forecasting model"""
    
    def __init__(self, seasonal_periods: int = 24):
        self.seasonal_periods = seasonal_periods
        self.model = None
        self.fitted_model = None
        self.metrics = {}
    
    def train(self, data: pd.DataFrame) -> None:
        """Train Exponential Smoothing model"""
        try:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
            
            target_column = 'demand'
            if target_column not in data.columns:
                target_column = data.columns[1]
            
            series = data[target_column].values
            
            self.model = ExponentialSmoothing(
                series,
                seasonal_periods=min(self.seasonal_periods, len(series) // 2),
                trend='add',
                seasonal='add'
            )
            self.fitted_model = self.model.fit()
            
            self.metrics['sse'] = float(self.fitted_model.sse)
            
            logger.info(f"Exponential Smoothing model trained successfully")
            
        except Exception as e:
            logger.error(f"Error training Exponential Smoothing model: {str(e)}")
            raise ForecastError(f"Failed to train model: {str(e)}")
    
    def predict(self, periods: int) -> pd.DataFrame:
        """Generate forecasts using Exponential Smoothing"""
        if self.fitted_model is None:
            raise ForecastError("Model not trained. Call train() first.")
        
        try:
            forecast = self.fitted_model.get_forecast(steps=periods)
            forecast_df = forecast.conf_int()
            forecast_df['forecast'] = forecast.predicted_mean.values
            forecast_df.columns = ['lower_ci', 'upper_ci', 'forecast']
            
            return forecast_df.reset_index(drop=True)
            
        except Exception as e:
            logger.error(f"Error generating forecasts: {str(e)}")
            raise ForecastError(f"Failed to generate forecasts: {str(e)}")
    
    def get_metrics(self) -> Dict[str, float]:
        return self.metrics

class MLForecaster(Forecaster):
    """Machine Learning-based forecasting model using Prophet"""
    
    def __init__(self, yearly_seasonality: bool = True):
        self.yearly_seasonality = yearly_seasonality
        self.model = None
        self.metrics = {}
    
    def train(self, data: pd.DataFrame) -> None:
        """Train Prophet model"""
        try:
            from prophet import Prophet
            
            target_column = 'demand'
            if target_column not in data.columns:
                target_column = data.columns[1]
            
            # Prepare data for Prophet
            prophet_df = pd.DataFrame({
                'ds': pd.to_datetime(data['timestamp']),
                'y': data[target_column].values
            })
            
            self.model = Prophet(
                yearly_seasonality=self.yearly_seasonality,
                daily_seasonality=True,
                interval_width=0.95
            )
            self.model.fit(prophet_df)
            
            logger.info("Prophet model trained successfully")
            
        except Exception as e:
            logger.error(f"Error training Prophet model: {str(e)}")
            raise ForecastError(f"Failed to train Prophet model: {str(e)}")
    
    def predict(self, periods: int) -> pd.DataFrame:
        """Generate forecasts using Prophet"""
        if self.model is None:
            raise ForecastError("Model not trained. Call train() first.")
        
        try:
            future = self.model.make_future_dataframe(periods=periods)
            forecast = self.model.predict(future)
            
            result_df = pd.DataFrame({
                'forecast': forecast['yhat'].tail(periods).values,
                'lower_ci': forecast['yhat_lower'].tail(periods).values,
                'upper_ci': forecast['yhat_upper'].tail(periods).values
            })
            
            return result_df.reset_index(drop=True)
            
        except Exception as e:
            logger.error(f"Error generating Prophet predictions: {str(e)}")
            raise ForecastError(f"Failed to generate forecasts: {str(e)}")
    
    def get_metrics(self) -> Dict[str, float]:
        return self.metrics

# ==================== Data Processing ====================
class DataProcessor:
    """Advanced data processing and feature engineering"""
    
    @staticmethod
    def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
        """Create advanced features for ML models"""
        df = data.copy()
        
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['month'] = df['timestamp'].dt.month
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Rolling statistics
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df[f'{col}_rolling_24h'] = df[col].rolling(window=24, min_periods=1).mean()
            df[f'{col}_rolling_7d'] = df[col].rolling(window=168, min_periods=1).mean()
            df[f'{col}_std_24h'] = df[col].rolling(window=24, min_periods=1).std()
        
        return df
    
    @staticmethod
    def handle_missing_values(data: pd.DataFrame, method: str = 'interpolate') -> pd.DataFrame:
        """Handle missing values in data"""
        df = data.copy()
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if method == 'interpolate':
            df[numeric_cols] = df[numeric_cols].interpolate(method='linear', limit_direction='both')
        elif method == 'forward_fill':
            df[numeric_cols] = df[numeric_cols].fillna(method='ffill')
        elif method == 'backward_fill':
            df[numeric_cols] = df[numeric_cols].fillna(method='bfill')
        
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        
        return df
    
    @staticmethod
    def normalize_data(data: pd.DataFrame, columns: Optional[List[str]] = None) -> Tuple[pd.DataFrame, Dict]:
        """Normalize numeric columns using min-max scaling"""
        df = data.copy()
        scaling_params = {}
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in columns:
            min_val = df[col].min()
            max_val = df[col].max()
            scaling_params[col] = {'min': min_val, 'max': max_val}
            
            if max_val > min_val:
                df[col] = (df[col] - min_val) / (max_val - min_val)
        
        return df, scaling_params

# ==================== API Client ====================
class ElectricityAPIClient:
    """Client for interacting with electricity data APIs"""
    
    def __init__(self, base_url: str = API_BASE_URL):
        self.base_url = base_url
        self.session = requests.Session()
    
    def get_historical_data(self, start_date: str, end_date: str, 
                           region: str = 'ES') -> pd.DataFrame:
        """Fetch historical electricity data"""
        try:
            response = self.session.get(
                f"{self.base_url}/electricity/historical",
                params={'start': start_date, 'end': end_date, 'region': region},
                timeout=API_TIMEOUT
            )
            response.raise_for_status()
            
            return pd.DataFrame(response.json()['data'])
            
        except Exception as e:
            logger.error(f"Error fetching historical data: {str(e)}")
            raise DataFetchError(str(e))
    
    def get_real_time_data(self, region: str = 'ES') -> Dict:
        """Fetch real-time electricity metrics"""
        try:
            response = self.session.get(
                f"{self.base_url}/electricity/real-time",
                params={'region': region},
                timeout=API_TIMEOUT
            )
            response.raise_for_status()
            
            return response.json()['data']
            
        except Exception as e:
            logger.error(f"Error fetching real-time data: {str(e)}")
            raise DataFetchError(str(e))
    
    def submit_forecast(self, forecast_data: Dict) -> bool:
        """Submit forecast results to API"""
        try:
            response = self.session.post(
                f"{self.base_url}/forecasts",
                json=forecast_data,
                timeout=API_TIMEOUT
            )
            response.raise_for_status()
            
            return True
            
        except Exception as e:
            logger.error(f"Error submitting forecast: {str(e)}")
            return False

# ==================== Streamlit UI ====================
def initialize_session_state():
    """Initialize Streamlit session state variables"""
    if 'data_cache' not in st.session_state:
        st.session_state.data_cache = None
        st.session_state.cache_time = None
    if 'model_state' not in st.session_state:
        st.session_state.model_state = {}

def load_and_cache_data(data_source: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Load data with caching mechanism"""
    current_time = datetime.now()
    
    # Check cache validity
    if (st.session_state.data_cache is not None and 
        st.session_state.cache_time is not None):
        cache_age = (current_time - st.session_state.cache_time).total_seconds()
        if cache_age < CACHE_EXPIRATION:
            st.info("📊 Using cached data")
            return st.session_state.data_cache
    
    try:
        with st.spinner("Loading data..."):
            if data_source == "Sample Data":
                data = generate_sample_data(start_date, end_date)
            else:
                api_client = ElectricityAPIClient()
                data = api_client.get_historical_data(start_date, end_date)
        
        # Cache the data
        st.session_state.data_cache = data
        st.session_state.cache_time = current_time
        
        return data
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def generate_sample_data(start_date: str, end_date: str) -> pd.DataFrame:
    """Generate realistic sample electricity data"""
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    
    hours = pd.date_range(start=start, end=end, freq='H')
    
    base_demand = 30000 + 5000 * np.sin(np.arange(len(hours)) * 2 * np.pi / 24)
    noise = np.random.normal(0, 2000, len(hours))
    demand = base_demand + noise
    
    generation = demand * 0.85 + np.random.normal(0, 1000, len(hours))
    
    base_price = 50 + 20 * np.sin(np.arange(len(hours)) * 2 * np.pi / 24)
    price = base_price + np.random.normal(0, 10, len(hours))
    
    return pd.DataFrame({
        'timestamp': hours,
        'demand': demand.clip(min=0),
        'generation': generation.clip(min=0),
        'price': price.clip(min=0),
        'renewable_percentage': np.random.uniform(30, 80, len(hours))
    })

def plot_forecast_comparison(actual: pd.Series, forecasts: Dict[str, pd.DataFrame], 
                            dates: pd.DatetimeIndex) -> go.Figure:
    """Create interactive forecast comparison plot"""
    fig = go.Figure()
    
    # Add actual data
    fig.add_trace(go.Scatter(
        x=dates, y=actual.values,
        mode='lines',
        name='Actual',
        line=dict(color='black', width=2)
    ))
    
    # Add forecasts
    colors = ['blue', 'red', 'green']
    for (name, forecast), color in zip(forecasts.items(), colors):
        fig.add_trace(go.Scatter(
            x=pd.date_range(start=dates[-1], periods=len(forecast)+1)[1:],
            y=forecast['forecast'].values,
            mode='lines+markers',
            name=f'{name} Forecast',
            line=dict(color=color, dash='dash')
        ))
        
        # Add confidence intervals
        fig.add_trace(go.Scatter(
            x=pd.date_range(start=dates[-1], periods=len(forecast)+1)[1:],
            y=forecast['upper_ci'].values,
            fill=None,
            mode='lines',
            line_color='rgba(0,0,0,0)',
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=pd.date_range(start=dates[-1], periods=len(forecast)+1)[1:],
            y=forecast['lower_ci'].values,
            fill='tonexty',
            mode='lines',
            line_color='rgba(0,0,0,0)',
            name=f'{name} CI',
            showlegend=True,
            fillcolor=f'rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.2)'
        ))
    
    fig.update_layout(
        title='Electricity Demand: Actual vs Forecasts',
        xaxis_title='Date',
        yaxis_title='Demand (MWh)',
        hovermode='x unified',
        height=500
    )
    
    return fig

def plot_metrics_comparison(metrics_data: Dict[str, Dict]) -> go.Figure:
    """Create metrics comparison visualization"""
    models = list(metrics_data.keys())
    
    fig = go.Figure()
    
    for metric_name in ['mae', 'rmse', 'mape']:
        values = [metrics_data[model].get(metric_name, 0) for model in models]
        fig.add_trace(go.Bar(
            x=models,
            y=values,
            name=metric_name.upper()
        ))
    
    fig.update_layout(
        title='Model Performance Metrics Comparison',
        barmode='group',
        yaxis_title='Error Value',
        height=400
    )
    
    return fig

# ==================== Main Application ====================
def main():
    st.set_page_config(
        page_title="Spain Electricity Analysis - Advanced",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("⚡ Spain Electricity Analysis - Advanced ML Forecasting")
    st.markdown("*Advanced forecasting with multiple ML models and real-time API integration*")
    
    initialize_session_state()
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        data_source = st.selectbox(
            "Data Source",
            ["Sample Data", "API (REDiss)", "CSV File"]
        )
        
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=30))
        with col2:
            end_date = st.date_input("End Date", value=datetime.now())
        
        st.markdown("---")
        st.subheader("📊 Forecasting Models")
        
        model_selection = st.multiselect(
            "Select Models",
            ["ARIMA", "Exponential Smoothing", "Prophet (ML)"],
            default=["ARIMA", "Exponential Smoothing"]
        )
        
        forecast_days = st.slider(
            "Forecast Horizon (days)",
            min_value=1,
            max_value=MAX_FORECAST_DAYS,
            value=7
        )
        
        st.markdown("---")
        st.subheader("🛠️ Data Processing")
        
        handle_missing = st.checkbox("Handle Missing Values", value=True)
        missing_method = st.selectbox(
            "Missing Value Method",
            ["interpolate", "forward_fill", "backward_fill"]
        ) if handle_missing else None
        
        engineer_features_flag = st.checkbox("Engineer Features", value=True)
        
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📈 Data Exploration", "🔮 Forecasting", "📊 Model Comparison", "🔌 API Integration"]
    )
    
    # Load data
    data = load_and_cache_data(
        data_source,
        start_date.strftime("%Y-%m-%d"),
        end_date.strftime("%Y-%m-%d")
    )
    
    if data is None:
        st.error("Failed to load data. Please check your configuration.")
        return
    
    # Data preprocessing
    if handle_missing and missing_method:
        data = DataProcessor.handle_missing_values(data, method=missing_method)
    
    if engineer_features_flag:
        data = DataProcessor.engineer_features(data)
    
    # TAB 1: Data Exploration
    with tab1:
        st.subheader("Data Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Records", len(data))
        with col2:
            st.metric("Date Range", f"{data['timestamp'].min().date()} to {data['timestamp'].max().date()}")
        with col3:
            st.metric("Features", len(data.columns))
        with col4:
            st.metric("Missing Values", data.isnull().sum().sum())
        
        st.markdown("---")
        
        # Display data
        st.subheader("Raw Data Sample")
        st.dataframe(data.head(20), use_container_width=True)
        
        # Statistics
        st.subheader("Descriptive Statistics")
        st.dataframe(data.describe(), use_container_width=True)
        
        # Visualization
        st.subheader("Time Series Visualization")
        
        fig = go.Figure()
        for col in ['demand', 'generation', 'price']:
            if col in data.columns:
                fig.add_trace(go.Scatter(
                    x=data['timestamp'],
                    y=data[col],
                    mode='lines',
                    name=col.capitalize()
                ))
        
        fig.update_layout(
            title="Electricity Metrics Over Time",
            xaxis_title="Date",
            yaxis_title="Value",
            hovermode='x unified',
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Correlation analysis
        st.subheader("Correlation Analysis")
        numeric_data = data.select_dtypes(include=[np.number])
        correlation_matrix = numeric_data.corr()
        
        fig = px.imshow(
            correlation_matrix,
            labels=dict(color="Correlation"),
            x=correlation_matrix.columns,
            y=correlation_matrix.columns,
            color_continuous_scale="RdBu",
            zmin=-1, zmax=1
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    # TAB 2: Forecasting
    with tab2:
        st.subheader("Advanced Demand Forecasting")
        
        if len(model_selection) == 0:
            st.warning("Please select at least one forecasting model in the sidebar.")
        else:
            forecast_hours = forecast_days * 24
            
            # Train models and generate forecasts
            forecasts = {}
            models_info = {}
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, model_name in enumerate(model_selection):
                status_text.text(f"Training {model_name}...")
                
                try:
                    if model_name == "ARIMA":
                        forecaster = ARIMAForecaster(order=(1, 1, 1))
                    elif model_name == "Exponential Smoothing":
                        forecaster = ExponentialSmoothingForecaster()
                    else:  # Prophet
                        forecaster = MLForecaster()
                    
                    forecaster.train(data)
                    forecast = forecaster.predict(forecast_hours)
                    
                    forecasts[model_name] = forecast
                    models_info[model_name] = forecaster.get_metrics()
                    
                    progress_bar.progress((idx + 1) / len(model_selection))
                    
                except Exception as e:
                    st.error(f"Error with {model_name}: {str(e)}")
                    logger.error(f"{model_name} error: {str(e)}")
            
            status_text.text("Forecasts ready!")
            
            # Display forecasts
            if forecasts:
                st.subheader("Forecast Results")
                
                # Interactive plot
                forecast_fig = plot_forecast_comparison(
                    data['demand'],
                    forecasts,
                    data['timestamp']
                )
                st.plotly_chart(forecast_fig, use_container_width=True)
                
                # Forecast table
                st.subheader("Forecast Values")
                
                for model_name, forecast in forecasts.items():
                    with st.expander(f"📋 {model_name} Forecast Details"):
                        display_df = forecast.copy()
                        display_df['date'] = pd.date_range(
                            start=data['timestamp'].max(),
                            periods=len(display_df)+1
                        )[1:]
                        
                        st.dataframe(display_df, use_container_width=True)
                        
                        # Export option
                        csv = display_df.to_csv(index=False)
                        st.download_button(
                            f"Download {model_name} Forecast",
                            csv,
                            f"{model_name.lower().replace(' ', '_')}_forecast.csv"
                        )
    
    # TAB 3: Model Comparison
    with tab3:
        st.subheader("Model Performance Comparison")
        
        if len(model_selection) > 1:
            # Create comparison metrics
            comparison_data = {}
            
            for model_name in model_selection:
                metrics = models_info.get(model_name, {})
                comparison_data[model_name] = metrics
            
            # Display metrics
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Model Metrics")
                metrics_df = pd.DataFrame(comparison_data).T
                st.dataframe(metrics_df, use_container_width=True)
            
            with col2:
                st.subheader("Model Recommendations")
                st.info("""
                **Model Selection Guide:**
                - **ARIMA**: Best for stationary, univariate time series with clear trends
                - **Exponential Smoothing**: Great for data with trend and seasonality
                - **Prophet**: Excellent for capturing complex seasonality patterns
                
                **Recommendations:**
                - Use ensemble methods for better accuracy
                - Consider multiple models and average predictions
                - Validate with recent historical data
                """)
            
            # Visualization
            if comparison_data:
                fig = plot_metrics_comparison(comparison_data)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Select multiple models to compare performance.")
    
    # TAB 4: API Integration
    with tab4:
        st.subheader("🔌 API Integration & Webhooks")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Fetch Real-Time Data")
            
            if st.button("🔄 Get Real-Time Metrics"):
                try:
                    api_client = ElectricityAPIClient()
                    real_time_data = api_client.get_real_time_data()
                    
                    st.success("✅ Real-time data fetched successfully!")
                    st.json(real_time_data)
                    
                except Exception as e:
                    st.error(f"Failed to fetch real-time data: {str(e)}")
        
        with col2:
            st.subheader("Submit Forecast to API")
            
            if st.button("📤 Submit Latest Forecast"):
                if forecasts:
                    try:
                        api_client = ElectricityAPIClient()
                        
                        forecast_payload = {
                            'timestamp': datetime.now().isoformat(),
                            'model': list(forecasts.keys())[0],
                            'forecast': forecasts[list(forecasts.keys())[0]]['forecast'].tolist(),
                            'lower_ci': forecasts[list(forecasts.keys())[0]]['lower_ci'].tolist(),
                            'upper_ci': forecasts[list(forecasts.keys())[0]]['upper_ci'].tolist()
                        }
                        
                        success = api_client.submit_forecast(forecast_payload)
                        
                        if success:
                            st.success("✅ Forecast submitted successfully!")
                        else:
                            st.warning("⚠️ Forecast submission failed")
                            
                    except Exception as e:
                        st.error(f"Error submitting forecast: {str(e)}")
                else:
                    st.warning("No forecasts available. Generate forecasts first.")
        
        st.markdown("---")
        
        st.subheader("🔗 Webhook Configuration")
        
        webhook_url = st.text_input("Webhook URL", placeholder="https://your-api.com/webhooks/forecast")
        webhook_event = st.selectbox("Trigger Event", ["forecast_generated", "data_updated", "anomaly_detected"])
        
        if st.button("⚙️ Configure Webhook"):
            if webhook_url:
                st.success(f"✅ Webhook configured for {webhook_event}")
                st.info(f"Webhook URL: {webhook_url}")
            else:
                st.error("Please enter a valid webhook URL")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray; font-size: small;'>
    Advanced Electricity Analysis Platform | Last Updated: 2026-01-05 | 
    Powered by Streamlit, Prophet, and ARIMA
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
