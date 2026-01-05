# Spain Electricity Market Analysis Dashboard

## Overview

The Spain Electricity Market Analysis Dashboard is a comprehensive analytical platform designed to provide real-time and historical insights into Spain's electricity market dynamics. This project combines data visualization, market analysis, and predictive modeling to help stakeholders understand trends, patterns, and opportunities in the Spanish electricity sector.

## Project Goals

- **Real-time Monitoring**: Track current electricity prices, demand, and supply metrics across Spain
- **Historical Analysis**: Analyze long-term trends to identify seasonal patterns and market behaviors
- **Market Insights**: Provide actionable intelligence for energy traders, utilities, and policymakers
- **Data Integration**: Aggregate data from multiple sources (REE, OMIE, ESIOS, etc.)
- **Forecasting**: Predict electricity prices and demand using advanced analytics

## Features

### Dashboard Components

#### 1. **Price Analysis**
- Real-time electricity price tracking
- Historical price trends with interactive charts
- Price volatility analysis
- Regional price comparisons

#### 2. **Demand & Supply Monitoring**
- Current electricity demand levels
- Generation capacity by source
- Supply-demand balance visualization
- Peak demand analysis

#### 3. **Generation Mix**
- Renewable energy contribution tracking
- Nuclear, thermal, and hydroelectric generation breakdown
- Real-time generation by source
- CO2 emission estimates

#### 4. **Market Insights**
- Trend analysis and pattern recognition
- Market anomalies detection
- Year-over-year comparisons
- Seasonal demand patterns

#### 5. **Forecasting Tools**
- Short-term price predictions
- Demand forecasting
- Renewable generation predictions
- Confidence intervals and model performance metrics

## Data Sources

This project integrates data from several authoritative sources:

- **REE (Red Eléctrica de España)**: Spanish electrical transmission system operator
- **OMIE (Operador del Mercado Ibérico de Electricidad)**: Iberian electricity market operator
- **ESIOS (Sistema de Información del Operador del Sistema)**: Energy information system
- **Public APIs**: Open data from Spanish energy authorities

## Technology Stack

### Backend
- **Python 3.8+**: Core data processing and analysis
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Scikit-learn**: Machine learning and forecasting models
- **SQLAlchemy**: Database ORM

### Frontend
- **Streamlit**: Interactive web dashboard
- **Plotly**: Interactive visualizations
- **Pandas DataFrames**: Data display and tables

### Database
- **PostgreSQL**: Historical data storage
- **Redis**: Caching and real-time data

### Deployment
- **Docker**: Containerization
- **GitHub Actions**: CI/CD pipeline

## Installation

### Prerequisites
- Python 3.8 or higher
- pip or conda package manager
- PostgreSQL (for production deployment)
- Docker (optional, for containerized deployment)

### Setup Instructions

1. **Clone the repository**
```bash
git clone https://github.com/xczernia/spain-electricity-analysis.git
cd spain-electricity-analysis
```

2. **Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure environment variables**
```bash
cp .env.example .env
# Edit .env with your configuration
```

5. **Initialize the database**
```bash
python scripts/setup_database.py
```

## Usage

### Running the Dashboard

```bash
streamlit run app/main.py
```

The dashboard will be available at `http://localhost:8501`

### Data Collection

To collect the latest electricity market data:

```bash
python scripts/collect_market_data.py
```

### Running Analysis

Execute analysis and generate reports:

```bash
python analysis/market_analysis.py
```

## Project Structure

```
spain-electricity-analysis/
├── app/
│   ├── main.py                 # Main Streamlit application
│   ├── pages/                  # Dashboard pages
│   │   ├── price_analysis.py
│   │   ├── demand_supply.py
│   │   ├── generation_mix.py
│   │   └── forecasting.py
│   └── components/             # Reusable UI components
├── data/
│   ├── raw/                    # Raw data from sources
│   ├── processed/              # Processed and cleaned data
│   └── queries/                # SQL queries
├── analysis/
│   ├── market_analysis.py      # Market analysis functions
│   ├── forecasting_models.py   # Predictive models
│   └── utils.py               # Utility functions
├── scripts/
│   ├── collect_market_data.py  # Data collection script
│   ├── setup_database.py       # Database initialization
│   └── scheduled_tasks.py      # Scheduled data updates
├── tests/
│   ├── test_analysis.py
│   ├── test_forecasting.py
│   └── test_data_collection.py
├── config/
│   ├── settings.py             # Configuration settings
│   └── logging.py              # Logging configuration
├── requirements.txt            # Python dependencies
├── docker-compose.yml          # Docker composition
├── Dockerfile                  # Container configuration
├── .env.example               # Example environment variables
└── README.md                  # This file
```

## Key Metrics & KPIs

### Price Metrics
- **Average Spot Price**: Mean electricity price in €/MWh
- **Price Volatility**: Standard deviation and coefficient of variation
- **Peak/Off-peak Ratio**: Price differential between peak and off-peak hours
- **Price Range**: Daily minimum and maximum prices

### Demand Metrics
- **Peak Demand**: Maximum hourly demand in MW
- **Average Demand**: Mean demand over period
- **Demand Growth**: YoY percentage change
- **Load Factor**: Actual demand vs. maximum capacity

### Supply Metrics
- **Renewable Penetration**: % of electricity from renewables
- **Generation by Type**: Breakdown by fuel source
- **Capacity Utilization**: Generation vs. available capacity
- **Reserve Margin**: Available capacity above demand

## API Reference

### Market Data Endpoints

```python
# Get current market price
from analysis.market_analysis import get_current_price
price = get_current_price()

# Get historical prices
from analysis.market_analysis import get_price_history
prices = get_price_history(start_date='2025-01-01', end_date='2025-12-31')

# Get demand data
from analysis.market_analysis import get_demand_data
demand = get_demand_data(period='today')
```

## Configuration

### Environment Variables

Create a `.env` file with the following variables:

```
# Database
DATABASE_URL=postgresql://user:password@localhost:5432/spain_electricity

# API Keys (if required)
REE_API_KEY=your_api_key
OMIE_API_KEY=your_api_key

# Data Settings
DATA_UPDATE_INTERVAL=3600  # seconds
HISTORICAL_DATA_RETENTION=365  # days

# Application
DEBUG=False
LOG_LEVEL=INFO
```

## Dashboard Walkthrough

### Home Page
- Current market status summary
- Key metrics overview
- System alerts and notifications
- Quick links to detailed analysis

### Price Analysis Page
- Interactive price charts with zoom and pan
- Historical trends with moving averages
- Price statistics (mean, median, std dev)
- Price correlation with demand and generation

### Demand & Supply Page
- Real-time demand vs. available capacity
- Hourly demand patterns
- Weekly and seasonal trends
- Supply adequacy indicators

### Generation Mix Page
- Pie charts showing generation by source
- Time series of renewable generation
- CO2 intensity metrics
- Fuel source comparison

### Forecasting Page
- Next-day price predictions
- Demand forecasts
- Model performance metrics
- Historical accuracy tracking

## Data Updates & Maintenance

### Automatic Updates
- Market data: Every hour
- Historical data: Daily batch processing
- Model retraining: Weekly
- Data cleanup: Monthly

### Manual Updates
```bash
# Force data refresh
python scripts/collect_market_data.py --force

# Rebuild cache
python scripts/clear_cache.py

# Validate data integrity
python scripts/validate_data.py
```

## Performance & Optimization

- **Data Caching**: Redis caching for frequently accessed data
- **Database Indexing**: Optimized queries on large historical datasets
- **Lazy Loading**: Dashboard pages load data on demand
- **Pagination**: Large datasets displayed with pagination controls

## Security & Privacy

- **Authentication**: Role-based access control (RBAC)
- **Data Encryption**: SSL/TLS for data transmission
- **API Rate Limiting**: Prevent unauthorized access
- **Audit Logging**: Track all data access and modifications
- **Data Privacy**: Compliance with GDPR and Spanish data protection laws

## Contributing

We welcome contributions! Please follow these steps:

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/AmazingFeature`)
3. **Commit changes** (`git commit -m 'Add some AmazingFeature'`)
4. **Push to branch** (`git push origin feature/AmazingFeature`)
5. **Open a Pull Request**

### Code Standards
- Follow PEP 8 style guide
- Write unit tests for new features
- Update documentation accordingly
- Ensure all tests pass before submitting PR

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=analysis --cov=app

# Run specific test file
pytest tests/test_analysis.py
```

### Test Coverage
- Unit tests: Core functions and classes
- Integration tests: Data pipeline and API endpoints
- End-to-end tests: Dashboard functionality

## Troubleshooting

### Common Issues

**Issue**: Database connection error
```
Solution: Verify DATABASE_URL in .env and ensure PostgreSQL is running
```

**Issue**: Data collection fails
```
Solution: Check API credentials and network connectivity
```

**Issue**: Dashboard loads slowly
```
Solution: Clear cache and restart the application
```

## Performance Monitoring

- **Dashboard Response Time**: Target < 2 seconds
- **Data Update Latency**: < 5 minutes behind real-time
- **API Availability**: Target 99.5% uptime
- **Data Accuracy**: Validated against official sources

## Roadmap

### Planned Features
- [ ] Mobile application for market alerts
- [ ] Advanced machine learning models (LSTM, ensemble methods)
- [ ] Real-time alerts for price anomalies
- [ ] Integration with trading platforms
- [ ] Multi-country European market comparison
- [ ] Carbon footprint tracking
- [ ] AI-powered trading recommendations

### Upcoming Improvements
- Enhanced forecasting accuracy
- Extended historical data retention
- API for external integrations
- Customizable dashboards
- Export functionality (CSV, Excel, PDF)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Red Eléctrica de España (REE) for electricity transmission data
- OMIE for market pricing information
- The open-source community for excellent tools and libraries
- Contributors and testers who help improve this project

## Contact & Support

- **Issues**: Report bugs and issues on [GitHub Issues](https://github.com/xczernia/spain-electricity-analysis/issues)
- **Discussions**: Join our [GitHub Discussions](https://github.com/xczernia/spain-electricity-analysis/discussions)
- **Email**: Support available via GitHub project contact

## Related Resources

- [Spanish Electricity Market Overview](https://www.omie.es/)
- [REE Official Website](https://www.ree.es/)
- [ESIOS Platform](https://www.esios.ree.es/)
- [EU Electricity Market Regulations](https://ec.europa.eu/energy/)

---

**Last Updated**: January 5, 2026

For the latest updates, visit our [GitHub repository](https://github.com/xczernia/spain-electricity-analysis)
