# Stock Price Anomaly Detector

A machine learning application designed to detect anomalies in stock price data using advanced statistical and machine learning techniques.

## Project Overview

This project provides tools for:
- Fetching historical stock price data
- Preprocessing and cleaning financial data
- Detecting price anomalies using multiple algorithms
- Visualizing results with interactive charts
- Interactive web interface using Streamlit

## Project Structure

```
stock-anomaly-detector/
│
├── data/               # Raw + processed data
├── notebooks/          # Exploratory analysis (.ipynb files)
├── src/                # Clean Python scripts
│   ├── fetch_data.py
│   ├── preprocess.py
│   ├── detect_anomalies.py
│   └── visualize.py
├── app.py              # Streamlit app
├── requirements.txt
├── .gitignore
└── README.md
```

## Requirements

- Python 3.8+
- Dependencies listed in `requirements.txt`

## Installation

1. Clone the repository:
```bash
git clone https://github.com/anishitagi-droid/Stock-Price-Anomaly-Detector.git
cd Stock-Price-Anomaly-Detector
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Run the Streamlit App
```bash
streamlit run app.py
```

### Use Python Scripts

#### Fetch Data
```python
from src.fetch_data import get_stock_data
data = get_stock_data('AAPL', start_date='2023-01-01', end_date='2024-01-01')
```

#### Preprocess Data
```python
from src.preprocess import preprocess_data
processed_data = preprocess_data(data)
```

#### Detect Anomalies
```python
from src.detect_anomalies import detect_anomalies
anomalies = detect_anomalies(processed_data)
```

#### Visualize Results
```python
from src.visualize import plot_anomalies
plot_anomalies(processed_data, anomalies)
```

## Algorithms

- **Isolation Forest**: Detects outliers based on feature isolation
- **Local Outlier Factor (LOF)**: Identifies local density-based anomalies
- **Z-Score**: Statistical method for outlier detection
- **Moving Average**: Deviation from moving average threshold

## Data

Raw data is stored in the `data/` directory. The project can fetch data from:
- Yahoo Finance (yfinance)
- Other financial data sources

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions or suggestions, please open an issue on the repository.
