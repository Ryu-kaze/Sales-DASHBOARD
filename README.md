# 📈 SalesLens — AI Sales Forecast Intelligence

A production-quality Streamlit app for sales forecasting, trend analysis, and AI-generated insights.

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the app
```bash
streamlit run app.py
```

Your browser will open at **http://localhost:8501**

---

## 🔥 Features

| Feature | Description |
|---|---|
| **CSV / Excel Upload** | Upload your own sales file — auto-detects date & sales columns |
| **Sample Products** | 4 built-in demo products with realistic seasonality |
| **ML Forecasting** | Polynomial regression + seasonal decomposition |
| **Confidence Intervals** | Upper/lower bounds on all forecasts |
| **AI Insights** | Auto-generated plain-English analysis of trends, returns, and seasonality |
| **Why Not Selling** | Built-in checklist of common sales failure factors |
| **Analytics** | Revenue waterfall, return rate bars, YoY comparison, monthly heatmap |
| **CSV Export** | Download history or forecast as CSV |

---

## 📂 Supported CSV Format

Your CSV should contain these columns (names are flexible — the app auto-detects):

```csv
date,product,units_sold,revenue,units_returned
2024-01-01,My Product,5420,162600.00,271
2024-02-01,My Product,4980,149400.00,249
```

**Minimum required:** a date column + a sales/units column.  
Everything else is optional.

A sample file is included: **`sample_data.csv`**

---

## 🎛️ Sidebar Controls

- **Data Source** — Switch between sample data or your own upload
- **Months to Forecast** — Slider from 3 to 12 months ahead
- **Model Complexity** — Simple (linear) → Standard (quadratic) → Advanced (cubic)
- **Display options** — Toggle raw table and formula view

---

## 🧠 How the Forecast Works

The model uses **Polynomial Regression with Seasonal Encoding**:
- **Time trend** — captures overall growth or decline
- **Sine/Cosine seasonality** — captures monthly seasonal cycles
- **Polynomial degree** — controls how complex the trend curve is

MAPE (Mean Absolute Percentage Error) measures accuracy:
- < 5% = Excellent
- 5–12% = Good
- > 12% = Moderate (more data helps)

---

## 📦 Requirements

```
streamlit>=1.32.0
pandas>=2.0.0
numpy>=1.24.0
plotly>=5.18.0
scikit-learn>=1.3.0
openpyxl>=3.1.0
```

---

## 🏗️ Project Structure

```
sales_forecast_app/
├── app.py              ← Main Streamlit application
├── requirements.txt    ← Python dependencies
├── sample_data.csv     ← Test data file
└── README.md           ← This file
```

---

Built with ❤️ using Streamlit, Plotly, and scikit-learn.
