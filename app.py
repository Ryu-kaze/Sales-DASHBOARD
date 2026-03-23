import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import io
import json
import re
import requests
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_percentage_error
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# GEMINI CONFIG
# ─────────────────────────────────────────────
GEMINI_API_KEY = "AIzaSyAY7dMSyjQ4sr8vvatYD-mluzaXQFLwE9w"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

def call_gemini(prompt: str) -> str:
    """Call Gemini API and return the text response."""
    try:
        resp = requests.post(
            f"{GEMINI_URL}?key={GEMINI_API_KEY}",
            headers={"Content-Type": "application/json"},
            json={"contents": [{"parts": [{"text": prompt}]}]},
            timeout=30
        )
        data = resp.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        return None


def build_gemini_prompt(df, forecast_df, mape, product_name):
    """Build a rich data prompt for Gemini to analyse."""
    units = df["units_sold"].tolist()
    dates = df["date"].dt.strftime("%b %Y").tolist()
    trend_slope = float(np.polyfit(range(len(units)), units, 1)[0])
    total_units = sum(units)
    avg_monthly = np.mean(units)
    next_3_forecast = forecast_df["forecast"].iloc[:3].tolist()
    forecast_months = forecast_df["date"].dt.strftime("%b %Y").iloc[:3].tolist()

    return_info = ""
    if "units_returned" in df.columns and df["units_returned"].sum() > 0:
        avg_rr = (df["units_returned"] / df["units_sold"]).mean() * 100
        return_info = f"- Average return rate: {avg_rr:.1f}%"

    revenue_info = ""
    if "revenue" in df.columns:
        total_rev = df["revenue"].sum()
        revenue_info = f"- Total revenue: ${total_rev:,.0f}"

    prompt = f"""You are a senior e-commerce sales analyst. Analyse the following product sales data and return a structured JSON response.

PRODUCT: {product_name}
SALES DATA (monthly units sold):
{dict(zip(dates, units))}

KEY METRICS:
- Total units sold: {total_units:,}
- Average monthly sales: {avg_monthly:.0f} units
- Monthly trend slope: {trend_slope:+.1f} units/month
- Forecast accuracy (MAPE): {mape:.1f}%
{return_info}
{revenue_info}

NEXT 3-MONTH FORECAST:
{dict(zip(forecast_months, next_3_forecast))}

Return ONLY a valid JSON object (no markdown, no backticks) with this exact structure:
{{
  "summary": "2-3 sentence executive summary of overall product performance",
  "insights": [
    {{
      "type": "success|warning|danger|info",
      "title": "Short insight title with emoji",
      "text": "2-3 sentence detailed insight with specific numbers from the data"
    }}
  ],
  "why_not_selling": [
    {{
      "type": "warning|danger|info",
      "title": "Reason title with emoji",
      "text": "Specific reason tailored to this product's data pattern"
    }}
  ],
  "recommendations": [
    "Actionable recommendation 1 with specific detail",
    "Actionable recommendation 2 with specific detail",
    "Actionable recommendation 3 with specific detail"
  ],
  "risk_level": "low|medium|high",
  "risk_reason": "One sentence explaining the risk assessment"
}}

Generate exactly 5 insights and 4 why_not_selling reasons. Base everything on the actual data provided. Be specific and quantitative."""
    return prompt

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="SalesLens – AI Forecast Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }
    .main-title {
        font-family: 'DM Serif Display', serif;
        font-size: 2.8rem;
        color: #0f172a;
        letter-spacing: -0.02em;
        line-height: 1.1;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1rem;
        color: #64748b;
        font-weight: 400;
        margin-bottom: 2rem;
    }
    .kpi-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 1.4rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        position: relative;
        overflow: hidden;
    }
    .kpi-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, #6366f1, #8b5cf6);
    }
    .kpi-label { font-size: 0.75rem; font-weight: 600; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.08em; }
    .kpi-value { font-size: 1.9rem; font-weight: 600; color: #0f172a; line-height: 1.2; margin: 0.3rem 0; }
    .kpi-sub { font-size: 0.8rem; color: #64748b; }
    .kpi-up { color: #10b981; font-weight: 600; }
    .kpi-down { color: #ef4444; font-weight: 600; }
    .insight-box {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        border: 1px solid #bae6fd;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.8rem;
    }
    .insight-box.warning {
        background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
        border-color: #fcd34d;
    }
    .insight-box.danger {
        background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
        border-color: #fca5a5;
    }
    .insight-box.success {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        border-color: #86efac;
    }
    .insight-title { font-weight: 600; color: #0f172a; font-size: 0.9rem; margin-bottom: 0.3rem; }
    .insight-text { font-size: 0.82rem; color: #475569; line-height: 1.5; }
    .section-header {
        font-family: 'DM Serif Display', serif;
        font-size: 1.4rem;
        color: #0f172a;
        margin-bottom: 0.5rem;
    }
    .badge {
        display: inline-block;
        padding: 0.2rem 0.7rem;
        border-radius: 999px;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }
    .badge-purple { background: #ede9fe; color: #7c3aed; }
    .badge-green { background: #dcfce7; color: #16a34a; }
    .badge-red { background: #fee2e2; color: #dc2626; }
    .badge-amber { background: #fef3c7; color: #d97706; }
    div[data-testid="stSidebar"] { background: #0f172a; }
    div[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
    div[data-testid="stSidebar"] .stSelectbox label,
    div[data-testid="stSidebar"] .stFileUploader label { color: #94a3b8 !important; font-size: 0.8rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; }
    .stTabs [data-baseweb="tab-list"] { gap: 0.5rem; border-bottom: 2px solid #e2e8f0; }
    .stTabs [data-baseweb="tab"] { background: transparent; border: none; padding: 0.6rem 1.2rem; font-weight: 500; color: #64748b; }
    .stTabs [aria-selected="true"] { color: #6366f1 !important; border-bottom: 2px solid #6366f1; }
    .stButton button { border-radius: 8px; font-weight: 500; }
    .stDataFrame { border-radius: 12px; overflow: hidden; }
    footer { display: none; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HELPER: SAMPLE DATA
# ─────────────────────────────────────────────
def generate_sample_data(product="Wireless Earbuds Pro"):
    months = pd.date_range(start="2023-01-01", periods=18, freq="MS")
    base = [8500, 7800, 9200, 9800, 10500, 11500,
            10800, 11200, 12500, 13500, 15500, 18000,
            9500, 8800, 10200, 11000, 11800, 12800]
    noise = np.random.normal(0, 300, 18).astype(int)
    units = [b + n for b, n in zip(base, noise)]
    prices = [29.99] * 18
    returns = [int(u * np.random.uniform(0.03, 0.08)) for u in units]
    df = pd.DataFrame({
        "date": months,
        "product": product,
        "units_sold": units,
        "price": prices,
        "revenue": [u * p for u, p in zip(units, prices)],
        "units_returned": returns,
        "category": "Electronics",
        "region": "North America"
    })
    return df


# ─────────────────────────────────────────────
# HELPER: PARSE UPLOADED CSV
# ─────────────────────────────────────────────
def parse_uploaded_file(uploaded_file):
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        # Try to auto-detect date column
        date_cols = [c for c in df.columns if any(k in c.lower() for k in ["date", "month", "period", "time", "week"])]
        sales_cols = [c for c in df.columns if any(k in c.lower() for k in ["sales", "units", "sold", "revenue", "qty", "quantity"])]

        if date_cols:
            df[date_cols[0]] = pd.to_datetime(df[date_cols[0]], errors="coerce")
            df = df.rename(columns={date_cols[0]: "date"})
        else:
            df["date"] = pd.date_range(start="2023-01-01", periods=len(df), freq="MS")

        if sales_cols:
            df = df.rename(columns={sales_cols[0]: "units_sold"})
        else:
            st.warning("Could not auto-detect sales column. Using first numeric column.")
            num_col = df.select_dtypes(include=np.number).columns[0]
            df = df.rename(columns={num_col: "units_sold"})

        if "product" not in df.columns:
            df["product"] = "Product A"
        if "revenue" not in df.columns:
            df["revenue"] = df["units_sold"] * 29.99
        if "units_returned" not in df.columns:
            df["units_returned"] = (df["units_sold"] * 0.05).astype(int)

        df = df.dropna(subset=["date", "units_sold"])
        df = df.sort_values("date").reset_index(drop=True)
        return df, None
    except Exception as e:
        return None, str(e)


# ─────────────────────────────────────────────
# FORECASTING ENGINE
# ─────────────────────────────────────────────
def run_forecast(df, periods=6, degree=2):
    df = df.copy()
    df["t"] = np.arange(len(df))
    df["month_sin"] = np.sin(2 * np.pi * df["date"].dt.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["date"].dt.month / 12)

    X = df[["t", "month_sin", "month_cos"]].values
    y = df["units_sold"].values

    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(X)

    model = LinearRegression()
    model.fit(X_poly, y)

    # In-sample predictions
    y_pred = model.predict(X_poly)
    mape = mean_absolute_percentage_error(y, y_pred) * 100

    # Future forecast
    last_t = df["t"].max()
    last_date = df["date"].max()
    future_dates = [last_date + pd.DateOffset(months=i+1) for i in range(periods)]
    future_t = [last_t + i + 1 for i in range(periods)]
    future_sin = [np.sin(2 * np.pi * d.month / 12) for d in future_dates]
    future_cos = [np.cos(2 * np.pi * d.month / 12) for d in future_dates]

    X_future = np.column_stack([future_t, future_sin, future_cos])
    X_future_poly = poly.transform(X_future)
    y_future = model.predict(X_future_poly)
    y_future = np.maximum(y_future, 0)

    # Confidence interval (±1.5 * residual std)
    residuals = y - y_pred
    std = np.std(residuals)
    ci_upper = y_future + 1.5 * std
    ci_lower = np.maximum(y_future - 1.5 * std, 0)

    forecast_df = pd.DataFrame({
        "date": future_dates,
        "forecast": y_future.astype(int),
        "ci_upper": ci_upper.astype(int),
        "ci_lower": ci_lower.astype(int),
    })

    return forecast_df, y_pred.astype(int), mape, model, poly


# ─────────────────────────────────────────────
# AI INSIGHTS GENERATOR  (Gemini-powered)
# ─────────────────────────────────────────────
def generate_insights_fallback(df, forecast_df, mape, product_name):
    """Fallback static insights if Gemini is unavailable."""
    insights, why_not, recs = [], [], []
    units = df["units_sold"].values
    trend_slope = np.polyfit(range(len(units)), units, 1)[0]

    if trend_slope > 50:
        insights.append(("success", "📈 Strong Growth Trend",
            f"{product_name} shows a consistent upward trend (+{trend_slope:.0f} units/month). Strong market demand detected."))
    elif trend_slope < -50:
        insights.append(("danger", "📉 Declining Sales",
            f"Sales are declining at ~{abs(trend_slope):.0f} units/month. Review pricing and competitor activity."))
    else:
        insights.append(("info", "➡️ Stable Sales Pattern",
            f"Sales are relatively stable ({trend_slope:+.0f} units/month drift). Focus on retention and upsell."))

    next_3 = forecast_df["forecast"].iloc[:3].mean()
    current_3 = df["units_sold"].iloc[-3:].mean()
    change_pct = ((next_3 - current_3) / current_3) * 100
    insights.append(("success" if change_pct > 0 else "warning", "📊 Forecast Outlook",
        f"Next 3-month avg: {next_3:.0f} units ({change_pct:+.1f}% vs recent performance)."))

    why_not = [
        ("warning", "🏷️ Pricing Pressure", "Prices may be misaligned with market expectations. Monitor competitor pricing weekly."),
        ("danger",  "📦 Inventory Risk",   "Potential stockout risk during peak periods. Review reorder points against forecast."),
        ("info",    "🔍 Search Visibility","Low organic discoverability may be suppressing demand. Audit keywords and listings."),
        ("warning", "⭐ Review Sentiment", "Customer sentiment directly impacts conversion. Maintain rating above 4.0 stars."),
    ]
    recs = [
        "Align inventory levels with the 3-month forecast to avoid stockouts.",
        "Run targeted promotions during historically low sales months.",
        "Audit product listings and keywords to improve search visibility.",
    ]
    summary = f"{product_name} has {total_units:,} total units sold with a MAPE of {mape:.1f}%."
    return summary, insights, why_not, recs, "medium", "Based on trend and forecast variance."


@st.cache_data(show_spinner=False)
def generate_gemini_insights(df_json, forecast_json, mape, product_name):
    """Call Gemini and parse structured JSON insights. Cached per data snapshot."""
    df = pd.read_json(io.StringIO(df_json))
    df["date"] = pd.to_datetime(df["date"])
    forecast_df = pd.read_json(io.StringIO(forecast_json))
    forecast_df["date"] = pd.to_datetime(forecast_df["date"])

    prompt = build_gemini_prompt(df, forecast_df, mape, product_name)
    raw = call_gemini(prompt)
    if raw is None:
        return None

    # Strip markdown fences if present
    clean = re.sub(r"```json|```", "", raw).strip()
    try:
        return json.loads(clean)
    except Exception:
        # Try extracting JSON block
        match = re.search(r"\{.*\}", clean, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except Exception:
                pass
    return None


# ─────────────────────────────────────────────
# PLOT FUNCTIONS
# ─────────────────────────────────────────────
def plot_forecast(df, forecast_df, y_pred, product_name):
    fig = go.Figure()

    # Confidence interval shading
    fig.add_trace(go.Scatter(
        x=list(forecast_df["date"]) + list(forecast_df["date"])[::-1],
        y=list(forecast_df["ci_upper"]) + list(forecast_df["ci_lower"])[::-1],
        fill="toself",
        fillcolor="rgba(99,102,241,0.1)",
        line=dict(color="rgba(255,255,255,0)"),
        hoverinfo="skip",
        name="Confidence Band"
    ))

    # Actual sales
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["units_sold"],
        mode="lines+markers",
        name="Actual Sales",
        line=dict(color="#0f172a", width=2.5),
        marker=dict(size=6, color="#0f172a"),
    ))

    # Model fit
    fig.add_trace(go.Scatter(
        x=df["date"], y=y_pred,
        mode="lines",
        name="Model Fit",
        line=dict(color="#6366f1", width=1.5, dash="dot"),
    ))

    # Forecast
    # Connect last actual to first forecast
    connect_x = [df["date"].iloc[-1], forecast_df["date"].iloc[0]]
    connect_y = [df["units_sold"].iloc[-1], forecast_df["forecast"].iloc[0]]
    fig.add_trace(go.Scatter(
        x=connect_x, y=connect_y,
        mode="lines", showlegend=False,
        line=dict(color="#6366f1", width=2.5, dash="dash")
    ))

    fig.add_trace(go.Scatter(
        x=forecast_df["date"], y=forecast_df["forecast"],
        mode="lines+markers",
        name="Forecast",
        line=dict(color="#6366f1", width=2.5, dash="dash"),
        marker=dict(size=7, color="#6366f1", symbol="diamond"),
    ))

    # Vertical divider
    fig.add_vline(
        x=df["date"].iloc[-1].timestamp() * 1000,
        line_dash="dot", line_color="#94a3b8", line_width=1.5,
        annotation_text="Forecast →", annotation_position="top right",
        annotation_font_color="#94a3b8"
    )

    fig.update_layout(
        title=dict(text=f"<b>{product_name}</b> — Sales History & Forecast", font=dict(size=16, color="#0f172a")),
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=420,
        margin=dict(l=0, r=0, t=50, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        xaxis=dict(showgrid=False, tickfont=dict(size=11)),
        yaxis=dict(gridcolor="#f1f5f9", tickfont=dict(size=11), title="Units Sold"),
    )
    return fig


def plot_monthly_heatmap(df):
    df = df.copy()
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    pivot = df.pivot_table(index="year", columns="month", values="units_sold", aggfunc="sum")
    pivot.columns = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

    fig = px.imshow(
        pivot, color_continuous_scale="Blues",
        labels=dict(color="Units Sold"),
        aspect="auto"
    )
    fig.update_layout(
        title="<b>Monthly Sales Heatmap</b>",
        plot_bgcolor="white", paper_bgcolor="white",
        height=220, margin=dict(l=0, r=0, t=40, b=0)
    )
    return fig


def plot_return_rate(df):
    if "units_returned" not in df.columns:
        return None
    df = df.copy()
    df["return_rate"] = df["units_returned"] / df["units_sold"] * 100
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["date"], y=df["return_rate"],
        marker_color=["#ef4444" if r > 8 else "#f59e0b" if r > 5 else "#10b981" for r in df["return_rate"]],
        name="Return Rate %"
    ))
    fig.add_hline(y=8, line_dash="dot", line_color="#ef4444", annotation_text="8% Alert")
    fig.add_hline(y=5, line_dash="dot", line_color="#f59e0b", annotation_text="5% Warning")
    fig.update_layout(
        title="<b>Monthly Return Rate (%)</b>",
        plot_bgcolor="white", paper_bgcolor="white",
        height=280, margin=dict(l=0, r=0, t=40, b=0),
        yaxis=dict(gridcolor="#f1f5f9", title="Return Rate %"),
        xaxis=dict(showgrid=False)
    )
    return fig


def plot_revenue_waterfall(df):
    monthly_rev = df.groupby(df["date"].dt.to_period("M"))["revenue"].sum().reset_index()
    monthly_rev["date"] = monthly_rev["date"].astype(str)
    monthly_rev["change"] = monthly_rev["revenue"].diff().fillna(0)
    colors = ["#10b981" if c >= 0 else "#ef4444" for c in monthly_rev["change"]]
    fig = go.Figure(go.Bar(
        x=monthly_rev["date"], y=monthly_rev["revenue"],
        marker_color=colors, name="Revenue"
    ))
    fig.update_layout(
        title="<b>Monthly Revenue</b>",
        plot_bgcolor="white", paper_bgcolor="white",
        height=280, margin=dict(l=0, r=0, t=40, b=0),
        yaxis=dict(gridcolor="#f1f5f9", title="Revenue ($)", tickformat="$,.0f"),
        xaxis=dict(showgrid=False, tickangle=-45)
    )
    return fig


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📈 SalesLens")
    st.markdown("<p style='color:#475569;font-size:0.82rem;margin-top:-10px;'>AI Forecast Intelligence</p>", unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("**DATA SOURCE**")
    data_source = st.radio("", ["📦 Use Sample Data", "📂 Upload CSV/Excel"], label_visibility="collapsed")

    uploaded_file = None
    selected_product = "Wireless Earbuds Pro"

    if data_source == "📂 Upload CSV/Excel":
        uploaded_file = st.file_uploader("Upload your sales file", type=["csv", "xlsx", "xls"])
        st.markdown("""
        <div style='background:#1e293b;border-radius:8px;padding:0.8rem;font-size:0.78rem;color:#94a3b8;margin-top:0.5rem;'>
        <b style='color:#e2e8f0'>Expected columns:</b><br>
        • <code>date</code> or <code>month</code><br>
        • <code>units_sold</code> or <code>sales</code><br>
        • <code>product</code> (optional)<br>
        • <code>revenue</code> (optional)<br>
        • <code>units_returned</code> (optional)
        </div>
        """, unsafe_allow_html=True)
    else:
        selected_product = st.selectbox("Sample Product", [
            "Wireless Earbuds Pro",
            "Smart Watch Series X",
            "Bluetooth Speaker Max",
            "Gaming Headset Ultra"
        ])

    st.markdown("---")
    st.markdown("**FORECAST SETTINGS**")
    forecast_periods = st.slider("Months to Forecast", 3, 12, 6)
    model_complexity = st.select_slider("Model Complexity", ["Simple", "Standard", "Advanced"], value="Standard")
    complexity_map = {"Simple": 1, "Standard": 2, "Advanced": 3}

    st.markdown("---")
    st.markdown("**DISPLAY**")
    show_raw = st.checkbox("Show Raw Data Table", value=False)
    show_formula = st.checkbox("Show Forecast Formulas", value=False)

    st.markdown("---")
    st.markdown("<p style='color:#334155;font-size:0.72rem;text-align:center;'>Built for sales intelligence teams</p>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
df = None
error_msg = None

if data_source == "📂 Upload CSV/Excel" and uploaded_file is not None:
    df, error_msg = parse_uploaded_file(uploaded_file)
    if df is not None:
        products = df["product"].unique() if "product" in df.columns else ["All Products"]
        if len(products) > 1:
            selected_product = st.sidebar.selectbox("Select Product", products)
            df = df[df["product"] == selected_product]
        else:
            selected_product = str(products[0])
elif data_source == "📦 Use Sample Data":
    df = generate_sample_data(selected_product)


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.markdown('<div class="main-title">SalesLens</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">AI-powered sales forecasting & intelligence platform</div>', unsafe_allow_html=True)
with col_h2:
    st.markdown("<br>", unsafe_allow_html=True)
    if df is not None:
        csv_export = df.to_csv(index=False)
        st.download_button("⬇️ Export Data", csv_export, f"{selected_product.replace(' ','_')}_sales.csv", "text/csv")


# ─────────────────────────────────────────────
# EMPTY STATE
# ─────────────────────────────────────────────
if df is None and data_source == "📂 Upload CSV/Excel":
    st.markdown("""
    <div style='text-align:center;padding:4rem 2rem;background:#f8fafc;border-radius:20px;border:2px dashed #e2e8f0;'>
        <div style='font-size:3rem;margin-bottom:1rem;'>📂</div>
        <div style='font-size:1.3rem;font-weight:600;color:#0f172a;margin-bottom:0.5rem;'>Upload your sales data</div>
        <div style='color:#64748b;max-width:400px;margin:0 auto;'>
            Upload a CSV or Excel file with your sales history to get AI-powered forecasts and insights.
            The app will auto-detect your date and sales columns.
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

if error_msg:
    st.error(f"Error parsing file: {error_msg}")
    st.stop()


# ─────────────────────────────────────────────
# RUN FORECAST
# ─────────────────────────────────────────────
degree = complexity_map[model_complexity]
forecast_df, y_pred, mape, model, poly = run_forecast(df, forecast_periods, degree)


# ─────────────────────────────────────────────
# KPI CARDS
# ─────────────────────────────────────────────
total_units = df["units_sold"].sum()
total_revenue = df["revenue"].sum() if "revenue" in df.columns else total_units * 29.99
avg_monthly = df["units_sold"].mean()
next_month_forecast = forecast_df["forecast"].iloc[0]
forecast_change = ((next_month_forecast - df["units_sold"].iloc[-1]) / df["units_sold"].iloc[-1]) * 100
total_returned = df["units_returned"].sum() if "units_returned" in df.columns else 0
return_rate = (total_returned / total_units * 100) if total_units > 0 else 0

c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Total Units Sold</div>
        <div class="kpi-value">{total_units:,}</div>
        <div class="kpi-sub">Across {len(df)} months</div>
    </div>""", unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Total Revenue</div>
        <div class="kpi-value">${total_revenue/1000:.1f}K</div>
        <div class="kpi-sub">Avg ${total_revenue/len(df)/1000:.1f}K/month</div>
    </div>""", unsafe_allow_html=True)

with c3:
    trend_arrow = "↑" if forecast_change >= 0 else "↓"
    trend_class = "kpi-up" if forecast_change >= 0 else "kpi-down"
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Next Month Forecast</div>
        <div class="kpi-value">{next_month_forecast:,}</div>
        <div class="kpi-sub"><span class="{trend_class}">{trend_arrow} {abs(forecast_change):.1f}% vs last month</span></div>
    </div>""", unsafe_allow_html=True)

with c4:
    acc_color = "kpi-up" if mape < 10 else "kpi-down"
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Forecast Accuracy</div>
        <div class="kpi-value">{100-mape:.1f}%</div>
        <div class="kpi-sub"><span class="{acc_color}">MAPE: {mape:.1f}%</span></div>
    </div>""", unsafe_allow_html=True)

with c5:
    rr_class = "kpi-down" if return_rate > 8 else "kpi-up"
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Return Rate</div>
        <div class="kpi-value">{return_rate:.1f}%</div>
        <div class="kpi-sub"><span class="{rr_class}">{total_returned:,} units returned</span></div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# MAIN TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📈 Forecast", "💡 Insights", "📊 Analytics", "📋 Data"])

with tab1:
    st.markdown(f'<div class="section-header">{selected_product} — Sales Forecast</div>', unsafe_allow_html=True)
    st.markdown(f"<p style='color:#64748b;font-size:0.85rem;'>Historical data + {forecast_periods}-month forecast with confidence intervals</p>", unsafe_allow_html=True)

    fig_forecast = plot_forecast(df, forecast_df, y_pred, selected_product)
    st.plotly_chart(fig_forecast, use_container_width=True)

    # Forecast table
    st.markdown("#### Forecast Details")
    fc_display = forecast_df.copy()
    fc_display["date"] = fc_display["date"].dt.strftime("%b %Y")
    fc_display.columns = ["Month", "Forecasted Units", "Upper Bound", "Lower Bound"]
    fc_display["Forecasted Units"] = fc_display["Forecasted Units"].apply(lambda x: f"{x:,}")
    fc_display["Upper Bound"] = fc_display["Upper Bound"].apply(lambda x: f"{x:,}")
    fc_display["Lower Bound"] = fc_display["Lower Bound"].apply(lambda x: f"{x:,}")
    st.dataframe(fc_display, use_container_width=True, hide_index=True)

    if show_formula:
        st.markdown("#### Forecast Model Formula")
        st.code(f"""
# Polynomial Regression with Seasonal Components (degree={degree})
# Features: time_index, sin(2π·month/12), cos(2π·month/12)

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import numpy as np

# Encode seasonality
month_sin = np.sin(2 * np.pi * month / 12)
month_cos = np.cos(2 * np.pi * month / 12)

X = [[time_index, month_sin, month_cos]]  # for each period
poly = PolynomialFeatures(degree={degree})
X_poly = poly.fit_transform(X)
model = LinearRegression().fit(X_poly, y)
forecast = model.predict(poly.transform(X_future))

# Model Performance
# MAPE: {mape:.2f}%
# Accuracy: {100-mape:.2f}%
        """, language="python")

with tab2:
    st.markdown('<div class="section-header">✨ Gemini AI Insights</div>', unsafe_allow_html=True)
    st.markdown(f"<p style='color:#64748b;font-size:0.85rem;'>Powered by Google Gemini 2.0 Flash — real-time analysis of <b>{selected_product}</b></p>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # ── Gemini call with spinner ──────────────────────────────
    with st.spinner("🤖 Gemini is analysing your sales data..."):
        gemini_result = generate_gemini_insights(
            df.to_json(),
            forecast_df.to_json(),
            mape,
            selected_product
        )

    # ── RENDER GEMINI RESPONSE ────────────────────────────────
    if gemini_result:

        # Risk badge
        risk = gemini_result.get("risk_level", "medium")
        risk_reason = gemini_result.get("risk_reason", "")
        risk_badge = {"low": "badge-green", "medium": "badge-amber", "high": "badge-red"}.get(risk, "badge-amber")
        risk_label = {"low": "🟢 Low Risk", "medium": "🟡 Medium Risk", "high": "🔴 High Risk"}.get(risk, "Medium Risk")

        # Executive summary
        summary = gemini_result.get("summary", "")
        st.markdown(f"""
        <div style='background:linear-gradient(135deg,#f8faff,#eef2ff);border:1px solid #c7d2fe;
                    border-radius:14px;padding:1.2rem 1.4rem;margin-bottom:1.4rem;'>
            <div style='display:flex;align-items:center;gap:0.8rem;margin-bottom:0.6rem;'>
                <span style='font-size:1.1rem;font-weight:700;color:#3730a3;'>Executive Summary</span>
                <span class="badge {risk_badge}">{risk_label}</span>
            </div>
            <div style='font-size:0.88rem;color:#374151;line-height:1.6;'>{summary}</div>
            <div style='font-size:0.78rem;color:#6b7280;margin-top:0.5rem;font-style:italic;'>{risk_reason}</div>
        </div>
        """, unsafe_allow_html=True)

        # ── Insights ─────────────────────────────────────────
        st.markdown("#### 📊 Performance Insights")
        insights = gemini_result.get("insights", [])
        for ins in insights:
            itype = ins.get("type", "info")
            type_class = f"insight-box {itype}" if itype in ["success","warning","danger"] else "insight-box"
            st.markdown(f"""
            <div class="{type_class}">
                <div class="insight-title">{ins.get('title','')}</div>
                <div class="insight-text">{ins.get('text','')}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>")

        # ── Why not selling ──────────────────────────────────
        st.markdown("#### ❓ Why This Product May Not Be Selling")
        why_list = gemini_result.get("why_not_selling", [])
        wc1, wc2 = st.columns(2)
        for i, w in enumerate(why_list):
            wtype = w.get("type", "warning")
            type_class = f"insight-box {wtype}" if wtype in ["success","warning","danger"] else "insight-box"
            html = f"""
            <div class="{type_class}">
                <div class="insight-title">{w.get('title','')}</div>
                <div class="insight-text">{w.get('text','')}</div>
            </div>"""
            if i % 2 == 0:
                wc1.markdown(html, unsafe_allow_html=True)
            else:
                wc2.markdown(html, unsafe_allow_html=True)

        st.markdown("<br>")

        # ── Recommendations ──────────────────────────────────
        st.markdown("#### 🎯 Gemini's Recommendations")
        recs = gemini_result.get("recommendations", [])
        for i, rec in enumerate(recs, 1):
            st.markdown(f"""
            <div style='display:flex;gap:0.8rem;align-items:flex-start;
                        background:white;border:1px solid #e2e8f0;border-radius:10px;
                        padding:0.9rem 1.1rem;margin-bottom:0.6rem;'>
                <span style='background:#6366f1;color:white;border-radius:50%;
                             width:24px;height:24px;display:flex;align-items:center;
                             justify-content:center;font-size:0.75rem;font-weight:700;
                             flex-shrink:0;'>{i}</span>
                <span style='font-size:0.85rem;color:#374151;line-height:1.5;'>{rec}</span>
            </div>
            """, unsafe_allow_html=True)

        # Regenerate button
        st.markdown("<br>")
        if st.button("🔄 Regenerate Insights", help="Clear cache and ask Gemini again"):
            generate_gemini_insights.clear()
            st.rerun()

    else:
        # ── Fallback if Gemini fails ─────────────────────────
        st.warning("⚠️ Gemini API is currently unreachable. Showing built-in insights instead.")
        total_units = df["units_sold"].sum()
        summary, fb_insights, fb_why, fb_recs, fb_risk, fb_risk_reason = generate_insights_fallback(
            df, forecast_df, mape, selected_product
        )
        st.markdown(f"**Summary:** {summary}")
        for itype, title, text in fb_insights:
            type_class = f"insight-box {itype}" if itype in ["success","warning","danger"] else "insight-box"
            st.markdown(f'<div class="{type_class}"><div class="insight-title">{title}</div>'
                        f'<div class="insight-text">{text}</div></div>', unsafe_allow_html=True)
        st.markdown("#### Why Products May Not Sell")
        c1f, c2f = st.columns(2)
        for i, (wtype, wtitle, wtext) in enumerate(fb_why):
            type_class = f"insight-box {wtype}" if wtype in ["success","warning","danger"] else "insight-box"
            html = f'<div class="{type_class}"><div class="insight-title">{wtitle}</div><div class="insight-text">{wtext}</div></div>'
            (c1f if i % 2 == 0 else c2f).markdown(html, unsafe_allow_html=True)

with tab3:
    st.markdown('<div class="section-header">Analytics Deep Dive</div>', unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        fig_rev = plot_revenue_waterfall(df)
        st.plotly_chart(fig_rev, use_container_width=True)

    with col_b:
        fig_ret = plot_return_rate(df)
        if fig_ret:
            st.plotly_chart(fig_ret, use_container_width=True)
        else:
            st.info("No return data available in your dataset.")

    try:
        if df["date"].dt.year.nunique() >= 2:
            fig_heat = plot_monthly_heatmap(df)
            st.plotly_chart(fig_heat, use_container_width=True)
    except:
        pass

    # YoY comparison
    df["year"] = df["date"].dt.year
    df["month_num"] = df["date"].dt.month
    years = sorted(df["year"].unique())
    if len(years) >= 2:
        st.markdown("#### Year-over-Year Comparison")
        fig_yoy = go.Figure()
        colors = ["#6366f1", "#10b981", "#f59e0b", "#ef4444"]
        for i, year in enumerate(years):
            yr_data = df[df["year"] == year].sort_values("month_num")
            fig_yoy.add_trace(go.Scatter(
                x=yr_data["month_num"], y=yr_data["units_sold"],
                mode="lines+markers", name=str(year),
                line=dict(color=colors[i % len(colors)], width=2),
                marker=dict(size=6)
            ))
        fig_yoy.update_layout(
            plot_bgcolor="white", paper_bgcolor="white",
            height=300, margin=dict(l=0, r=0, t=20, b=0),
            xaxis=dict(tickvals=list(range(1,13)),
                       ticktext=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"],
                       showgrid=False),
            yaxis=dict(gridcolor="#f1f5f9", title="Units Sold"),
            legend=dict(orientation="h", y=1.1)
        )
        st.plotly_chart(fig_yoy, use_container_width=True)

with tab4:
    st.markdown('<div class="section-header">Raw Data</div>', unsafe_allow_html=True)
    st.markdown(f"**{len(df)} records** loaded for **{selected_product}**")

    display_df = df.copy()
    display_df["date"] = display_df["date"].dt.strftime("%b %Y")
    if "revenue" in display_df.columns:
        display_df["revenue"] = display_df["revenue"].apply(lambda x: f"${x:,.2f}")
    if "units_sold" in display_df.columns:
        display_df["units_sold"] = display_df["units_sold"].apply(lambda x: f"{x:,}")
    if "units_returned" in display_df.columns:
        display_df["units_returned"] = display_df["units_returned"].apply(lambda x: f"{x:,}")

    st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.markdown("#### Sample CSV Format")
    sample_csv = """date,product,units_sold,revenue,units_returned
2024-01-01,My Product,5420,162600.00,271
2024-02-01,My Product,4980,149400.00,249
2024-03-01,My Product,6100,183000.00,305
2024-04-01,My Product,6850,205500.00,342"""
    st.code(sample_csv, language="csv")

    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        st.download_button("⬇️ Download Sales History",
                           df.to_csv(index=False),
                           f"{selected_product.replace(' ','_')}_history.csv", "text/csv")
    with col_dl2:
        st.download_button("⬇️ Download Forecast",
                           forecast_df.to_csv(index=False),
                           f"{selected_product.replace(' ','_')}_forecast.csv", "text/csv")


# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center;color:#94a3b8;font-size:0.78rem;padding:1rem 0;'>
    <b>SalesLens</b> · AI Sales Forecast Intelligence · 
    Data is processed locally — nothing leaves your machine.
</div>
""", unsafe_allow_html=True)
