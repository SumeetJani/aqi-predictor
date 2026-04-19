import streamlit as st
import pandas as pd
import numpy as np
import pickle
import requests
from datetime import datetime, timedelta

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Karachi AQI Predictor",
    page_icon="🌫️",
    layout="wide"
)

API_TOKEN = "7c147697499565401cafd50954f445a6582e0396"
CITY      = "karachi"

# ── AQI color and label ──────────────────────────────────────
def aqi_info(val):
    if val is None:
        return "Unknown", "#888888"
    val = int(val)
    if val <= 50:   return "Good",                           "#00e400"
    if val <= 100:  return "Moderate",                       "#ffff00"
    if val <= 150:  return "Unhealthy for Sensitive Groups",  "#ff7e00"
    if val <= 200:  return "Unhealthy",                      "#ff0000"
    if val <= 300:  return "Very Unhealthy",                 "#8f3f97"
    return                  "Hazardous",                     "#7e0023"

# ── Load model ───────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        with open("model/best_model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("model/features.pkl", "rb") as f:
            features = pickle.load(f)
        with open("model/model_info.pkl", "rb") as f:
            info = pickle.load(f)
        return model, features, info
    except FileNotFoundError:
        return None, None, None

# ── Fetch live AQI ───────────────────────────────────────────
@st.cache_data(ttl=3600)
def fetch_live():
    try:
        url  = f"https://api.waqi.info/feed/{CITY}/?token={API_TOKEN}"
        resp = requests.get(url, timeout=10).json()
        if resp["status"] == "ok":
            return resp["data"]
    except:
        pass
    return None

# ── Build one feature row for prediction ─────────────────────
def build_feature_row(raw, target_date, feature_cols):
    iaqi = raw.get("iaqi", {})
    def g(k): return iaqi.get(k, {}).get("v", np.nan)

    # Build a dict with ALL possible features
    all_vals = {
        "aqi":           raw.get("aqi", np.nan),
        "pm25":          g("pm25"),
        "pm10":          g("pm10"),
        "o3":            g("o3"),
        "no2":           g("no2"),
        "so2":           g("so2"),
        "co":            g("co"),
        "temperature":   g("t"),
        "humidity":      g("h"),
        "wind_speed":    g("w"),
        "hour":          target_date.hour,
        "day_of_week":   target_date.weekday(),
        "month":         target_date.month,
        "is_weekend":    1 if target_date.weekday() >= 5 else 0,
        "day_of_year":   target_date.timetuple().tm_yday,
        "aqi_change":    0,
        "aqi_change_pct": 0,
        "aqi_rolling_3d": raw.get("aqi", np.nan),
        "aqi_rolling_7d": raw.get("aqi", np.nan),
    }

    # Only keep the features the model was trained on
    row = {col: all_vals.get(col, np.nan) for col in feature_cols}
    df  = pd.DataFrame([row])

    # Fill any NaN with 0 (safe fallback)
    df = df.fillna(0)
    return df[feature_cols]

# ════════════════════════════════════════════════════════════
#  MAIN UI
# ════════════════════════════════════════════════════════════
st.title("🌫️ Karachi AQI Predictor")
st.caption(f"Data refreshes every hour  |  Last checked: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

model, feature_cols, model_info = load_model()
raw = fetch_live()

if raw is None:
    st.error("Could not fetch live AQI data. Check your internet connection.")
    st.stop()

live_aqi        = raw.get("aqi", None)
label, color    = aqi_info(live_aqi)
iaqi            = raw.get("iaqi", {})

# ── TOP METRIC CARDS ─────────────────────────────────────────
st.markdown("---")
c1, c2, c3, c4 = st.columns(4)
c1.metric("🌍 Current AQI",   str(live_aqi),  label)
c2.metric("💨 PM2.5",         str(iaqi.get("pm25", {}).get("v", "N/A")))
c3.metric("🌡️ Temperature",   str(iaqi.get("t",   {}).get("v", "N/A")) + " °C")
c4.metric("💧 Humidity",      str(iaqi.get("h",   {}).get("v", "N/A")) + " %")

# ── ALERT BANNER ─────────────────────────────────────────────
if live_aqi and int(live_aqi) > 150:
    st.error(f"⚠️ HAZARD ALERT — AQI is {live_aqi} ({label}). Avoid all outdoor activity!")
elif live_aqi and int(live_aqi) > 100:
    st.warning(f"⚡ CAUTION — AQI is {live_aqi} ({label}). Sensitive groups stay indoors.")
else:
    st.success(f"✅ Air quality is {label} today (AQI: {live_aqi})")

st.markdown("---")

# ── 3-DAY FORECAST ───────────────────────────────────────────
st.subheader("📅 3-Day AQI Forecast")

if model is None:
    st.info("Model not found. Make sure model/best_model.pkl exists.")
else:
    forecast_rows = []
    for day_offset in range(1, 4):
        target_date = datetime.now() + timedelta(days=day_offset)
        feat_df     = build_feature_row(raw, target_date, feature_cols)
        predicted   = float(model.predict(feat_df)[0])
        predicted   = max(0, round(predicted, 1))
        lbl, clr    = aqi_info(predicted)
        forecast_rows.append({
            "Day":      target_date.strftime("%A, %b %d"),
            "AQI":      predicted,
            "Category": lbl,
            "Color":    clr,
        })

    # Forecast cards
    cols = st.columns(3)
    for i, row in enumerate(forecast_rows):
        with cols[i]:
            st.markdown(f"""
            <div style='background:{row["Color"]}22;
                        border-left:4px solid {row["Color"]};
                        padding:16px; border-radius:8px;'>
                <div style='font-size:13px;color:#666;'>{row["Day"]}</div>
                <div style='font-size:38px;font-weight:700;color:{row["Color"]};'>{row["AQI"]}</div>
                <div style='font-size:13px;color:#444;'>{row["Category"]}</div>
            </div>
            """, unsafe_allow_html=True)

    # Forecast line chart
    st.markdown(" ")
    chart_data = pd.DataFrame({
        "Day": ["Today"] + [r["Day"] for r in forecast_rows],
        "AQI": [float(live_aqi)] + [r["AQI"] for r in forecast_rows]
    }).set_index("Day")
    st.line_chart(chart_data)

st.markdown("---")

# ── POLLUTANT BREAKDOWN ──────────────────────────────────────
st.subheader("🧪 Current Pollutant Levels")
pollutants = {
    "PM2.5":  iaqi.get("pm25", {}).get("v"),
    "PM10":   iaqi.get("pm10", {}).get("v"),
    "Ozone":  iaqi.get("o3",   {}).get("v"),
    "NO₂":    iaqi.get("no2",  {}).get("v"),
    "SO₂":    iaqi.get("so2",  {}).get("v"),
    "CO":     iaqi.get("co",   {}).get("v"),
}
poll_df = pd.DataFrame([
    {"Pollutant": k, "Value": v}
    for k, v in pollutants.items() if v is not None
])
if not poll_df.empty:
    st.bar_chart(poll_df.set_index("Pollutant"))
else:
    st.info("No pollutant breakdown available for this station.")

st.markdown("---")

# ── HISTORICAL TREND ─────────────────────────────────────────
st.subheader("📈 Historical AQI — Last 30 Days")
try:
    hist_df = pd.read_csv("aqi_historical.csv")
    hist_df = hist_df.tail(30)[["date", "aqi"]].dropna()
    hist_df = hist_df.set_index("date")
    st.line_chart(hist_df)
except FileNotFoundError:
    st.info("Run backfill_pipeline.py to see historical data here.")

st.markdown("---")

# ── MODEL INFO ───────────────────────────────────────────────
if model_info:
    st.subheader("🤖 Model Performance")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Model",  model_info.get("name", "N/A"))
    m2.metric("RMSE",   f"{model_info.get('rmse', 0):.2f}")
    m3.metric("MAE",    f"{model_info.get('mae',  0):.2f}")
    m4.metric("R²",     f"{model_info.get('r2',   0):.3f}")
    st.caption(f"Last trained: {model_info.get('trained_at', 'N/A')}")

# ── SHAP FEATURE IMPORTANCE ──────────────────────────────────
try:
    shap_df = pd.read_csv("model/shap_importance.csv")
    st.subheader("📊 What Drives AQI? (SHAP Feature Importance)")
    st.bar_chart(shap_df.set_index("feature")["importance"])
    st.caption("Higher value = this feature has more influence on the prediction")
except FileNotFoundError:
    pass

st.markdown("---")
st.caption("Built with Streamlit · Data: AQICN API · Models: scikit-learn & XGBoost")
