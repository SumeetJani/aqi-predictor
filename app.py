import streamlit as st
import pandas as pd
import numpy as np
import pickle
import requests
from datetime import datetime, timedelta

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Global AQI Predictor",
    page_icon="🌍",
    layout="wide"
)

API_TOKEN = "7c147697499565401cafd50954f445a6582e0396"

# ── AQI color and label ──────────────────────────────────────
def aqi_info(val):
    if val is None:
        return "Unknown", "#888888"
    val = int(val)
    if val <= 50:   return "Good",                           "#00c853"
    if val <= 100:  return "Moderate",                       "#ffd600"
    if val <= 150:  return "Unhealthy for Sensitive Groups",  "#ff6d00"
    if val <= 200:  return "Unhealthy",                      "#dd2c00"
    if val <= 300:  return "Very Unhealthy",                 "#6a1b9a"
    return                  "Hazardous",                     "#880e4f"

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

# ── Fetch AQI for any city ───────────────────────────────────
@st.cache_data(ttl=3600)
def fetch_city(city):
    try:
        url  = f"https://api.waqi.info/feed/{city}/?token={API_TOKEN}"
        resp = requests.get(url, timeout=10).json()
        if resp["status"] == "ok":
            return resp["data"], None
        return None, resp.get("data", "City not found")
    except Exception as e:
        return None, str(e)

# ── Build feature row for prediction ─────────────────────────
def build_feature_row(raw, target_date, feature_cols):
    iaqi = raw.get("iaqi", {})
    def g(k): return iaqi.get(k, {}).get("v", np.nan)
    all_vals = {
        "aqi":            raw.get("aqi", np.nan),
        "pm25":           g("pm25"), "pm10": g("pm10"),
        "o3":             g("o3"),   "no2":  g("no2"),
        "so2":            g("so2"),  "co":   g("co"),
        "temperature":    g("t"),    "humidity": g("h"),
        "wind_speed":     g("w"),
        "hour":           target_date.hour,
        "day_of_week":    target_date.weekday(),
        "month":          target_date.month,
        "is_weekend":     1 if target_date.weekday() >= 5 else 0,
        "day_of_year":    target_date.timetuple().tm_yday,
        "aqi_change":     0, "aqi_change_pct": 0,
        "aqi_rolling_3d": raw.get("aqi", np.nan),
        "aqi_rolling_7d": raw.get("aqi", np.nan),
    }
    row = {col: all_vals.get(col, np.nan) for col in feature_cols}
    df  = pd.DataFrame([row]).fillna(0)
    return df[feature_cols]

# ════════════════════════════════════════════════════════════
#  MAIN UI
# ════════════════════════════════════════════════════════════

# ── HEADER ───────────────────────────────────────────────────
st.title("🌍 Global AQI Predictor")
st.caption("Check air quality and get a 3-day forecast for any city in the world")

# ── CITY SEARCH BOX ──────────────────────────────────────────
st.markdown("---")
col_search, col_btn = st.columns([4, 1])

with col_search:
    city_input = st.text_input(
        label="Enter any city",
        value="karachi",
        placeholder="e.g. london, delhi, new-york, dubai, beijing...",
        label_visibility="collapsed"
    )

with col_btn:
    search_clicked = st.button("🔍 Search", use_container_width=True)

# Popular cities quick select
st.markdown("**Quick select:**")
quick_cols = st.columns(8)
quick_cities = ["Karachi", "Lahore", "Delhi", "Dubai", "London", "Beijing", "New York", "Paris"]
selected_quick = None
for i, city in enumerate(quick_cities):
    with quick_cols[i]:
        if st.button(city, use_container_width=True):
            selected_quick = city.lower().replace(" ", "-")

# Determine which city to show
city = selected_quick if selected_quick else city_input.strip().lower().replace(" ", "-")

st.markdown("---")

# ── FETCH DATA ───────────────────────────────────────────────
if not city:
    st.info("Enter a city name above to get started.")
    st.stop()

with st.spinner(f"Fetching AQI data for {city.title()}..."):
    raw, err = fetch_city(city)

if raw is None:
    st.error(f"Could not find AQI data for **{city.title()}**. Try a different spelling — e.g. 'new-york', 'los-angeles', 'hong-kong'")
    st.stop()

# ── CITY HEADER ──────────────────────────────────────────────
live_aqi      = raw.get("aqi", None)
label, color  = aqi_info(live_aqi)
iaqi          = raw.get("iaqi", {})
station_name  = raw.get("city", {}).get("name", city.title())
updated_time  = raw.get("time", {}).get("s", "N/A")

st.subheader(f"📍 {station_name}")
st.caption(f"Station last updated: {updated_time}  |  Refreshes every hour")

# ── METRIC CARDS ─────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("🌫️ AQI",         str(live_aqi), label)
c2.metric("💨 PM2.5",        str(iaqi.get("pm25", {}).get("v", "N/A")))
c3.metric("🌡️ Temperature",  str(iaqi.get("t",    {}).get("v", "N/A")) + " °C")
c4.metric("💧 Humidity",     str(iaqi.get("h",    {}).get("v", "N/A")) + " %")
c5.metric("💨 Wind",         str(iaqi.get("w",    {}).get("v", "N/A")) + " m/s")

# ── ALERT BANNER ─────────────────────────────────────────────
if live_aqi and int(live_aqi) > 300:
    st.error(f"☠️ HAZARDOUS — AQI {live_aqi}. Stay indoors, wear N95 mask if going out!")
elif live_aqi and int(live_aqi) > 200:
    st.error(f"⚠️ VERY UNHEALTHY — AQI {live_aqi}. Avoid all outdoor activity!")
elif live_aqi and int(live_aqi) > 150:
    st.warning(f"🔴 UNHEALTHY — AQI {live_aqi}. Limit time outdoors.")
elif live_aqi and int(live_aqi) > 100:
    st.warning(f"🟠 MODERATE RISK — AQI {live_aqi}. Sensitive groups should be careful.")
else:
    st.success(f"✅ Air quality is {label} in {station_name} (AQI: {live_aqi})")

st.markdown("---")

# ── 3-DAY FORECAST ───────────────────────────────────────────
st.subheader("📅 3-Day AQI Forecast")

model, feature_cols, model_info = load_model()

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
    f1, f2, f3 = st.columns(3)
    for i, (col, row) in enumerate(zip([f1, f2, f3], forecast_rows)):
        with col:
            st.markdown(f"""
            <div style='background:{row["Color"]}22;
                        border-left:5px solid {row["Color"]};
                        padding:20px; border-radius:10px;'>
                <div style='font-size:13px;color:#888;margin-bottom:4px;'>{row["Day"]}</div>
                <div style='font-size:42px;font-weight:700;color:{row["Color"]};line-height:1'>{row["AQI"]}</div>
                <div style='font-size:13px;color:#555;margin-top:6px;'>{row["Category"]}</div>
            </div>
            """, unsafe_allow_html=True)

    # Forecast chart
    st.markdown(" ")
    chart_data = pd.DataFrame({
        "Day": ["Today"] + [r["Day"] for r in forecast_rows],
        "AQI": [float(live_aqi)] + [r["AQI"] for r in forecast_rows]
    }).set_index("Day")
    st.line_chart(chart_data)

st.markdown("---")

# ── AQI SCALE GUIDE ──────────────────────────────────────────
st.subheader("🎨 AQI Scale Guide")
scale_cols = st.columns(6)
scale = [
    ("0-50",   "Good",             "#00c853"),
    ("51-100", "Moderate",         "#ffd600"),
    ("101-150","Sensitive Groups",  "#ff6d00"),
    ("151-200","Unhealthy",        "#dd2c00"),
    ("201-300","Very Unhealthy",   "#6a1b9a"),
    ("300+",   "Hazardous",        "#880e4f"),
]
for col, (rng, lbl, clr) in zip(scale_cols, scale):
    with col:
        st.markdown(f"""
        <div style='background:{clr}33; border-top:4px solid {clr};
                    padding:10px; border-radius:8px; text-align:center;'>
            <div style='font-size:16px;font-weight:700;color:{clr};'>{rng}</div>
            <div style='font-size:11px;color:#555;margin-top:4px;'>{lbl}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# ── POLLUTANT BREAKDOWN ──────────────────────────────────────
st.subheader("🧪 Pollutant Breakdown")
pollutants = {
    "PM2.5":  iaqi.get("pm25", {}).get("v"),
    "PM10":   iaqi.get("pm10", {}).get("v"),
    "Ozone":  iaqi.get("o3",   {}).get("v"),
    "NO₂":    iaqi.get("no2",  {}).get("v"),
    "SO₂":    iaqi.get("so2",  {}).get("v"),
    "CO":     iaqi.get("co",   {}).get("v"),
}
poll_df = pd.DataFrame([
    {"Pollutant": k, "Value": float(v)}
    for k, v in pollutants.items() if v is not None
])
if not poll_df.empty:
    st.bar_chart(poll_df.set_index("Pollutant"))
else:
    st.info("No pollutant breakdown available for this station.")

st.markdown("---")

# ── MODEL INFO ───────────────────────────────────────────────
if model_info:
    st.subheader("🤖 Model Performance")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Model", model_info.get("name", "N/A"))
    m2.metric("RMSE",  f"{model_info.get('rmse', 0):.2f}")
    m3.metric("MAE",   f"{model_info.get('mae',  0):.2f}")
    m4.metric("R²",    f"{model_info.get('r2',   0):.3f}")
    st.caption(f"Trained on Karachi data · Last trained: {model_info.get('trained_at', 'N/A')}")

# ── SHAP IMPORTANCE ──────────────────────────────────────────
try:
    shap_df = pd.read_csv("model/shap_importance.csv")
    st.subheader("📊 What Drives AQI Predictions? (SHAP)")
    st.bar_chart(shap_df.set_index("feature")["importance"])
    st.caption("Higher value = stronger influence on the prediction")
except FileNotFoundError:
    pass

st.markdown("---")
st.caption("🌍 Data: AQICN API  ·  Model: scikit-learn & XGBoost  ·  Built with Streamlit")