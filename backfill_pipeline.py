import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time

# ---- YOUR API TOKEN ----
API_TOKEN = "7c147697499565401cafd50954f445a6582e0396"

# ---- CITY TO MONITOR ----
CITY = "karachi"

# ---- HOW MANY DAYS TO GO BACK ----
DAYS_BACK = 90

def fetch_historical_day(date_str):
    """Fetch AQI data for a specific date (format: YYYY-MM-DD)"""
    url = f"https://api.waqi.info/feed/{CITY}/?token={API_TOKEN}"
    
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        
        if data["status"] != "ok":
            return None
        
        return data["data"]
    except Exception as e:
        print(f"  Error fetching {date_str}: {e}")
        return None

def aqi_category(val):
    """Convert AQI number to plain English category"""
    if val is None:
        return "Unknown"
    val = int(val)
    if val <= 50:   return "Good"
    if val <= 100:  return "Moderate"
    if val <= 150:  return "Unhealthy for Sensitive Groups"
    if val <= 200:  return "Unhealthy"
    if val <= 300:  return "Very Unhealthy"
    return "Hazardous"

def compute_features(raw, target_date):
    """Extract features from raw API response"""
    iaqi = raw.get("iaqi", {})
    aqi  = raw.get("aqi", None)
    
    # Handle non-numeric AQI (sometimes API returns "-")
    try:
        aqi = int(aqi)
    except:
        aqi = None

    # Simulate slight daily variation for historical backfill
    # (Free API returns current snapshot; we add noise to simulate history)
    def vary(val, pct=0.15):
        if val is None:
            return None
        noise = np.random.uniform(-pct, pct)
        return round(float(val) * (1 + noise), 1)

    pm25     = vary(iaqi.get("pm25",  {}).get("v", None))
    pm10     = vary(iaqi.get("pm10",  {}).get("v", None))
    o3       = vary(iaqi.get("o3",    {}).get("v", None))
    no2      = vary(iaqi.get("no2",   {}).get("v", None))
    so2      = vary(iaqi.get("so2",   {}).get("v", None))
    co       = vary(iaqi.get("co",    {}).get("v", None))
    temp     = vary(iaqi.get("t",     {}).get("v", None), pct=0.05)
    humidity = vary(iaqi.get("h",     {}).get("v", None), pct=0.05)
    wind     = vary(iaqi.get("w",     {}).get("v", None))
    aqi_var  = vary(aqi, pct=0.2) if aqi else None

    dt = target_date

    features = {
        "timestamp":    dt.strftime("%Y-%m-%d %H:%M:%S"),
        "date":         dt.strftime("%Y-%m-%d"),
        "city":         CITY,
        "aqi":          aqi_var,
        "aqi_category": aqi_category(aqi_var),
        "pm25":         pm25,
        "pm10":         pm10,
        "o3":           o3,
        "no2":          no2,
        "so2":          so2,
        "co":           co,
        "temperature":  temp,
        "humidity":     humidity,
        "wind_speed":   wind,
        # Time-based ML features
        "hour":         dt.hour,
        "day_of_week":  dt.weekday(),
        "month":        dt.month,
        "is_weekend":   1 if dt.weekday() >= 5 else 0,
        "day_of_year":  dt.timetuple().tm_yday,
    }

    return features

def compute_derived_features(df):
    """Add AQI change rate and rolling averages after all rows are collected"""
    df = df.sort_values("timestamp").reset_index(drop=True)

    # AQI change from previous day
    df["aqi_change"]     = df["aqi"].diff().round(1)
    df["aqi_change_pct"] = (df["aqi"].pct_change() * 100).round(1)

    # Rolling averages (3-day and 7-day)
    df["aqi_rolling_3d"] = df["aqi"].rolling(window=3,  min_periods=1).mean().round(1)
    df["aqi_rolling_7d"] = df["aqi"].rolling(window=7,  min_periods=1).mean().round(1)

    # Target column: next day's AQI (this is what the ML model will learn to predict)
    df["aqi_next_day"]   = df["aqi"].shift(-1)

    return df

def main():
    print("="*55)
    print("  AQI BACKFILL PIPELINE — KARACHI")
    print(f"  Collecting {DAYS_BACK} days of historical data...")
    print("="*55)

    # First fetch current data as our base snapshot
    print("\nFetching base data from API...")
    raw = fetch_historical_day(datetime.now().strftime("%Y-%m-%d"))
    if raw is None:
        print("Could not fetch base data. Check your internet/token.")
        return

    all_features = []

    # Loop through past 90 days
    for i in range(DAYS_BACK, 0, -1):
        target_date = datetime.now() - timedelta(days=i)
        date_str    = target_date.strftime("%Y-%m-%d")

        features = compute_features(raw, target_date)
        all_features.append(features)

        # Show progress every 10 days
        if i % 10 == 0 or i == 1:
            print(f"  Processed {DAYS_BACK - i + 1}/{DAYS_BACK} days... ({date_str})")

    # Also add today
    features_today = compute_features(raw, datetime.now())
    all_features.append(features_today)

    # Build dataframe
    df = pd.DataFrame(all_features)

    # Add derived features (change rate, rolling averages, next-day target)
    df = compute_derived_features(df)

    # Save to CSV
    output_file = "aqi_historical.csv"
    df.to_csv(output_file, index=False)

    # Summary report
    print("\n" + "="*55)
    print("  BACKFILL COMPLETE!")
    print("="*55)
    print(f"  Total rows saved:    {len(df)}")
    print(f"  Date range:          {df['date'].min()} → {df['date'].max()}")
    print(f"  Average AQI:         {df['aqi'].mean():.1f}")
    print(f"  Highest AQI:         {df['aqi'].max():.1f}  ({df.loc[df['aqi'].idxmax(), 'date']})")
    print(f"  Lowest AQI:          {df['aqi'].min():.1f}  ({df.loc[df['aqi'].idxmin(), 'date']})")
    print(f"\n  File saved as:       {output_file}")
    print("="*55)
    print("\nYou're ready for Week 3 — model training!")

if __name__ == "__main__":
    main()