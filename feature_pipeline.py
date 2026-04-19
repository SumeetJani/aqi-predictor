import requests
import pandas as pd
from datetime import datetime
import json

# ---- YOUR API TOKEN ----
API_TOKEN = "7c147697499565401cafd50954f445a6582e0396"

# ---- CITY TO MONITOR ----
CITY = "karachi"

def fetch_aqi_data():
    """Fetch live AQI data from AQICN API"""
    url = f"https://api.waqi.info/feed/{CITY}/?token={API_TOKEN}"
    
    print(f"Fetching AQI data for {CITY.upper()}...")
    response = requests.get(url)
    data = response.json()
    
    if data["status"] != "ok":
        print("Error fetching data:", data)
        return None
    
    return data["data"]

def compute_features(raw):
    """Extract and compute features from raw API data"""
    now = datetime.now()
    
    # --- Pollutant readings (iaqi = individual air quality index) ---
    iaqi = raw.get("iaqi", {})
    
    pm25  = iaqi.get("pm25",  {}).get("v", None)
    pm10  = iaqi.get("pm10",  {}).get("v", None)
    o3    = iaqi.get("o3",    {}).get("v", None)
    no2   = iaqi.get("no2",   {}).get("v", None)
    so2   = iaqi.get("so2",   {}).get("v", None)
    co    = iaqi.get("co",    {}).get("v", None)
    temp  = iaqi.get("t",     {}).get("v", None)
    humidity = iaqi.get("h",  {}).get("v", None)
    wind  = iaqi.get("w",     {}).get("v", None)
    
    # --- Main AQI value ---
    aqi = raw.get("aqi", None)
    
    # --- AQI Category (what it means in plain English) ---
    def aqi_category(val):
        if val is None: return "Unknown"
        val = int(val)
        if val <= 50:  return "Good"
        if val <= 100: return "Moderate"
        if val <= 150: return "Unhealthy for Sensitive Groups"
        if val <= 200: return "Unhealthy"
        if val <= 300: return "Very Unhealthy"
        return "Hazardous"
    
    # --- Time-based features ---
    features = {
        "timestamp":    now.strftime("%Y-%m-%d %H:%M:%S"),
        "city":         CITY,
        "aqi":          aqi,
        "aqi_category": aqi_category(aqi),
        "pm25":         pm25,
        "pm10":         pm10,
        "o3":           o3,
        "no2":          no2,
        "so2":          so2,
        "co":           co,
        "temperature":  temp,
        "humidity":     humidity,
        "wind_speed":   wind,
        # Time-based features for ML model
        "hour":         now.hour,
        "day_of_week":  now.weekday(),   # 0=Monday, 6=Sunday
        "month":        now.month,
        "is_weekend":   1 if now.weekday() >= 5 else 0,
    }
    
    return features

def save_to_csv(features):
    """Save features to a local CSV file (our simple feature store for now)"""
    filename = "aqi_features.csv"
    df_new = pd.DataFrame([features])
    
    try:
        # If file already exists, append to it
        df_existing = pd.read_csv(filename)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined.to_csv(filename, index=False)
        print(f"Appended to existing file. Total rows: {len(df_combined)}")
    except FileNotFoundError:
        # First time — create the file
        df_new.to_csv(filename, index=False)
        print(f"Created new file: {filename}")

def display_summary(features):
    """Print a nice readable summary"""
    print("\n" + "="*50)
    print(f"  AQI REPORT — {features['city'].upper()}")
    print("="*50)
    print(f"  Time:        {features['timestamp']}")
    print(f"  AQI:         {features['aqi']} ({features['aqi_category']})")
    print(f"  PM2.5:       {features['pm25']}")
    print(f"  PM10:        {features['pm10']}")
    print(f"  Ozone (O3):  {features['o3']}")
    print(f"  NO2:         {features['no2']}")
    print(f"  Temperature: {features['temperature']} °C")
    print(f"  Humidity:    {features['humidity']} %")
    print("="*50)
    
    # Alert if air quality is bad
    aqi_val = features.get("aqi")
    if aqi_val and int(aqi_val) > 150:
        print("  ⚠️  WARNING: Air quality is UNHEALTHY today!")
    print()

def main():
    # Step 1: Fetch raw data
    raw = fetch_aqi_data()
    if raw is None:
        return
    
    # Step 2: Compute features
    features = compute_features(raw)
    
    # Step 3: Display summary
    display_summary(features)
    
    # Step 4: Save to CSV
    save_to_csv(features)
    print(f"Data saved! Check 'aqi_features.csv' in your project folder.")

if __name__ == "__main__":
    main()