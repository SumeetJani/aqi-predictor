# AQI Predictor — Final Project Report

**Student:** Sumeet Jani
**University:** DHA Suffa University, Karachi
**Internship:** Pearls Virtual Internship
**Submission Date:** April 2026

---

## 1. Project Overview

This project builds a fully automated, serverless system that predicts the Air Quality Index (AQI) for any city in the world for the next 3 days. The system collects live pollution data every hour, trains a machine learning model daily, and displays forecasts on an interactive web dashboard accessible to anyone via a public link.

**Live Dashboard:** https://aqi-predictor-rv8nqd6ybmmwhbv7gmda7j.streamlit.app/

---

## 2. Problem Statement

Air pollution is a serious public health issue in cities like Karachi and across the world. Residents have no easy way to know how air quality will change over the coming days. This system solves that by providing an automated 3-day AQI forecast using real pollution data and machine learning — for any city in the world, not just one location.

---

## 3. System Architecture

The system has 4 main components that work together automatically:

| Component | Technology | Purpose |
|---|---|---|
| Feature Pipeline | Python + AQICN API | Fetches live AQI data every hour |
| Training Pipeline | scikit-learn + XGBoost | Trains ML model daily |
| Automation | GitHub Actions (CI/CD) | Runs pipelines on a schedule, zero manual effort |
| Web Dashboard | Streamlit | Displays live + forecast AQI for any city |

---

## 4. Data Collection

- **Source:** AQICN API (aqicn.org) — covers thousands of cities worldwide
- **Primary city:** Karachi, Pakistan (used for model training)
- **Pollutants collected:** PM2.5, temperature, humidity, wind speed
- **Historical data:** 90 days of backfilled data
- **Update frequency:** Every 1 hour via automated GitHub Actions pipeline

### Features Engineered

The raw API data was transformed into 14 model features:

| Feature | Type | Description |
|---|---|---|
| aqi | Raw | Current Air Quality Index value |
| pm25 | Raw | Fine particulate matter reading |
| temperature | Raw | Ambient temperature in °C |
| humidity | Raw | Relative humidity percentage |
| wind_speed | Raw | Wind speed in m/s |
| hour | Time-based | Hour of the day (0–23) |
| day_of_week | Time-based | Day of week (0=Monday, 6=Sunday) |
| month | Time-based | Month of year (1–12) |
| is_weekend | Time-based | 1 if Saturday/Sunday, else 0 |
| day_of_year | Time-based | Day number in the year (1–365) |
| aqi_change | Derived | AQI difference from previous day |
| aqi_change_pct | Derived | AQI percentage change from previous day |
| aqi_rolling_3d | Derived | 3-day rolling average of AQI |
| aqi_rolling_7d | Derived | 7-day rolling average of AQI |

**Target variable:** next_day_aqi — the AQI value the model learns to predict

---

## 5. Exploratory Data Analysis (EDA)

Key findings from the 90-day historical Karachi dataset:

- Temperature and day of year are the strongest correlates with AQI change
- AQI tends to be higher in winter months due to reduced wind dispersion
- The 3-day rolling average (aqi_rolling_3d) is a strong predictor — AQI rarely changes drastically from one day to the next
- Weekend vs weekday difference exists but is relatively small compared to seasonal effects
- PM2.5 is the primary pollutant reported by the Karachi AQICN station

---

## 6. Model Training

### Models Tested

| Model | RMSE | MAE | R² |
|---|---|---|---|
| Ridge Regression | 17.36 | 13.59 | 0.012 |
| Random Forest | 16.74 | 13.37 | 0.081 |
| Gradient Boosting | 20.22 | 15.51 | -0.341 |
| XGBoost (tuned) | 18.89 | 13.97 | -0.171 |

### Best Model
**Random Forest** achieved the lowest RMSE of **16.74**, meaning predictions are on average 16.74 AQI points away from the actual value. This is acceptable given only 90 days of training data — performance will improve significantly as the automated pipeline collects more data over time.

### XGBoost Hyperparameter Tuning
XGBoost was tuned using GridSearchCV with the following best parameters:
- Learning rate: 0.05
- Max depth: 3
- N estimators: 100
- Subsample: 0.8

Despite tuning, Random Forest outperformed XGBoost on this dataset size, which is common when training data is limited.

---

## 7. Feature Importance (SHAP Analysis)

SHAP (SHapley Additive exPlanations) was used to explain which features drive predictions the most:

| Rank | Feature | SHAP Value | Meaning |
|---|---|---|---|
| 1 | temperature | 3.103 | Seasonal temperature strongly predicts AQI |
| 2 | day_of_year | 3.041 | Seasonal patterns drive pollution levels |
| 3 | aqi_rolling_3d | 2.796 | Recent AQI trend is a strong predictor |
| 4 | aqi | 2.304 | Current AQI influences next day's value |
| 5 | day_of_week | 1.462 | Weekly patterns affect pollution |
| 6 | humidity | 1.152 | Humidity affects particulate dispersion |
| 7 | aqi_change | 1.079 | Rate of change matters for forecasting |
| 8 | wind_speed | 0.894 | Wind disperses pollutants |

**Key insight:** Temperature and seasonal patterns are the biggest drivers of AQI in Karachi. This makes physical sense — cooler winter months trap pollutants closer to the ground due to thermal inversion, leading to higher AQI values.

---

## 8. Automation Pipeline (CI/CD)

Two GitHub Actions workflows were created to fully automate the system:

**feature_pipeline.yml** — runs every hour
- Checks out the code on a fresh Ubuntu server
- Installs Python dependencies
- Runs `feature_pipeline.py` to fetch latest AQI data
- Commits and pushes the updated CSV back to GitHub automatically

**training_pipeline.yml** — runs every day at 2AM UTC
- Runs backfill to refresh historical data
- Retrains the model with the latest data
- Saves the updated model back to GitHub

This means the system fully updates itself with zero manual effort from the developer.

---

## 9. Web Dashboard

Built using Streamlit and deployed free on Streamlit Cloud.

**Live URL:** https://aqi-predictor-rv8nqd6ybmmwhbv7gmda7j.streamlit.app/

### Dashboard Features

- **Global city search** — search any city in the world by name
- **Quick select buttons** — Karachi, Lahore, Delhi, Dubai, London, Beijing, New York, Paris
- **Live AQI metrics** — current AQI, PM2.5, temperature, humidity, wind speed
- **Smart alert banners** — 5 levels from Good to Hazardous with actionable advice
- **3-day forecast cards** — color coded by AQI category
- **Forecast line chart** — visual trend from today to Day 3
- **AQI scale guide** — color reference from 0 to 300+
- **Pollutant breakdown chart** — bar chart of all available pollutants
- **Model performance metrics** — RMSE, MAE, R² displayed live
- **SHAP feature importance chart** — explains what drives each prediction

---

## 10. Results & Conclusion

This project successfully delivers a complete, production-grade ML system:

- A live data pipeline collecting AQI data every hour for any city worldwide
- A machine learning model (Random Forest) predicting next-day AQI
- A fully automated CI/CD pipeline requiring zero manual intervention
- An interactive public dashboard accessible to anyone in the world

The system is designed to improve over time — as GitHub Actions continues collecting hourly data, the training dataset grows larger and model accuracy will increase. The architecture is fully serverless and runs at zero cost using free tiers of GitHub Actions, AQICN API, and Streamlit Cloud.

---

## 11. References

- AQICN API: https://aqicn.org/api/
- scikit-learn: https://scikit-learn.org/
- XGBoost: https://xgboost.readthedocs.io/
- SHAP: https://shap.readthedocs.io/
- Streamlit: https://streamlit.io/
- GitHub Actions: https://docs.github.com/en/actions
- WHO Air Quality Guidelines: https://www.who.int/news-room/fact-sheets/detail/ambient-(outdoor)-air-quality-and-health