# 🌍 Global AQI Predictor — Pearls Virtual Internship

A fully automated, serverless Air Quality Index (AQI) prediction system that collects live pollution data every hour, trains a machine learning model daily, and displays a 3-day forecast for any city in the world on an interactive web dashboard.

**🔗 Live Dashboard:** https://aqi-predictor-rv8nqd6ybmmwhbv7gmda7j.streamlit.app/

---

## 📌 Project Overview

Air pollution is a serious public health issue in cities like Karachi and across the world. This system solves the problem of not knowing how air quality will change in the coming days by providing an automated 3-day AQI forecast using real pollution data and machine learning — for any city in the world.

---

## 🏗️ System Architecture

```
Weather & Pollution API
        │
        ▼
Feature Pipeline (every hour)
        │
        ▼
Feature Store (aqi_features.csv)
        │
        ▼
Training Pipeline (every day)
        │
        ▼
Model Registry (model/)
        │
        ▼
Streamlit Web Dashboard
```

| Component | Technology | Purpose |
|---|---|---|
| Feature Pipeline | Python + AQICN API | Fetches live AQI data every hour |
| Training Pipeline | scikit-learn + XGBoost | Trains ML model daily |
| Automation | GitHub Actions (CI/CD) | Runs pipelines on schedule automatically |
| Web Dashboard | Streamlit | Displays live + forecast AQI for any city |

---

## 📁 Repository Structure

```
aqi-predictor/
├── .github/
│   └── workflows/
│       ├── feature_pipeline.yml    # Runs every hour automatically
│       └── training_pipeline.yml   # Runs every day automatically
├── model/
│   ├── best_model.pkl              # Trained ML model
│   ├── features.pkl                # Feature list
│   ├── model_info.pkl              # Model metadata
│   └── shap_importance.csv         # SHAP feature importance values
├── app.py                          # Streamlit web dashboard
├── feature_pipeline.py             # Hourly data collection script
├── backfill_pipeline.py            # Historical data backfill script
├── training_pipeline.py            # ML model training script
├── advanced_training.py            # XGBoost + SHAP advanced training
├── aqi_features.csv                # Live collected features
├── aqi_historical.csv              # 90 days of historical data
├── requirements.txt                # Python dependencies
└── report.md                       # Full project report
```

---

## 🚀 Features

- **🌍 Global city search** — search any city in the world by name
- **⚡ Quick select buttons** — Karachi, Lahore, Delhi, Dubai, London, Beijing, New York, Paris
- **📊 Live AQI metrics** — current AQI, PM2.5, temperature, humidity, wind speed
- **🚨 Smart health alerts** — 5 levels from Good to Hazardous with actionable advice
- **📅 3-day forecast** — color coded forecast cards with line chart
- **🎨 AQI scale guide** — color reference from 0 to 300+
- **🧪 Pollutant breakdown** — bar chart of all available pollutants
- **🤖 Model performance** — live RMSE, MAE, R² metrics
- **📈 SHAP explanations** — shows which factors drive each prediction

---

## 🧠 Machine Learning

### Features Used (14 total)

| Feature | Type |
|---|---|
| aqi, pm25, temperature, humidity, wind_speed | Raw sensor data |
| hour, day_of_week, month, is_weekend, day_of_year | Time-based |
| aqi_change, aqi_change_pct, aqi_rolling_3d, aqi_rolling_7d | Derived |

### Models Tested

| Model | RMSE | MAE | R² |
|---|---|---|---|
| Ridge Regression | 17.36 | 13.59 | 0.012 |
| **Random Forest** ✓ | **16.74** | **13.37** | **0.081** |
| Gradient Boosting | 20.22 | 15.51 | -0.341 |
| XGBoost (tuned) | 18.89 | 13.97 | -0.171 |

**Best model: Random Forest** (lowest RMSE of 16.74)

### Top Features by SHAP Importance

| Rank | Feature | SHAP Value |
|---|---|---|
| 1 | temperature | 3.103 |
| 2 | day_of_year | 3.041 |
| 3 | aqi_rolling_3d | 2.796 |
| 4 | aqi | 2.304 |
| 5 | day_of_week | 1.462 |

---

## ⚙️ Automated Pipeline

Two GitHub Actions workflows run automatically:

- **Feature Pipeline** — runs every hour, fetches fresh AQI data, saves to CSV
- **Training Pipeline** — runs every day at 2AM UTC, retrains model with latest data

The system updates itself with zero manual effort.

---

## 🛠️ How to Run Locally

**1. Clone the repository**
```bash
git clone https://github.com/SumeetJani/aqi-predictor.git
cd aqi-predictor
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the feature pipeline**
```bash
python feature_pipeline.py
```

**4. Run the training pipeline**
```bash
python training_pipeline.py
```

**5. Launch the dashboard**
```bash
streamlit run app.py
```

---

## 📦 Dependencies

```
requests
pandas
numpy
scikit-learn
xgboost
shap
streamlit
```

---

## 🌐 Data Source

- **API:** [AQICN](https://aqicn.org/api/) — World Air Quality Index Project
- **Coverage:** Thousands of cities worldwide
- **Update frequency:** Every hour via automated pipeline

---

## 📊 AQI Scale Reference

| AQI Range | Category | Health Advice |
|---|---|---|
| 0 - 50 | Good ✅ | Air quality is satisfactory |
| 51 - 100 | Moderate 🟡 | Acceptable for most people |
| 101 - 150 | Unhealthy for Sensitive Groups 🟠 | Sensitive groups should limit outdoor activity |
| 151 - 200 | Unhealthy 🔴 | Everyone should limit outdoor activity |
| 201 - 300 | Very Unhealthy 🟣 | Avoid outdoor activity |
| 300+ | Hazardous ☠️ | Stay indoors |

---

## 👨‍💻 Author

**Sumeet Jani**
DHA Suffa University, Karachi
Pearls Virtual Internship — SHINE Intern Cohort 8

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
