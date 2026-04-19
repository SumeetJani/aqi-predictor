import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("XGBoost not found. Run: pip install xgboost")

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("SHAP not found. Run: pip install shap")

print("="*55)
print("  AQI ADVANCED TRAINING — KARACHI")
print("="*55)

# ── 1. Load data ─────────────────────────────────────────────
df = pd.read_csv("aqi_historical.csv")
df = df.dropna(subset=["aqi_next_day"])

WANTED_FEATURES = [
    "aqi", "pm25", "pm10", "o3", "no2", "so2", "co",
    "temperature", "humidity", "wind_speed",
    "hour", "day_of_week", "month", "is_weekend", "day_of_year",
    "aqi_change", "aqi_change_pct", "aqi_rolling_3d", "aqi_rolling_7d"
]

# Only use columns that exist AND have real data
FEATURE_COLS = [
    c for c in WANTED_FEATURES
    if c in df.columns and df[c].notna().sum() > 0
]

TARGET_COL = "aqi_next_day"

X = df[FEATURE_COLS].copy()
y = df[TARGET_COL].copy()

# Fill remaining missing values with column mean
for col in FEATURE_COLS:
    if X[col].isnull().any():
        X[col] = X[col].fillna(X[col].mean())

print(f"\nLoaded {len(df)} rows")
print(f"Features used: {len(FEATURE_COLS)} — {FEATURE_COLS}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── 2. Load previous best score to compare ───────────────────
prev_rmse = 9999
prev_name = "None"
try:
    with open("model/model_info.pkl", "rb") as f:
        info = pickle.load(f)
    prev_rmse = info["rmse"]
    prev_name = info["name"]
    print(f"\nPrevious best: {prev_name} (RMSE={prev_rmse:.2f})")
except:
    print("\nNo previous model found — starting fresh.")

# ── 3. Train XGBoost with hyperparameter tuning ──────────────
if not HAS_XGB:
    print("XGBoost not available. Exiting.")
    exit()

print("\nTraining XGBoost with hyperparameter tuning...")
print("(This may take 3-5 minutes — please wait)\n")

param_grid = {
    "n_estimators":  [100, 200],
    "max_depth":     [3, 5],
    "learning_rate": [0.05, 0.1],
    "subsample":     [0.8, 1.0],
}

xgb_model = xgb.XGBRegressor(random_state=42, verbosity=0)
grid = GridSearchCV(
    xgb_model, param_grid,
    cv=3, scoring="neg_root_mean_squared_error",
    n_jobs=-1
)
grid.fit(X_train, y_train)

best_xgb = grid.best_estimator_
preds    = best_xgb.predict(X_test)
xgb_rmse = np.sqrt(mean_squared_error(y_test, preds))
xgb_mae  = mean_absolute_error(y_test, preds)
xgb_r2   = r2_score(y_test, preds)

print(f"  XGBoost Results")
print(f"    Best params : {grid.best_params_}")
print(f"    RMSE        : {xgb_rmse:.2f}  (lower is better)")
print(f"    MAE         : {xgb_mae:.2f}  (lower is better)")
print(f"    R²          : {xgb_r2:.3f}  (closer to 1.0 is better)")

# ── 4. Compare and save best ─────────────────────────────────
print("\n" + "─"*55)
os.makedirs("model", exist_ok=True)

if xgb_rmse < prev_rmse:
    print(f"  ✓ XGBoost is BETTER! ({xgb_rmse:.2f} vs {prev_rmse:.2f})")
    print(f"  Saving XGBoost as new best model...")
    final_model = best_xgb

    with open("model/best_model.pkl", "wb") as f:
        pickle.dump(best_xgb, f)
    with open("model/features.pkl", "wb") as f:
        pickle.dump(FEATURE_COLS, f)
    with open("model/model_info.pkl", "wb") as f:
        pickle.dump({
            "name":       "XGBoost",
            "rmse":       xgb_rmse,
            "mae":        xgb_mae,
            "r2":         xgb_r2,
            "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "features":   FEATURE_COLS,
            "params":     grid.best_params_,
        }, f)
else:
    print(f"  Previous model still wins ({prev_name}, RMSE={prev_rmse:.2f})")
    print(f"  Keeping existing best model.")
    with open("model/best_model.pkl", "rb") as f:
        final_model = pickle.load(f)

# ── 5. SHAP feature importance ───────────────────────────────
if HAS_SHAP:
    print("\n" + "─"*55)
    print("  SHAP FEATURE IMPORTANCE")
    print("  (Which factors affect AQI the most?)")
    print("─"*55)

    explainer   = shap.TreeExplainer(best_xgb)
    shap_values = explainer.shap_values(X_test)
    mean_shap   = np.abs(shap_values).mean(axis=0)

    shap_df = pd.DataFrame({
        "feature":    FEATURE_COLS,
        "importance": mean_shap
    }).sort_values("importance", ascending=False)

    for _, row in shap_df.head(8).iterrows():
        bar = "█" * int(row["importance"] / mean_shap.max() * 30)
        print(f"  {row['feature']:<22} {bar} {row['importance']:.3f}")

    shap_df.to_csv("model/shap_importance.csv", index=False)
    print("\n  SHAP values saved to model/shap_importance.csv")
else:
    print("\nSHAP not available — skipping feature importance.")

print("\n" + "="*55)

print("="*55)