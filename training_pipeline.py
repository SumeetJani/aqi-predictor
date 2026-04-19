import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

print("="*55)
print("  AQI TRAINING PIPELINE — KARACHI")
print("="*55)

# ── 1. Load data ─────────────────────────────────────────────
df = pd.read_csv("aqi_historical.csv")
print(f"\nLoaded {len(df)} rows from aqi_historical.csv")

# Drop rows where target is missing
df = df.dropna(subset=["aqi_next_day"])

# ── 2. Select features ───────────────────────────────────────
WANTED_FEATURES = [
    "aqi", "pm25", "pm10", "o3", "no2", "so2", "co",
    "temperature", "humidity", "wind_speed",
    "hour", "day_of_week", "month", "is_weekend", "day_of_year",
    "aqi_change", "aqi_change_pct", "aqi_rolling_3d", "aqi_rolling_7d"
]

TARGET_COL = "aqi_next_day"

# ── KEY FIX: only use columns that EXIST and have at least some data ──
FEATURE_COLS = []
for col in WANTED_FEATURES:
    if col in df.columns and df[col].notna().sum() > 0:
        FEATURE_COLS.append(col)
    else:
        print(f"  Skipping column '{col}' — not in data or all empty")

print(f"\nFeatures used: {len(FEATURE_COLS)}")
print(f"Columns: {FEATURE_COLS}")
print(f"Target: {TARGET_COL}")

X = df[FEATURE_COLS].copy()
y = df[TARGET_COL].copy()

# ── 3. Fill remaining missing values ─────────────────────────
# Fill each column with its own mean (safe since we checked above)
for col in FEATURE_COLS:
    if X[col].isnull().any():
        X[col] = X[col].fillna(X[col].mean())

print(f"Missing values after filling: {X.isnull().sum().sum()}")

# ── 4. Train / test split ────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nTrain size: {len(X_train)} rows")
print(f"Test size:  {len(X_test)} rows")

# ── 5. Scale features ────────────────────────────────────────
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ── 6. Train multiple models ─────────────────────────────────
models = {
    "Ridge Regression":  Ridge(alpha=1.0),
    "Random Forest":     RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
}

def evaluate(name, model, X_tr, X_te, y_tr, y_te):
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)
    rmse  = np.sqrt(mean_squared_error(y_te, preds))
    mae   = mean_absolute_error(y_te, preds)
    r2    = r2_score(y_te, preds)
    print(f"\n  {name}")
    print(f"    RMSE : {rmse:.2f}  (lower is better)")
    print(f"    MAE  : {mae:.2f}  (lower is better)")
    print(f"    R²   : {r2:.3f}  (closer to 1.0 is better)")
    return rmse, mae, r2, model

print("\n" + "─"*55)
print("  MODEL RESULTS")
print("─"*55)

results = {}
for name, model in models.items():
    if "Ridge" in name:
        rmse, mae, r2, fitted = evaluate(
            name, model, X_train_sc, X_test_sc, y_train, y_test)
    else:
        rmse, mae, r2, fitted = evaluate(
            name, model, X_train, X_test, y_train, y_test)
    results[name] = {"rmse": rmse, "mae": mae, "r2": r2, "model": fitted}

# ── 7. Pick best model ───────────────────────────────────────
best_name  = min(results, key=lambda k: results[k]["rmse"])
best_model = results[best_name]["model"]
print(f"\n  ✓ Best model: {best_name} (RMSE={results[best_name]['rmse']:.2f})")

# ── 8. Feature importance ────────────────────────────────────
if hasattr(best_model, "feature_importances_"):
    print("\n" + "─"*55)
    print("  TOP 5 MOST IMPORTANT FEATURES")
    print("─"*55)
    imp  = pd.Series(best_model.feature_importances_, index=FEATURE_COLS)
    top5 = imp.sort_values(ascending=False).head(5)
    for feat, score in top5.items():
        bar = "█" * int(score * 40)
        print(f"  {feat:<22} {bar} {score:.3f}")

# ── 9. Save everything ───────────────────────────────────────
os.makedirs("model", exist_ok=True)

with open("model/best_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

with open("model/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("model/features.pkl", "wb") as f:
    pickle.dump(FEATURE_COLS, f)

with open("model/model_info.pkl", "wb") as f:
    pickle.dump({
        "name":       best_name,
        "rmse":       results[best_name]["rmse"],
        "mae":        results[best_name]["mae"],
        "r2":         results[best_name]["r2"],
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "features":   FEATURE_COLS,
    }, f)

print("\n" + "="*55)
print("  TRAINING COMPLETE!")
print(f"  Best model : {best_name}")
print(f"  Saved to   : model/best_model.pkl")
print("  Ready for Week 4 — advanced models!")
print("="*55)