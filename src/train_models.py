import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

os.makedirs("models", exist_ok=True)
train = pd.read_csv("data/train.csv", index_col="time", parse_dates=True)
test = pd.read_csv("data/test.csv", index_col="time", parse_dates=True)

targets = ["target_1h", "target_24h"]
features = [c for c in train.columns if c not in targets]

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(
        n_estimators=100, max_depth=15, n_jobs=-1, random_state=42),
    "XGBoost": XGBRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42),
}

def score(y_true, y_pred):
    return (mean_absolute_error(y_true, y_pred),
            np.sqrt(mean_squared_error(y_true, y_pred)),
            r2_score(y_true, y_pred))

results = []
for target in targets:
    X_train, y_train = train[features], train[target]
    X_test, y_test = test[features], test[target]

    # Baseline: assume temperature stays the same as right now
    mae, rmse, r2 = score(y_test, X_test["temperature_2m"])
    results.append([target, "Baseline (no change)", mae, rmse, r2])

    best_name, best_rmse, best_model = None, 1e9, None
    for name, model in models.items():
        print(f"Training {name} for {target}...")
        model.fit(X_train, y_train)
        mae, rmse, r2 = score(y_test, model.predict(X_test))
        results.append([target, name, mae, rmse, r2])
        if rmse < best_rmse:
            best_name, best_rmse, best_model = name, rmse, model

    joblib.dump(best_model, f"models/best_{target}.joblib")
    print(f"Best for {target}: {best_name}\n")

out = pd.DataFrame(results, columns=["Target", "Model", "MAE", "RMSE", "R2"]).round(3)
out.to_csv("data/model_results.csv", index=False)
joblib.dump(features, "models/feature_list.joblib")
print(out.to_string(index=False))