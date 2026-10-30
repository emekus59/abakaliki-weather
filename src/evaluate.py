import joblib
import pandas as pd
import matplotlib.pyplot as plt

test = pd.read_csv("data/test.csv", index_col="time", parse_dates=True)
features = joblib.load("models/feature_list.joblib")

for target in ["target_1h", "target_24h"]:
    model = joblib.load(f"models/best_{target}.joblib")
    pred = pd.Series(model.predict(test[features]), index=test.index)

    # Actual vs predicted for the first 7 days of the test period
    plt.figure(figsize=(12, 4))
    test[target].iloc[:168].plot(label="Actual")
    pred.iloc[:168].plot(label="Predicted", linestyle="--")
    plt.title(f"Actual vs predicted temperature ({target})")
    plt.ylabel("°C")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"plots/5_actual_vs_pred_{target}.png")
    plt.close()

# Most important features (1-hour model)
model = joblib.load("models/best_target_1h.joblib")
imp = pd.Series(model.feature_importances_, index=features).nlargest(12)
plt.figure(figsize=(8, 5))
imp.sort_values().plot(kind="barh")
plt.title("Top 12 features (1-hour model)")
plt.tight_layout()
plt.savefig("plots/6_feature_importance.png")
plt.close()

print("Done. Open the plots folder.")