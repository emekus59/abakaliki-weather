import numpy as np
import pandas as pd

df = pd.read_csv("data/abakaliki_clean.csv", index_col="time", parse_dates=True)

# Time features (the model learns daily and seasonal patterns)
df["hour"] = df.index.hour
df["month"] = df.index.month
df["dayofyear"] = df.index.dayofyear
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

# Wind direction is a circle (360 is next to 1), so split it in two
df["wind_dir_sin"] = np.sin(np.radians(df["wind_direction_10m"]))
df["wind_dir_cos"] = np.cos(np.radians(df["wind_direction_10m"]))
df = df.drop(columns=["wind_direction_10m"])

# Lag features: what the weather was 1, 2, 3, 6, 12, 24 hours ago
for col in ["temperature_2m", "relative_humidity_2m", "surface_pressure"]:
    for lag in [1, 2, 3, 6, 12, 24]:
        df[f"{col}_lag{lag}"] = df[col].shift(lag)

# Rolling averages and trends
df["temp_roll3"] = df["temperature_2m"].rolling(3).mean()
df["temp_roll6"] = df["temperature_2m"].rolling(6).mean()
df["temp_roll24"] = df["temperature_2m"].rolling(24).mean()
df["rain_24h"] = df["precipitation"].rolling(24).sum()
df["pressure_change_3h"] = df["surface_pressure"].diff(3)

# Targets: the temperature we want to predict
df["target_1h"] = df["temperature_2m"].shift(-1)    # 1 hour ahead
df["target_24h"] = df["temperature_2m"].shift(-24)  # 24 hours ahead

df = df.dropna()

# Split by time, never randomly
train = df[df.index < "2025-01-01"]
test = df[df.index >= "2025-01-01"]

train.to_csv("data/train.csv")
test.to_csv("data/test.csv")

print("Columns:", df.shape[1])
print("Train rows:", len(train), "| from", train.index.min(), "to", train.index.max())
print("Test rows:", len(test), "| from", test.index.min(), "to", test.index.max())