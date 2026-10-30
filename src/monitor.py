import sqlite3
import joblib
import numpy as np
import pandas as pd
import requests
from apscheduler.schedulers.blocking import BlockingScheduler

LAT, LON = 6.32, 8.11  # Abakaliki
VARS = ["temperature_2m", "relative_humidity_2m", "dew_point_2m",
        "surface_pressure", "precipitation", "cloud_cover",
        "wind_speed_10m", "wind_direction_10m"]

model_1h = joblib.load("models/best_target_1h.joblib")
model_24h = joblib.load("models/best_target_24h.joblib")
features = joblib.load("models/feature_list.joblib")


def fetch_recent():
    params = {"latitude": LAT, "longitude": LON, "hourly": ",".join(VARS),
              "past_days": 2, "forecast_days": 1, "timezone": "Africa/Lagos"}
    for attempt in range(1, 6):
        try:
            r = requests.get("https://api.open-meteo.com/v1/forecast",
                             params=params, timeout=60)
            r.raise_for_status()
            df = pd.DataFrame(r.json()["hourly"])
            df["time"] = pd.to_datetime(df["time"])
            return df.set_index("time")
        except requests.RequestException as e:
            print(f"Attempt {attempt} failed: {e}")
    return None


def build_features(df):
    df = df.copy()
    df["hour"] = df.index.hour
    df["month"] = df.index.month
    df["dayofyear"] = df.index.day_of_year
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["wind_dir_sin"] = np.sin(np.radians(df["wind_direction_10m"]))
    df["wind_dir_cos"] = np.cos(np.radians(df["wind_direction_10m"]))
    for col in ["temperature_2m", "relative_humidity_2m", "surface_pressure"]:
        for lag in [1, 2, 3, 6, 12, 24]:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)
    df["temp_roll3"] = df["temperature_2m"].rolling(3).mean()
    df["temp_roll6"] = df["temperature_2m"].rolling(6).mean()
    df["temp_roll24"] = df["temperature_2m"].rolling(24).mean()
    df["rain_24h"] = df["precipitation"].rolling(24).sum()
    df["pressure_change_3h"] = df["surface_pressure"].diff(3)
    return df


def check_alerts(row, p24):
    alerts = []
    if row["temperature_2m"] >= 35:
        alerts.append("HEAT: temperature 35C or above")
    if row["precipitation"] >= 7.5:
        alerts.append("HEAVY RAIN: 7.5mm/h or more")
    if row["wind_speed_10m"] >= 30:
        alerts.append("STRONG WIND: 30km/h or more")
    if p24 >= 35:
        alerts.append("HEAT WARNING: 24h forecast 35C or above")
    return "; ".join(alerts) if alerts else "None"


def run_once():
    print("\nFetching latest weather...")
    df = fetch_recent()
    if df is None:
        print("Could not fetch data. Will try again next cycle.")
        return
    now = pd.Timestamp.now(tz="Africa/Lagos").tz_localize(None)
    df = df[df.index <= now.floor("h")]
    feats = build_features(df)
    X = feats[features].iloc[[-1]]
    if X.isna().any().any():
        print("Not enough history yet. Skipping this cycle.")
        return

    p1 = float(model_1h.predict(X)[0])
    p24 = float(model_24h.predict(X)[0])
    row = df.iloc[-1]
    alerts = check_alerts(row, p24)

    con = sqlite3.connect("data/weather.db")
    con.execute("""CREATE TABLE IF NOT EXISTS readings (
        time TEXT PRIMARY KEY, temperature REAL, humidity REAL,
        pressure REAL, rain REAL, wind REAL,
        pred_1h REAL, pred_24h REAL, alerts TEXT)""")
    con.execute("INSERT OR REPLACE INTO readings VALUES (?,?,?,?,?,?,?,?,?)",
                (str(df.index[-1]), row["temperature_2m"],
                 row["relative_humidity_2m"], row["surface_pressure"],
                 row["precipitation"], row["wind_speed_10m"],
                 p1, p24, alerts))
    con.commit()
    con.close()

    print(f"Time:          {df.index[-1]}")
    print(f"Temperature:   {row['temperature_2m']:.1f} C")
    print(f"Humidity:      {row['relative_humidity_2m']:.0f} %")
    print(f"Rain:          {row['precipitation']:.1f} mm")
    print(f"Wind:          {row['wind_speed_10m']:.1f} km/h")
    print(f"Next hour:     {p1:.1f} C")
    print(f"In 24 hours:   {p24:.1f} C")
    print(f"Alerts:        {alerts}")


if __name__ == "__main__":
    run_once()
    scheduler = BlockingScheduler()
    scheduler.add_job(run_once, "interval", minutes=60)
    print("\nMonitoring started. Press Ctrl+C to stop.")
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        print("Stopped.")