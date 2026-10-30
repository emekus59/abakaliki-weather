import requests
import pandas as pd

LAT, LON = 6.32, 8.11  # Abakaliki
START, END = "2021-01-01", "2026-10-25"

url = "https://archive-api.open-meteo.com/v1/archive"
params = {
    "latitude": LAT,
    "longitude": LON,
    "start_date": START,
    "end_date": END,
    "hourly": ",".join([
        "temperature_2m", "relative_humidity_2m", "dew_point_2m",
        "surface_pressure", "precipitation", "cloud_cover",
        "wind_speed_10m", "wind_direction_10m",
    ]),
    "timezone": "Africa/Lagos",
}

for attempt in range(1, 6):
    try:
        print(f"Attempt {attempt}...")
        r = requests.get(url, params=params, timeout=120)
        r.raise_for_status()
        break
    except requests.RequestException as e:
        print("Failed:", e)
else:
    raise SystemExit("Could not download. Try again on a better connection.")

df = pd.DataFrame(r.json()["hourly"])
df["time"] = pd.to_datetime(df["time"])
df.to_csv("data/abakaliki_hourly.csv", index=False)
print(df.shape)
print(df.head())