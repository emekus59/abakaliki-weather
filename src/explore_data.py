import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

os.makedirs("plots", exist_ok=True)
df = pd.read_csv("data/abakaliki_clean.csv", index_col="time", parse_dates=True)

# 1. Daily average temperature over the years
plt.figure(figsize=(12, 4))
df["temperature_2m"].resample("D").mean().plot()
plt.title("Daily average temperature, Abakaliki")
plt.ylabel("°C")
plt.tight_layout()
plt.savefig("plots/1_temperature_trend.png")
plt.close()

# 2. Average temperature by hour of the day
plt.figure(figsize=(8, 4))
df.groupby(df.index.hour)["temperature_2m"].mean().plot(marker="o")
plt.title("Average temperature by hour of day")
plt.xlabel("Hour")
plt.ylabel("°C")
plt.tight_layout()
plt.savefig("plots/2_daily_cycle.png")
plt.close()

# 3. Monthly pattern: temperature and rainfall
monthly_temp = df.groupby(df.index.month)["temperature_2m"].mean()
monthly_rain = df.resample("ME")["precipitation"].sum().groupby(lambda d: d.month).mean()
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
monthly_temp.plot(kind="bar", ax=ax[0], color="orange")
ax[0].set_title("Average temperature by month")
ax[0].set_xlabel("Month")
monthly_rain.plot(kind="bar", ax=ax[1], color="steelblue")
ax[1].set_title("Average monthly rainfall (mm)")
ax[1].set_xlabel("Month")
plt.tight_layout()
plt.savefig("plots/3_monthly_pattern.png")
plt.close()

# 4. Correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation between variables")
plt.tight_layout()
plt.savefig("plots/4_correlation.png")
plt.close()

print("Done. Open the 'plots' folder to see the charts.")