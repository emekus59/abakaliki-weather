import pandas as pd

df = pd.read_csv("data/abakaliki_hourly.csv", parse_dates=["time"])

print("Rows:", len(df))
print("\nMissing values per column:")
print(df.isna().sum())
print("\nDuplicate timestamps:", df["time"].duplicated().sum())

# Sort by time and remove duplicates
df = df.sort_values("time").drop_duplicates(subset="time")

# Fill small gaps by interpolating between neighbouring hours
df = df.set_index("time")
df = df.interpolate(method="time", limit=6)
df = df.dropna()

print("\nValue ranges:")
print(df.describe().T[["min", "max", "mean"]])

df.to_csv("data/abakaliki_clean.csv")
print("\nSaved data/abakaliki_clean.csv with", len(df), "rows")