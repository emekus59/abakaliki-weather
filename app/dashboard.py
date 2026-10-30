import sys
import sqlite3
sys.path.append("src")

import pandas as pd
import streamlit as st
from monitor import run_once

st.set_page_config(page_title="Abakaliki Weather", layout="wide")
st.title("Abakaliki Weather Forecast and Monitor")

if st.button("Update now"):
    with st.spinner("Fetching latest weather..."):
        run_once()
    st.rerun()

try:
    con = sqlite3.connect("data/weather.db")
    df = pd.read_sql("select * from readings order by time", con,
                     parse_dates=["time"])
    con.close()
except Exception:
    df = pd.DataFrame()

if df.empty:
    st.info("No readings yet. Click 'Update now'.")
    st.stop()

latest = df.iloc[-1]
st.caption(f"Latest reading: {latest['time']}")

# Alerts
if latest["alerts"] != "None":
    st.error(f"ALERT: {latest['alerts']}")
else:
    st.success("No weather alerts")

# Current conditions and forecast
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Temperature", f"{latest['temperature']:.1f} °C")
c2.metric("Humidity", f"{latest['humidity']:.0f} %")
c3.metric("Wind", f"{latest['wind']:.1f} km/h")
c4.metric("Next hour", f"{latest['pred_1h']:.1f} °C",
          f"{latest['pred_1h'] - latest['temperature']:+.1f}")
c5.metric("In 24 hours", f"{latest['pred_24h']:.1f} °C",
          f"{latest['pred_24h'] - latest['temperature']:+.1f}")

st.subheader("Recorded temperature")
st.line_chart(df.set_index("time")[["temperature"]])

st.subheader("Latest readings")
st.dataframe(df.tail(24).sort_values("time", ascending=False),
             use_container_width=True)

st.subheader("Average temperature by month (2021 to 2026)")
hist = pd.read_csv("data/abakaliki_clean.csv", index_col="time",
                   parse_dates=True)
st.bar_chart(hist.groupby(hist.index.month)["temperature_2m"].mean())

st.subheader("Model performance (test data)")
st.dataframe(pd.read_csv("data/model_results.csv"), use_container_width=True)