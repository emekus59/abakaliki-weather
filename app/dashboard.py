import sqlite3
import sys

sys.path.append("src")

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from monitor import LAT, LON, run_once

st.set_page_config(page_title="Abakaliki Weather", layout="centered")

BLUE, ORANGE, RED = "#3b82e6", "#e8643a", "#e5484d"
GRID = "rgba(128,128,128,0.25)"
URL = "https://api.open-meteo.com/v1/forecast"

st.markdown(
    """
<style>
.block-container { padding-top: 4.5rem; max-width: 760px; }
.w-name { font-size: 18px; font-weight: 600; }
.w-muted, .w-unit, .w-side, .w-lab, .w-dlt, .w-sub, .w-right { opacity: 0.65; }
.w-muted { font-size: 13px; }
.w-strip { display: flex; align-items: center; gap: 8px;
           background: rgba(128,128,128,0.10); border: 0.5px solid rgba(128,128,128,0.28);
           border-radius: 8px; padding: 8px 12px; margin: 8px 0 14px; font-size: 13px; }
.w-dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; }
.w-ok { background: #1a9d3a; }
.w-bad { background: #e5484d; }
.w-right { margin-left: auto; }
.w-hero { display: flex; justify-content: space-between; align-items: flex-end;
          margin-bottom: 14px; }
.w-big { font-family: Georgia, "Times New Roman", serif; font-size: 64px; line-height: 1; }
.w-unit { font-size: 20px; margin-left: 4px; }
.w-side { text-align: right; font-size: 13px; line-height: 1.7; }
.w-tiles { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 18px; }
.w-tile { background: rgba(128,128,128,0.10); border: 0.5px solid rgba(128,128,128,0.28);
          border-radius: 8px; padding: 10px 12px; }
.w-lab { font-size: 12px; }
.w-val { font-size: 24px; font-weight: 600; }
.w-dlt { font-size: 12px; }
.w-sec { font-size: 15px; font-weight: 600; margin-top: 8px; }
.w-sub { font-size: 12px; margin-bottom: 4px; }
@media (max-width: 600px) { .w-big { font-size: 48px; } .w-val { font-size: 20px; } }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data(ttl=900, show_spinner=False)
def _fetch_hourly():
    params = {
        "latitude": LAT, "longitude": LON, "hourly": "temperature_2m",
        "past_days": 6, "forecast_days": 1, "timezone": "Africa/Lagos",
    }
    r = requests.get(URL, params=params, timeout=15)
    r.raise_for_status()
    h = r.json()["hourly"]
    return pd.DataFrame({"time": pd.to_datetime(h["time"]), "temp": h["temperature_2m"]})


def get_hourly(df, latest_time):
    try:
        hr = _fetch_hourly()
        hr = hr[hr["time"] <= latest_time].dropna()
        if len(hr) > 24:
            return hr
    except (requests.RequestException, KeyError, ValueError):
        pass
    return df.rename(columns={"temperature": "temp"})[["time", "temp"]]


def load_readings():
    try:
        con = sqlite3.connect("data/weather.db")
        df = pd.read_sql("select * from readings order by time", con,
                         parse_dates=["time"])
        con.close()
        return df
    except Exception:
        return pd.DataFrame()


def style(fig, height=290):
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=14, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


def hourly_fig(hr, latest):
    t0 = latest["time"]
    now_t = float(latest["temperature"])
    p1, p24 = float(latest["pred_1h"]), float(latest["pred_24h"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hr["time"], y=hr["temp"], mode="lines", name="Recorded",
        line=dict(color=BLUE, width=2), fill="tozeroy",
        fillcolor="rgba(59,130,230,0.15)",
        hovertemplate="%{x|%a %d %b, %H:%M}<br>%{y:.1f} °C<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=[t0, t0 + pd.Timedelta(hours=1), t0 + pd.Timedelta(hours=24)],
        y=[now_t, p1, p24], mode="lines+markers", name="ML forecast",
        line=dict(color=ORANGE, width=2, dash="dot"),
        marker=dict(symbol="diamond", size=[0, 10, 10], color=ORANGE),
        hovertemplate="%{x|%a %d %b, %H:%M}<br>%{y:.1f} °C (forecast)<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=[t0], y=[now_t], mode="markers+text", text=[f"Now {now_t:.1f}"],
        textposition="top center", marker=dict(color=BLUE, size=9),
        showlegend=False, hoverinfo="skip"))
    fig.add_hline(y=35, line_dash="dash", line_color=RED, line_width=1,
                  annotation_text="Alert level 35 °C", annotation_position="top left",
                  annotation_font_color=RED)
    days = pd.date_range(hr["time"].min().normalize(),
                         t0.normalize() + pd.Timedelta(days=1), freq="D")
    top = max(36, max(float(hr["temp"].max()), p1, p24) + 1.5)
    style(fig)
    fig.update_layout(showlegend=True, legend=dict(orientation="h", y=-0.12, x=0))
    fig.update_xaxes(tickvals=days, ticktext=[d.strftime("%a") for d in days],
                     showgrid=False,
                     range=[hr["time"].min(), t0 + pd.Timedelta(hours=27)])
    fig.update_yaxes(range=[float(hr["temp"].min()) - 2, top], ticksuffix="°")
    return fig


def range_fig(hr, latest):
    d = (hr.assign(day=hr["time"].dt.normalize())
           .groupby("day")["temp"].agg(["min", "max"]).reset_index())
    labels = d["day"].dt.strftime("%a")
    now_t = float(latest["temperature"])
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels, y=d["max"] - d["min"], base=d["min"], width=0.35,
        marker_color=BLUE, customdata=d[["min", "max"]].values,
        hovertemplate="%{x}<br>Low %{customdata[0]:.1f} °C<br>High %{customdata[1]:.1f} °C<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=labels, y=d["max"], mode="text", text=[f"{v:.0f}" for v in d["max"]],
        textposition="top center", hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=labels, y=d["min"], mode="text", text=[f"{v:.0f}" for v in d["min"]],
        textposition="bottom center", hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=[labels.iloc[-1]], y=[now_t], mode="markers",
        marker=dict(color=ORANGE, size=9),
        hovertemplate="Latest reading %{y:.1f} °C<extra></extra>"))
    fig.add_hline(y=35, line_dash="dash", line_color=RED, line_width=1,
                  annotation_text="Alert level 35 °C", annotation_position="top left",
                  annotation_font_color=RED)
    style(fig)
    fig.update_layout(bargap=0.4)
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(range=[float(d["min"].min()) - 3, max(36, float(d["max"].max()) + 2)],
                     ticksuffix="°")
    return fig


def heat_fig(hr):
    h = hr.assign(day=hr["time"].dt.normalize(), hour=hr["time"].dt.hour)
    z = h.pivot_table(index="day", columns="hour", values="temp").reindex(columns=range(24))
    fig = go.Figure(go.Heatmap(
        z=z.values, x=list(range(24)), y=z.index.strftime("%a"),
        colorscale="YlOrRd",
        zmin=float(hr["temp"].min()), zmax=float(hr["temp"].max()),
        xgap=2, ygap=2, hoverongaps=False,
        colorbar=dict(thickness=10, ticksuffix="°", outlinewidth=0),
        hovertemplate="%{y} %{x}:00<br>%{z:.1f} °C<extra></extra>"))
    style(fig, height=240)
    fig.update_xaxes(tickvals=[0, 6, 12, 18], ticktext=["0:00", "6:00", "12:00", "18:00"],
                     showgrid=False)
    fig.update_yaxes(autorange="reversed", showgrid=False)
    return fig


# Header with the update button (always visible, even before the first reading)
left, right = st.columns([3, 1])
with right:
    if st.button("↻ Update now"):
        with st.spinner("Fetching latest weather..."):
            run_once()
            _fetch_hourly.clear()
        st.rerun()

df = load_readings()
if df.empty:
    with left:
        st.markdown('<div class="w-name">Abakaliki</div>', unsafe_allow_html=True)
    st.info("No readings yet. Click 'Update now'.")
    st.stop()

latest = df.iloc[-1]
age = pd.Timestamp.now(tz="Africa/Lagos").tz_localize(None) - latest["time"]
mins = max(int(age.total_seconds() // 60), 0)
ago = f"{mins} min ago" if mins < 90 else f"{mins // 60} h ago"
with left:
    st.markdown(
        f'<div class="w-name">Abakaliki</div><div class="w-muted">Updated {ago}</div>',
        unsafe_allow_html=True,
    )

alerts = str(latest["alerts"])
if alerts != "None":
    strip = f'<div class="w-strip"><span class="w-dot w-bad"></span>{alerts}</div>'
else:
    strip = ('<div class="w-strip"><span class="w-dot w-ok"></span>All clear'
             '<span class="w-right">Alert levels: 35 °C · 7.5 mm/h · 30 km/h</span></div>')

p1, p24 = float(latest["pred_1h"]), float(latest["pred_24h"])
d1 = p1 - float(latest["temperature"])
d24 = p24 - float(latest["temperature"])
top_html = (
    strip
    + '<div class="w-hero"><div>'
    + f'<span class="w-big">{latest["temperature"]:.1f}</span><span class="w-unit">°C</span></div>'
    + f'<div class="w-side">Humidity {latest["humidity"]:.0f}%<br>Wind {latest["wind"]:.1f} km/h</div></div>'
    + '<div class="w-tiles">'
    + f'<div class="w-tile"><div class="w-lab">Next hour (ML)</div><div class="w-val">{p1:.1f} °C</div><div class="w-dlt">{d1:+.1f} from now</div></div>'
    + f'<div class="w-tile"><div class="w-lab">In 24 hours (ML)</div><div class="w-val">{p24:.1f} °C</div><div class="w-dlt">{d24:+.1f} from now</div></div>'
    + '</div>'
)
st.markdown(top_html, unsafe_allow_html=True)

hr = get_hourly(df, latest["time"])
cfg = {"displayModeBar": False}

st.markdown('<div class="w-sec">Recorded temperature</div>'
            '<div class="w-sub">Last 7 days from Open-Meteo, with the ML forecast points</div>',
            unsafe_allow_html=True)
view = st.radio("View", ["Hourly", "Daily range"], horizontal=True,
                label_visibility="collapsed")
fig = hourly_fig(hr, latest) if view == "Hourly" else range_fig(hr, latest)
st.plotly_chart(fig, config=cfg)

st.markdown('<div class="w-sec">Daily cycle</div>'
            '<div class="w-sub">Temperature by hour of day, darker is hotter</div>',
            unsafe_allow_html=True)
st.plotly_chart(heat_fig(hr), config=cfg)

tab1, tab2, tab3 = st.tabs(["Readings", "Climate", "Models"])
with tab1:
    table = df.tail(24).sort_values("time", ascending=False).rename(columns={
        "time": "Time", "temperature": "Temp (°C)", "humidity": "Humidity (%)",
        "pressure": "Pressure (hPa)", "rain": "Rain (mm)", "wind": "Wind (km/h)",
        "pred_1h": "Next hour (°C)", "pred_24h": "In 24 h (°C)", "alerts": "Alerts",
    })
    st.dataframe(table, hide_index=True)
with tab2:
    hist = pd.read_csv("data/abakaliki_clean.csv", index_col="time", parse_dates=True)
    st.caption("Average temperature by month, 2021 to 2026 (°C)")
    st.bar_chart(hist.groupby(hist.index.month)["temperature_2m"].mean())
with tab3:
    st.dataframe(pd.read_csv("data/model_results.csv"), hide_index=True)

st.caption("Weather data by [Open-Meteo.com](https://open-meteo.com/). "
           "ML forecasts come from XGBoost models trained on Abakaliki data.")
