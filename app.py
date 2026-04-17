import os
import tempfile
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

# ---------------- CONFIG ----------------
DEFAULT_DATA_URL = "https://docs.google.com/spreadsheets/d/1OjJv6zwE-Be-nMu8DPeJb1j9dmjhMPJlMnN2hMfqdlU/gviz/tq?tqx=out:csv&sheet=Generated%20Data"

st.set_page_config(page_title="Mining Data Dashboard", layout="wide")

# ---------------- DATA CLEAN ----------------
def normalize_data(df):
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])

    for col in df.columns:
        if col != "Date":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna()
    df = df.sort_values("Date").reset_index(drop=True)

    return df

# ---------------- LOAD ----------------
st.sidebar.header("Data Source")
source_mode = st.sidebar.radio("Source", ["Google Sheet", "Upload CSV"])

if source_mode == "Google Sheet":
    url = st.sidebar.text_input("CSV URL", DEFAULT_DATA_URL)
    try:
        df = pd.read_csv(url)
        df = normalize_data(df)
    except Exception as e:
        st.error(f"❌ Failed to load data: {e}")
        st.stop()
else:
    file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if file is None:
        st.warning("Upload file first")
        st.stop()

    df = pd.read_csv(file)
    df = normalize_data(df)

# ---------------- SAFETY ----------------
if df is None or df.empty:
    st.error("No valid data")
    st.stop()

# ---------------- MAIN ----------------
mine_columns = [col for col in df.columns if col != "Date"]
df["Total"] = df[mine_columns].sum(axis=1)
series_columns = mine_columns + ["Total"]

# ---------------- SETTINGS ----------------
st.sidebar.header("Detection Settings")
z_thresh = st.sidebar.slider("Z-score", 1.0, 5.0, 2.5)
iqr_factor = st.sidebar.slider("IQR", 1.0, 4.0, 1.5)
ma_window = st.sidebar.slider("MA Window", 2, 30, 7)

# ---------------- FUNCTIONS ----------------
def detect(series):
    s = series.astype(float)

    # Z-score safe
    std = s.std()
    if std == 0 or np.isnan(std):
        z_flag = pd.Series(False, index=s.index)
    else:
        z = (s - s.mean()).abs() / std
        z_flag = z > z_thresh

    # IQR
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    iqr_flag = (s < q1 - iqr_factor * iqr) | (s > q3 + iqr_factor * iqr)

    # Moving average safe
    ma = s.rolling(ma_window, min_periods=ma_window).mean()
    ma_flag = ((s - ma).abs() / ma).fillna(0) > 0.2

    return z_flag | iqr_flag | ma_flag


def trend(series):
    x = np.arange(len(series))
    y = series.values

    deg = min(2, len(series) - 1)
    if deg < 1:
        return y

    coeff = np.polyfit(x, y, deg)
    return np.poly1d(coeff)(x)

# ---------------- UI ----------------
st.title("Mining Dashboard")

col1, col2, col3 = st.columns(3)
col1.metric("Rows", len(df))

start = df["Date"].min()
end = df["Date"].max()

col2.metric("Start", str(start.date()) if pd.notna(start) else "Invalid")
col3.metric("End", str(end.date()) if pd.notna(end) else "Invalid")

selected = st.selectbox("Select Mine", series_columns)

# ---------------- ANALYSIS ----------------
anomaly_flag = detect(df[selected])
trend_line = trend(df[selected])

# ---------------- CHART ----------------
fig = px.line(df, x="Date", y=selected, title=selected)

fig.add_scatter(x=df["Date"], y=trend_line, name="Trend")

fig.add_scatter(
    x=df["Date"][anomaly_flag],
    y=df[selected][anomaly_flag],
    mode="markers",
    name="Anomalies",
    marker=dict(color="red", size=8)
)

st.plotly_chart(fig, use_container_width=True)

# ---------------- TABLE ----------------
st.subheader("Anomalies")

anomalies = df[anomaly_flag]

if anomalies.empty:
    st.info("No anomalies")
else:
    st.dataframe(anomalies)

# ---------------- PDF ----------------
def create_pdf():
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()

    elements = []
    elements.append(Paragraph("Mining Report", styles["Title"]))
    elements.append(Spacer(1, 10))
    elements.append(Paragraph(f"Rows: {len(df)}", styles["Normal"]))
    elements.append(Paragraph(f"Series: {selected}", styles["Normal"]))

    doc.build(elements)
    buffer.seek(0)
    return buffer

if st.button("Generate PDF"):
    pdf = create_pdf()
    st.download_button("Download PDF", pdf, "report.pdf")