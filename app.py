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
from reportlab.lib.units import inch
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer

# ---------------- CONFIG ----------------
DEFAULT_DATA_URL = "https://docs.google.com/spreadsheets/d/1OjJv6zwE-Be-nMu8DPeJb1j9dmjhMPJlMnN2hMfqdlU/gviz/tq?tqx=out:csv&sheet=Generated%20Data"

st.set_page_config(page_title="Mining Dashboard", layout="wide")

# ---------------- CLEAN DATA ----------------
def normalize_data(df):
    # Fix column names
    df.columns = df.columns.str.strip()

    if "Date" not in df.columns:
        raise ValueError("❌ 'Date' column missing")

    # Convert Date
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # Drop only invalid dates
    df = df.dropna(subset=["Date"])

    # Convert numeric safely
    for col in df.columns:
        if col != "Date":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows where ALL numeric values are NaN
    value_cols = [c for c in df.columns if c != "Date"]
    df = df.dropna(subset=value_cols, how="all")

    df = df.sort_values("Date").reset_index(drop=True)

    if df.empty:
        raise ValueError("❌ No valid rows after cleaning (check sheet format)")

    return df

# ---------------- LOAD ----------------
st.sidebar.header("Data Source")
mode = st.sidebar.radio("Source", ["Google Sheet", "Upload CSV"])

df = None

if mode == "Google Sheet":
    url = st.sidebar.text_input("CSV URL", DEFAULT_DATA_URL)

    try:
        df = pd.read_csv(url)

        if df.empty:
            st.error("❌ Sheet is empty")
            st.stop()

        df = normalize_data(df)

    except Exception as e:
        st.error(f"❌ Failed to load data: {e}")
        st.stop()

else:
    file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

    if file is None:
        st.warning("Upload file first")
        st.stop()

    try:
        df = pd.read_csv(file)
        df = normalize_data(df)
    except Exception as e:
        st.error(f"❌ CSV Error: {e}")
        st.stop()

# Safety
if df is None or df.empty:
    st.error("❌ Data not loaded")
    st.stop()

# ---------------- MAIN ----------------
mine_cols = [c for c in df.columns if c != "Date"]
df["Total"] = df[mine_cols].sum(axis=1)
series_cols = mine_cols + ["Total"]

# ---------------- SETTINGS ----------------
st.sidebar.header("Detection Settings")
z_thresh = st.sidebar.slider("Z-score", 1.0, 5.0, 2.5)
iqr_factor = st.sidebar.slider("IQR", 1.0, 4.0, 1.5)
ma_window = st.sidebar.slider("MA Window", 2, 30, 7)

# ---------------- FUNCTIONS ----------------
def detect(series):
    s = series.astype(float)

    std = s.std()
    if std == 0 or np.isnan(std):
        z_flag = pd.Series(False, index=s.index)
    else:
        z_flag = ((s - s.mean()).abs() / std) > z_thresh

    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    iqr_flag = (s < q1 - iqr_factor * iqr) | (s > q3 + iqr_factor * iqr)

    ma = s.rolling(ma_window, min_periods=ma_window).mean()
    ma_flag = ((s - ma).abs() / ma).fillna(0) > 0.2

    return z_flag | iqr_flag | ma_flag


def trend(series):
    x = np.arange(len(series))
    deg = min(2, len(series) - 1)

    if deg < 1:
        return series.values

    return np.poly1d(np.polyfit(x, series.values, deg))(x)

# ---------------- UI ----------------
st.title("📊 Mining Data Dashboard")

c1, c2, c3 = st.columns(3)
c1.metric("Rows", len(df))

start = df["Date"].min()
end = df["Date"].max()

c2.metric("Start", str(start.date()) if pd.notna(start) else "Invalid")
c3.metric("End", str(end.date()) if pd.notna(end) else "Invalid")

selected = st.selectbox("Select Series", series_cols)

# ---------------- ANALYSIS ----------------
flags = detect(df[selected])
trend_line = trend(df[selected])

# ---------------- CHART ----------------
fig = px.line(df, x="Date", y=selected, title=selected)

fig.add_scatter(x=df["Date"], y=trend_line, name="Trend")

fig.add_scatter(
    x=df["Date"][flags],
    y=df[selected][flags],
    mode="markers",
    name="Anomalies",
    marker=dict(color="red", size=8),
)

st.plotly_chart(fig, use_container_width=True)

# ---------------- TABLE ----------------
st.subheader("🔍 Anomalies")
anomalies = df[flags]

if anomalies.empty:
    st.info("No anomalies detected")
else:
    st.dataframe(anomalies)

# ---------------- PDF ----------------
def create_pdf():
    buffer = BytesIO()
    styles = getSampleStyleSheet()

    doc = SimpleDocTemplate(buffer, pagesize=A4)
    elements = []

    # Title
    elements.append(Paragraph("Mining Analytics Report", styles["Title"]))
    elements.append(Spacer(1, 10))

    # Summary
    elements.append(Paragraph(f"Rows: {len(df)}", styles["Normal"]))
    elements.append(Paragraph(f"Series: {selected}", styles["Normal"]))
    elements.append(Spacer(1, 10))

    # Chart image
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")

    plt.figure(figsize=(8, 3))
    plt.plot(df["Date"], df[selected], label="Data")
    plt.plot(df["Date"], trend_line, label="Trend")
    plt.scatter(df["Date"][flags], df[selected][flags], color="red", label="Anomalies")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(tmp.name)
    plt.close()

    elements.append(Image(tmp.name, width=6 * inch, height=3 * inch))
    elements.append(PageBreak())

    # Detailed anomalies
    elements.append(Paragraph("Detailed Anomalies", styles["Heading2"]))

    if anomalies.empty:
        elements.append(Paragraph("No anomalies detected.", styles["Normal"]))
    else:
        for i, (_, row) in enumerate(anomalies.iterrows(), 1):
            elements.append(Paragraph(f"Anomaly {i}", styles["Heading3"]))
            elements.append(Paragraph(f"Date: {row['Date'].date()}", styles["Normal"]))
            elements.append(Paragraph(f"Value: {row[selected]:.2f}", styles["Normal"]))
            elements.append(Spacer(1, 8))

    doc.build(elements)
    buffer.seek(0)
    return buffer

# ---------------- BUTTON ----------------
if st.button("📄 Generate PDF Report"):
    pdf = create_pdf()
    st.download_button("Download PDF", pdf, "mining_report.pdf")