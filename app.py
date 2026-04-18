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
from reportlab.platypus import (
    Paragraph, SimpleDocTemplate, Spacer,
    Image, PageBreak, Table
)

# ---------------- CONFIG ----------------
DEFAULT_DATA_URL = "https://docs.google.com/spreadsheets/d/1OjJv6zwE-Be-nMu8DPeJb1j9dmjhMPJlMnN2hMfqdlU/gviz/tq?tqx=out:csv&sheet=Generated%20Data"

st.set_page_config(page_title="Mining Dashboard", layout="wide")

# ---------------- DATA CLEAN (FINAL FIX) ----------------
def normalize_data(df):
    df.columns = df.columns.str.strip()

    if "Date" not in df.columns:
        raise ValueError("❌ Date column missing")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])

    df = df.sort_values("Date").reset_index(drop=True)

    value_cols = [c for c in df.columns if c != "Date"]

    for col in value_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 🔥 Only interpolate SMALL gaps (not full history)
    df[value_cols] = df[value_cols].interpolate(limit=3)

    return df


# ---------------- LOAD ----------------
st.sidebar.header("Data Source")
mode = st.sidebar.radio("Source", ["Google Sheet", "Upload CSV"])

df = None

if mode == "Google Sheet":
    url = st.sidebar.text_input("CSV URL", DEFAULT_DATA_URL)
    try:
        df = pd.read_csv(url)
        df = normalize_data(df)
    except Exception as e:
        st.error(f"❌ Load error: {e}")
        st.stop()
else:
    file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if file is None:
        st.stop()
    df = pd.read_csv(file)
    df = normalize_data(df)

if df is None or df.empty:
    st.error("❌ No valid data")
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
    s = series.dropna().astype(float)

    if len(s) < 5:
        return pd.Series(False, index=series.index)

    std = s.std()
    if std == 0 or np.isnan(std):
        z_flag = pd.Series(False, index=s.index)
    else:
        z_flag = ((s - s.mean()).abs() / std) > z_thresh

    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    iqr_flag = (s < q1 - iqr_factor * iqr) | (s > q3 + iqr_factor * iqr)

    ma = s.rolling(ma_window, min_periods=1).mean()
    ma_flag = ((s - ma).abs() / ma).fillna(0) > 0.2

    flags = z_flag | iqr_flag | ma_flag

    # map back
    full = pd.Series(False, index=series.index)
    full.loc[s.index] = flags

    return full


def trend(series):
    s = series.dropna()

    if len(s) < 5:
        return [None] * len(series)

    x = np.arange(len(s))
    y = s.values

    trend_vals = np.poly1d(np.polyfit(x, y, 2))(x)

    full = pd.Series(index=series.index, dtype=float)
    full.loc[s.index] = trend_vals

    return full


# ---------------- UI ----------------
st.title("📊 Mining Data Dashboard")

c1, c2, c3 = st.columns(3)
c1.metric("Rows", len(df))

start = df["Date"].min()
end = df["Date"].max()

c2.metric("Start", str(start.date()))
c3.metric("End", str(end.date()))

selected = st.selectbox("Select Mine", series_cols)

# show real data count
st.caption(f"📊 Real data points: {df[selected].notna().sum()}")

# ---------------- ANALYSIS ----------------
flags = detect(df[selected])
trend_line = trend(df[selected])

# ---------------- CHART ----------------
fig = px.line(df, x="Date", y=selected, title=selected)

# 🔥 connect gaps but DON'T fake data
fig.update_traces(connectgaps=True)

fig.add_scatter(x=df["Date"], y=trend_line, name="Trend")

fig.add_scatter(
    x=df["Date"][flags],
    y=df[selected][flags],
    mode="markers",
    name="Anomalies",
    marker=dict(color="red", size=8),
)

st.plotly_chart(fig, width="stretch")

# ---------------- TABLE ----------------
st.subheader("🔍 Anomalies")

valid_data = df[df[selected].notna()]
anomalies = valid_data[flags.loc[valid_data.index]]

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

    elements.append(Paragraph("Mining Analytics Report", styles["Title"]))
    elements.append(Spacer(1, 10))

    elements.append(Paragraph(f"Rows: {len(df)}", styles["Normal"]))
    elements.append(Paragraph(f"Selected Mine: {selected}", styles["Normal"]))
    elements.append(Spacer(1, 10))

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")

    plt.figure(figsize=(8, 3))
    plt.plot(df["Date"], df[selected], label="Data")
    plt.plot(df["Date"], trend_line, label="Trend")
    plt.scatter(df["Date"][flags], df[selected][flags], color="red")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(tmp.name)
    plt.close()

    elements.append(Image(tmp.name, width=6 * inch, height=3 * inch))
    elements.append(PageBreak())

    elements.append(Paragraph("Detailed Anomalies", styles["Heading2"]))

    mean_val = df[selected].mean()

    if anomalies.empty:
        elements.append(Paragraph("No anomalies detected.", styles["Normal"]))
    else:
        for i, (_, row) in enumerate(anomalies.iterrows(), 1):
            elements.append(Paragraph(f"Anomaly {i}", styles["Heading3"]))
            elements.append(Paragraph(f"Date: {row['Date'].date()}", styles["Normal"]))
            elements.append(Paragraph(f"Value: {row[selected]:.2f}", styles["Normal"]))

            label = "Spike" if row[selected] > mean_val else "Drop"
            elements.append(Paragraph(f"Type: {label}", styles["Normal"]))
            elements.append(Spacer(1, 10))

    doc.build(elements)
    buffer.seek(0)
    return buffer


# ---------------- BUTTON ----------------
if st.button("📄 Generate Full Report"):
    pdf = create_pdf()
    st.download_button("Download PDF", pdf, "mining_report.pdf")