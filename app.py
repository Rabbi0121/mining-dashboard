import os
import tempfile
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from scipy import stats

DEFAULT_DATA_URL = (
    "https://docs.google.com/spreadsheets/d/1OjJv6zwE-Be-nMu8DPeJb1j9dmjhMPJlMnN2hMfqdlU/"
    "gviz/tq?tqx=out:csv&sheet=Generated%20Data"
)

st.set_page_config(page_title="Mining Data Dashboard", layout="wide")


@st.cache_data(show_spinner=False)
def load_sheet_data(url: str) -> pd.DataFrame:
    data = pd.read_csv(url)
    return normalize_data(data)


def normalize_data(df: pd.DataFrame) -> pd.DataFrame:
    if "Date" not in df.columns:
        raise ValueError("Input data must include a 'Date' column.")

    clean_df = df.copy()
    clean_df["Date"] = pd.to_datetime(clean_df["Date"], errors="coerce")
    clean_df = clean_df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    value_columns = [col for col in clean_df.columns if col != "Date"]
    if not value_columns:
        raise ValueError("Input data must include at least one mine column besides 'Date'.")

    for col in value_columns:
        clean_df[col] = pd.to_numeric(clean_df[col], errors="coerce")

    clean_df = clean_df.dropna().reset_index(drop=True)
    if clean_df.empty:
        raise ValueError("No valid rows remain after cleaning. Check your source values.")

    return clean_df


def compute_stats(series: pd.Series) -> dict:
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    return {
        "Mean": series.mean(),
        "Std Dev": series.std(),
        "Median": series.median(),
        "Q1": q1,
        "Q3": q3,
        "IQR": q3 - q1,
    }


def grubbs_flags(series: pd.Series, alpha: float) -> pd.Series:
    s = series.astype(float)
    n = len(s)

    if n < 3:
        return pd.Series(False, index=s.index)

    std = s.std(ddof=1)
    if std == 0 or np.isnan(std):
        return pd.Series(False, index=s.index)

    mean = s.mean()
    g_stat = (s - mean).abs() / std

    t_crit = stats.t.ppf(1 - alpha / (2 * n), n - 2)
    g_crit = ((n - 1) / np.sqrt(n)) * np.sqrt((t_crit**2) / (n - 2 + t_crit**2))

    return (g_stat > g_crit).fillna(False)


def detect_anomalies(
    series: pd.Series,
    z_thresh: float,
    iqr_factor: float,
    ma_window: int,
    ma_thresh: float,
    grubbs_alpha: float,
) -> pd.DataFrame:
    s = series.astype(float)

    std0 = s.std(ddof=0)
    if std0 == 0 or np.isnan(std0):
        z_scores = pd.Series(0.0, index=s.index)
    else:
        z_scores = (s - s.mean()).abs() / std0
    z_outliers = (z_scores > z_thresh).fillna(False)

    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0 or np.isnan(iqr):
        iqr_outliers = pd.Series(False, index=s.index)
    else:
        iqr_outliers = ((s < q1 - iqr_factor * iqr) | (s > q3 + iqr_factor * iqr)).fillna(False)

    moving_avg = s.rolling(ma_window, min_periods=ma_window).mean()
    ma_base = moving_avg.abs().replace(0, np.nan)
    ma_dist = ((s - moving_avg).abs() / ma_base).fillna(0)
    ma_outliers = (ma_dist > ma_thresh).fillna(False)

    g_outliers = grubbs_flags(s, alpha=grubbs_alpha)

    any_outlier = (z_outliers | iqr_outliers | ma_outliers | g_outliers).fillna(False)

    detected_by = []
    anomaly_type = []
    baseline = moving_avg.fillna(s.mean())

    for idx in s.index:
        methods = []
        if z_outliers.loc[idx]:
            methods.append("Z-score")
        if iqr_outliers.loc[idx]:
            methods.append("IQR")
        if ma_outliers.loc[idx]:
            methods.append("MA Distance")
        if g_outliers.loc[idx]:
            methods.append("Grubbs")

        detected_by.append(", ".join(methods))
        if any_outlier.loc[idx]:
            anomaly_type.append("Spike" if s.loc[idx] >= baseline.loc[idx] else "Drop")
        else:
            anomaly_type.append("")

    return pd.DataFrame(
        {
            "Value": s,
            "Moving Avg": moving_avg,
            "Z-score": z_scores,
            "MA Distance": ma_dist,
            "Z-score Flag": z_outliers,
            "IQR Flag": iqr_outliers,
            "MA Flag": ma_outliers,
            "Grubbs Flag": g_outliers,
            "Any Outlier": any_outlier,
            "Detected By": detected_by,
            "Anomaly Type": anomaly_type,
        },
        index=s.index,
    )


def compute_trend(series: pd.Series, degree: int) -> pd.Series:
    s = series.astype(float)
    n = len(s)

    if n < 2:
        return pd.Series(s.values, index=s.index)

    safe_degree = max(1, min(degree, n - 1))
    x = np.arange(n)
    coeff = np.polyfit(x, s.values, safe_degree)
    trend = np.poly1d(coeff)(x)
    return pd.Series(trend, index=s.index)


def build_chart(
    df: pd.DataFrame,
    mine_cols: list[str],
    selected_series: str,
    chart_type: str,
    trend_degree: int,
    trend_series: pd.Series,
    selected_anomalies: pd.DataFrame,
) -> go.Figure:
    title = f"Output Analysis: {selected_series}"

    if chart_type == "Stacked":
        long_df = df[["Date"] + mine_cols].melt(
            id_vars="Date", value_vars=mine_cols, var_name="Mine", value_name="Output"
        )
        fig = px.bar(
            long_df,
            x="Date",
            y="Output",
            color="Mine",
            barmode="stack",
            title=f"Stacked Mine Output with {selected_series} Outliers/Trend",
        )
    elif chart_type == "Bar":
        fig = px.bar(df, x="Date", y=selected_series, title=title)
    else:
        fig = px.line(df, x="Date", y=selected_series, title=title)

    fig.add_trace(
        go.Scatter(
            x=df["Date"],
            y=trend_series,
            mode="lines",
            name=f"Polynomial Trend (deg {trend_degree})",
            line=dict(color="#D81B60", width=3),
        )
    )

    flagged = selected_anomalies[selected_anomalies["Any Outlier"]]
    fig.add_trace(
        go.Scatter(
            x=flagged["Date"],
            y=flagged["Value"],
            mode="markers",
            name="Outliers",
            marker=dict(color="#E53935", size=9, symbol="circle"),
            hovertext=flagged["Detected By"],
            hovertemplate="%{x|%Y-%m-%d}<br>Value: %{y:.2f}<br>%{hovertext}<extra></extra>",
        )
    )

    fig.update_layout(legend_title="Series", height=520)
    fig.update_xaxes(title="Date")
    fig.update_yaxes(title="Output")

    return fig


def _build_table(data, col_widths=None):
    table = Table(data, colWidths=col_widths, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F4E78")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#B0B7C3")),
                ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#F7F9FC")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
            ]
        )
    )
    return table


def _matplotlib_chart_image(
    df: pd.DataFrame,
    mine_cols: list[str],
    selected_series: str,
    chart_type: str,
    trend_series: pd.Series,
    selected_anomalies: pd.DataFrame,
) -> str:
    fig, ax = plt.subplots(figsize=(11.5, 4.5))

    if chart_type == "Stacked":
        ax.stackplot(
            df["Date"],
            *[df[col].values for col in mine_cols],
            labels=mine_cols,
            alpha=0.7,
        )
    elif chart_type == "Bar":
        ax.bar(df["Date"], df[selected_series], color="#4F81BD", width=2.0)
    else:
        ax.plot(df["Date"], df[selected_series], color="#2E75B6", linewidth=2)

    ax.plot(df["Date"], trend_series, color="#D81B60", linewidth=2.2, label="Trend")

    flagged = selected_anomalies[selected_anomalies["Any Outlier"]]
    ax.scatter(flagged["Date"], flagged["Value"], color="#E53935", s=28, zorder=3, label="Outliers")

    ax.set_title(f"Output Analysis: {selected_series}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Output")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="upper left", fontsize=8)
    fig.autofmt_xdate()

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    fig.savefig(tmp.name, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return tmp.name


def save_chart_image(
    plotly_fig: go.Figure,
    df: pd.DataFrame,
    mine_cols: list[str],
    selected_series: str,
    chart_type: str,
    trend_series: pd.Series,
    selected_anomalies: pd.DataFrame,
) -> tuple[str, str]:
    try:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        plotly_fig.write_image(tmp.name, width=1400, height=700, scale=2)
        return tmp.name, ""
    except Exception as exc:  # noqa: BLE001
        fallback_path = _matplotlib_chart_image(
            df, mine_cols, selected_series, chart_type, trend_series, selected_anomalies
        )
        return fallback_path, f"Plotly image export failed ({exc}). Used matplotlib fallback."


def create_pdf(
    df: pd.DataFrame,
    mine_cols: list[str],
    series_cols: list[str],
    selected_series: str,
    chart_type: str,
    trend_degree: int,
    settings: dict,
    stats_df: pd.DataFrame,
    anomaly_summary_df: pd.DataFrame,
    anomalies_by_series: dict[str, pd.DataFrame],
    plotly_fig: go.Figure,
    trend_series: pd.Series,
) -> tuple[bytes, str]:
    chart_path, chart_note = save_chart_image(
        plotly_fig,
        df,
        mine_cols,
        selected_series,
        chart_type,
        trend_series,
        anomalies_by_series[selected_series],
    )

    buffer = BytesIO()
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=0.6 * inch,
        rightMargin=0.6 * inch,
        topMargin=0.6 * inch,
        bottomMargin=0.6 * inch,
    )
    elements = []

    elements.append(Paragraph("Weyland-Yutani Mining Analytics Report", styles["Title"]))
    elements.append(Spacer(1, 8))
    elements.append(
        Paragraph(
            (
                f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}<br/>"
                f"Date range: {df['Date'].min().date()} to {df['Date'].max().date()}<br/>"
                f"Rows analyzed: {len(df)} | Selected series: {selected_series} | "
                f"Chart: {chart_type} | Trend degree: {trend_degree}"
            ),
            styles["Normal"],
        )
    )
    elements.append(Spacer(1, 10))

    settings_rows = [["Parameter", "Value"]] + [[k, str(v)] for k, v in settings.items()]
    elements.append(Paragraph("Detection Settings", styles["Heading2"]))
    elements.append(_build_table(settings_rows, col_widths=[2.3 * inch, 3.8 * inch]))
    elements.append(Spacer(1, 12))

    stats_rows = [["Series", "Mean", "Std Dev", "Median", "Q1", "Q3", "IQR"]]
    for name in series_cols:
        row = stats_df.loc[name]
        stats_rows.append(
            [
                name,
                f"{row['Mean']:.2f}",
                f"{row['Std Dev']:.2f}",
                f"{row['Median']:.2f}",
                f"{row['Q1']:.2f}",
                f"{row['Q3']:.2f}",
                f"{row['IQR']:.2f}",
            ]
        )

    elements.append(Paragraph("Summary Statistics (Each Mine + Total)", styles["Heading2"]))
    elements.append(_build_table(stats_rows, col_widths=[1.3 * inch] + [0.9 * inch] * 6))
    elements.append(Spacer(1, 12))

    summary_rows = [[
        "Series",
        "Z-score",
        "IQR",
        "MA Dist",
        "Grubbs",
        "Any",
        "Spikes",
        "Drops",
    ]]
    for name in series_cols:
        row = anomaly_summary_df.loc[name]
        summary_rows.append(
            [
                name,
                str(int(row["Z-score"])),
                str(int(row["IQR"])),
                str(int(row["MA Distance"])),
                str(int(row["Grubbs"])),
                str(int(row["Any Outlier"])),
                str(int(row["Spikes"])),
                str(int(row["Drops"])),
            ]
        )

    elements.append(Paragraph("Anomaly Counts by Series", styles["Heading2"]))
    elements.append(_build_table(summary_rows, col_widths=[1.3 * inch] + [0.7 * inch] * 7))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph(f"Chart ({chart_type}) with Outliers and Trend", styles["Heading2"]))
    elements.append(Image(chart_path, width=7.0 * inch, height=3.4 * inch))
    if chart_note:
        elements.append(Spacer(1, 4))
        elements.append(Paragraph(chart_note, styles["Italic"]))
    elements.append(PageBreak())

    elements.append(Paragraph("Detailed Anomaly Sections", styles["Heading2"]))
    elements.append(Spacer(1, 6))

    has_any = False
    for series_name in series_cols:
        series_anoms = anomalies_by_series[series_name]
        flagged = series_anoms[series_anoms["Any Outlier"]].copy()

        elements.append(Paragraph(f"Series: {series_name}", styles["Heading3"]))

        if flagged.empty:
            elements.append(Paragraph("No anomalies detected for this series.", styles["Normal"]))
            elements.append(Spacer(1, 8))
            continue

        has_any = True
        for i, (_, row) in enumerate(flagged.iterrows(), start=1):
            elements.append(
                Paragraph(
                    f"Anomaly {i}: {row['Anomaly Type']} on {row['Date'].date()}",
                    styles["Normal"],
                )
            )

            detail_rows = [
                ["Field", "Value"],
                ["Value", f"{row['Value']:.2f}"],
                ["Detected by", row["Detected By"]],
                ["Z-score", f"{row['Z-score']:.3f}"],
                ["MA distance", f"{row['MA Distance'] * 100:.2f}%"],
            ]
            elements.append(_build_table(detail_rows, col_widths=[1.3 * inch, 2.3 * inch]))
            elements.append(Spacer(1, 6))

        elements.append(Spacer(1, 8))

    if not has_any:
        elements.append(Paragraph("No anomalies detected across all series.", styles["Normal"]))

    doc.build(elements)
    buffer.seek(0)

    try:
        os.remove(chart_path)
    except OSError:
        pass

    return buffer.getvalue(), chart_note


st.title("Mining Data Dashboard")
st.caption("Data engineering analytics for mine output series with configurable anomaly detection.")

st.sidebar.header("Data Source")
source_mode = st.sidebar.radio("Source", ["Google Sheet URL", "Upload CSV"], horizontal=False)

if source_mode == "Google Sheet URL":
    data_url = st.sidebar.text_input("CSV URL", value=DEFAULT_DATA_URL)
    try:
        df = load_sheet_data(data_url)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Failed to load data from URL: {exc}")
        st.stop()
else:
    uploaded_file = st.sidebar.file_uploader("Upload CSV with Date + mine columns", type=["csv"])
    if uploaded_file is None:
        st.info("Upload a CSV file to continue.")
        st.stop()
    try:
        df = normalize_data(pd.read_csv(uploaded_file))
    except Exception as exc:  # noqa: BLE001
        st.error(f"Failed to read uploaded CSV: {exc}")
        st.stop()

mine_columns = [col for col in df.columns if col != "Date"]
df["Total"] = df[mine_columns].sum(axis=1)
series_columns = mine_columns + ["Total"]

st.sidebar.header("Detection Settings")
z_thresh = st.sidebar.slider("Z-score threshold", 1.0, 5.0, 2.5, 0.1)
iqr_factor = st.sidebar.slider("IQR factor", 1.0, 4.0, 1.5, 0.1)
ma_window = st.sidebar.slider("Moving average window", 2, 60, 7, 1)
ma_thresh = st.sidebar.slider("Distance from MA (%)", 1, 100, 20, 1) / 100.0
grubbs_alpha = st.sidebar.slider("Grubbs alpha", 0.001, 0.20, 0.05, 0.001)

st.sidebar.header("Chart Settings")
selected_series = st.sidebar.selectbox("Series for detailed chart", series_columns)
chart_type = st.sidebar.selectbox("Chart type", ["Line", "Bar", "Stacked"])
trend_degree = st.sidebar.selectbox("Trend polynomial degree", [1, 2, 3, 4], index=1)

stats_records = []
anomaly_summary_records = []
anomalies_by_series: dict[str, pd.DataFrame] = {}
trend_by_series: dict[str, pd.Series] = {}

for series_name in series_columns:
    s = df[series_name]
    stat_map = compute_stats(s)
    stats_records.append({"Series": series_name, **stat_map})

    detected = detect_anomalies(
        s,
        z_thresh=z_thresh,
        iqr_factor=iqr_factor,
        ma_window=ma_window,
        ma_thresh=ma_thresh,
        grubbs_alpha=grubbs_alpha,
    )
    detected = detected.copy()
    detected["Date"] = df["Date"]

    anomalies_by_series[series_name] = detected
    trend_by_series[series_name] = compute_trend(s, trend_degree)

    flagged = detected[detected["Any Outlier"]]
    anomaly_summary_records.append(
        {
            "Series": series_name,
            "Z-score": int(detected["Z-score Flag"].sum()),
            "IQR": int(detected["IQR Flag"].sum()),
            "MA Distance": int(detected["MA Flag"].sum()),
            "Grubbs": int(detected["Grubbs Flag"].sum()),
            "Any Outlier": int(detected["Any Outlier"].sum()),
            "Spikes": int((flagged["Anomaly Type"] == "Spike").sum()),
            "Drops": int((flagged["Anomaly Type"] == "Drop").sum()),
        }
    )

stats_df = pd.DataFrame(stats_records).set_index("Series")
anomaly_summary_df = pd.DataFrame(anomaly_summary_records).set_index("Series")

st.subheader("Data Overview")
col1, col2, col3 = st.columns(3)
col1.metric("Rows", f"{len(df):,}")
col2.metric("Date range start", str(df["Date"].min().date()))
col3.metric("Date range end", str(df["Date"].max().date()))

st.subheader("Summary Statistics (Each Mine + Total)")
st.dataframe(
    stats_df[["Mean", "Std Dev", "Median", "IQR"]].round(2),
    use_container_width=True,
)

st.subheader("Anomaly Counts by Series")
st.dataframe(anomaly_summary_df, use_container_width=True)

selected_details = anomalies_by_series[selected_series].copy()
selected_details["Date"] = df["Date"]
selected_trend = trend_by_series[selected_series]

chart_fig = build_chart(
    df=df,
    mine_cols=mine_columns,
    selected_series=selected_series,
    chart_type=chart_type,
    trend_degree=trend_degree,
    trend_series=selected_trend,
    selected_anomalies=selected_details,
)
st.plotly_chart(chart_fig, use_container_width=True)

selected_anomalies = selected_details[selected_details["Any Outlier"]].copy()

st.subheader(f"Detailed Anomalies: {selected_series}")
if selected_anomalies.empty:
    st.info("No anomalies detected for the selected series with current settings.")
else:
    display_cols = ["Date", "Value", "Anomaly Type", "Detected By", "Z-score", "MA Distance"]
    table_df = selected_anomalies[display_cols].copy()
    table_df["Value"] = table_df["Value"].round(2)
    table_df["Z-score"] = table_df["Z-score"].round(3)
    table_df["MA Distance"] = (table_df["MA Distance"] * 100).round(2).astype(str) + "%"
    st.dataframe(table_df, use_container_width=True)

settings_map = {
    "Z-score threshold": z_thresh,
    "IQR factor": iqr_factor,
    "Moving average window": ma_window,
    "Distance from MA (%)": round(ma_thresh * 100, 2),
    "Grubbs alpha": grubbs_alpha,
}

if st.button("Generate PDF Report"):
    pdf_bytes, chart_note = create_pdf(
        df=df,
        mine_cols=mine_columns,
        series_cols=series_columns,
        selected_series=selected_series,
        chart_type=chart_type,
        trend_degree=trend_degree,
        settings=settings_map,
        stats_df=stats_df,
        anomaly_summary_df=anomaly_summary_df,
        anomalies_by_series=anomalies_by_series,
        plotly_fig=chart_fig,
        trend_series=selected_trend,
    )
    st.session_state["pdf_bytes"] = pdf_bytes
    st.session_state["pdf_ready"] = True
    if chart_note:
        st.warning(chart_note)
    else:
        st.success("PDF report generated successfully.")

if st.session_state.get("pdf_ready"):
    st.download_button(
        label="Download PDF",
        data=st.session_state["pdf_bytes"],
        file_name="mining_report.pdf",
        mime="application/pdf",
    )
