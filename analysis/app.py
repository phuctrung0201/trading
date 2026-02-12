"""Shiny app for OHLCV analysis.

Run with:
    shiny run --app-dir analysis app:app
"""

import os

import numpy as np
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from shiny import App, reactive, render, ui

# ---------------------------------------------------------------------------
# Discover CSV files in data/
# ---------------------------------------------------------------------------

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
_csv_files = sorted(
    f for f in os.listdir(DATA_DIR)
    if f.endswith(".csv") and "book" not in f
) if os.path.isdir(DATA_DIR) else []

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.h4("Data"),
        ui.input_select("file", "CSV file", choices=_csv_files, selected=_csv_files[-1] if _csv_files else None),
        ui.hr(),

        ui.h4("EMA"),
        ui.input_checkbox("show_ema", "Show EMA", value=True),
        ui.input_numeric("ema_short", "Short period", value=10, min=2, max=500),
        ui.input_numeric("ema_long", "Long period", value=20, min=2, max=500),

        ui.hr(),
        ui.h4("Volume"),
        ui.input_checkbox("show_vol_ma", "Show Rolling Volume", value=False),
        ui.input_numeric("vol_ma_length", "Period", value=20, min=2, max=500),

        ui.hr(),
        ui.h4("Date range"),
        ui.output_ui("date_range_ui"),

        width=280,
    ),

    ui.navset_card_tab(
        ui.nav_panel("Chart", ui.output_ui("chart_ui")),
        ui.nav_panel("Statistics", ui.output_ui("stats_ui")),
    ),

    title="OHLCV Analysis",
)

# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

def server(input, output, session):

    @reactive.calc
    def raw_data():
        path = os.path.join(DATA_DIR, input.file())
        df = pd.read_csv(path, parse_dates=["timestamp"], index_col="timestamp")
        df.columns = [c.strip().lower() for c in df.columns]
        for col in ("open", "high", "low", "close", "volume"):
            df[col] = pd.to_numeric(df[col])
        return df

    @output
    @render.ui
    def date_range_ui():
        df = raw_data()
        mn, mx = df.index.min().date(), df.index.max().date()
        return ui.input_date_range("dates", "Range", start=mn, end=mx, min=mn, max=mx)

    @reactive.calc
    def data():
        df = raw_data()
        if input.dates() is not None:
            start, end = input.dates()
            df = df.loc[str(start):str(end)]
        return df

    # ---- Chart ----------------------------------------------------------

    @output
    @render.ui
    def chart_ui():
        df = data()
        if df.empty:
            return ui.p("No data in selected range.")

        # Determine subplot layout
        row_specs = [
            [{"secondary_y": False}],  # candlestick
            [{"secondary_y": False}],  # volume
        ]
        row_names = ["Price", "Volume"]
        heights = [5, 1]

        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=heights,
            subplot_titles=row_names,
            specs=row_specs,
        )

        # -- Candlestick --
        fig.add_trace(go.Candlestick(
            x=df.index, open=df["open"], high=df["high"],
            low=df["low"], close=df["close"], name="OHLC",
            increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
        ), row=1, col=1)

        # -- EMA (MACross signal) --
        if input.show_ema():
            ema_s = ta.ema(df["close"], length=input.ema_short())
            ema_l = ta.ema(df["close"], length=input.ema_long())
            fig.add_trace(go.Scatter(
                x=df.index, y=ema_s, name=f"EMA {input.ema_short()}",
                line=dict(width=1, color="#2196F3"),
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=df.index, y=ema_l, name=f"EMA {input.ema_long()}",
                line=dict(width=1, color="#FF9800"),
            ), row=1, col=1)

        # -- Volume --
        colors = np.where(df["close"] >= df["open"], "#26a69a", "#ef5350")
        fig.add_trace(go.Bar(
            x=df.index, y=df["volume"], name="Volume",
            marker_color=colors, opacity=0.7,
        ), row=2, col=1)

        if input.show_vol_ma():
            vol_ma = df["volume"].rolling(window=input.vol_ma_length()).mean()
            fig.add_trace(go.Scatter(
                x=df.index, y=vol_ma, name=f"Vol MA({input.vol_ma_length()})",
                line=dict(width=1.5, color="#FFD54F"),
            ), row=2, col=1)

        # -- Layout --
        total_height = 800
        fig.update_layout(
            height=total_height,
            template="plotly_dark",
            xaxis_rangeslider_visible=False,
            showlegend=True,
            legend=dict(orientation="h", y=1.02, x=0.5, xanchor="center"),
            margin=dict(l=50, r=20, t=60, b=30),
        )
        fig.update_xaxes(type="date")

        return ui.HTML(fig.to_html(full_html=False, include_plotlyjs="cdn"))

    # ---- Statistics -----------------------------------------------------

    @output
    @render.ui
    def stats_ui():
        df = data()
        if df.empty:
            return ui.p("No data.")

        close = df["close"]
        returns = close.pct_change().dropna()
        duration = df.index[-1] - df.index[0]

        stats = {
            "Period": f"{df.index[0].date()} → {df.index[-1].date()}",
            "Duration": str(duration),
            "Bars": f"{len(df):,}",
            "Open": f"{close.iloc[0]:,.2f}",
            "Close": f"{close.iloc[-1]:,.2f}",
            "Change": f"{(close.iloc[-1] / close.iloc[0] - 1) * 100:+.2f}%",
            "High": f"{df['high'].max():,.2f}",
            "Low": f"{df['low'].min():,.2f}",
            "Avg Volume": f"{df['volume'].mean():,.2f}",
            "Volatility (daily)": f"{returns.std() * np.sqrt(1440):.4f}" if len(returns) > 1 else "N/A",
            "Mean Return": f"{returns.mean():.6f}",
            "Sharpe (annualized)": f"{returns.mean() / returns.std() * np.sqrt(365 * 1440):.4f}" if returns.std() > 0 else "N/A",
            "Max Drawdown": _max_drawdown(close),
        }

        rows = [ui.tags.tr(ui.tags.td(k, style="font-weight:600; padding:6px 12px;"),
                           ui.tags.td(v, style="padding:6px 12px;"))
                for k, v in stats.items()]

        return ui.tags.table(
            *rows,
            style="border-collapse:collapse; width:100%; max-width:500px;",
        )


def _max_drawdown(close: pd.Series) -> str:
    peak = close.cummax()
    dd = (close - peak) / peak
    max_dd = dd.min()
    return f"{max_dd * 100:.2f}%"


# ---------------------------------------------------------------------------

app = App(app_ui, server)
