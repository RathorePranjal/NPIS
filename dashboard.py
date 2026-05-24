"""
Matplotlib-based visualization layer for NPIS outputs.

This module imports the computed indicators from `analysis.py` and provides
Python-level controls for sorting/filtering along with policy-grade charts.

Core analysis logic remains in `analysis.py`; this file only reads its outputs.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import analysis

# Indicators available from analysis.py
INDICATORS = [
    "PDI",
    "Education_PDI",
    "Transport_PDI",
    "Utilities_PDI",
    "Jobs_PDI",
    "NPI",
]


def prepare_data():
    """Build district, state, and combined views from analysis outputs."""
    base = analysis.aadhaar.copy()

    district_df = base[["state", "district"] + INDICATORS].dropna(subset=["district"])

    # State-level mean scores across districts
    state_df = (
        district_df.groupby("state", as_index=False)[INDICATORS]
        .mean()
        .rename(columns={col: f"{col}_state_avg" for col in INDICATORS})
    )

    # District peaks per state for comparison against state averages
    district_peaks = (
        district_df.groupby("state", as_index=False)[INDICATORS]
        .max()
        .rename(columns={col: f"{col}_district_peak" for col in INDICATORS})
    )

    combined_df = pd.merge(state_df, district_peaks, on="state", how="inner")

    return district_df, state_df, combined_df


def filter_districts(
    df: pd.DataFrame,
    indicator: str,
    state: str | None = None,
    min_threshold: float | None = None,
    top_n: int | None = None,
    ascending: bool = False,
) -> pd.DataFrame:
    """Apply state/threshold filters and sorting for district-level views."""
    if indicator not in df.columns:
        raise ValueError(f"Indicator {indicator} not found in dataframe.")

    data = df.copy()
    if state:
        data = data[data["state"] == state]
    if min_threshold is not None:
        data = data[data[indicator] >= min_threshold]

    data = data.sort_values(by=indicator, ascending=ascending)

    if top_n:
        data = data.head(top_n)

    return data


def _annotate_bars(ax):
    """Add value labels on top of bars."""
    for patch in ax.patches:
        height = patch.get_height()
        ax.annotate(
            f"{height:.3f}",
            (patch.get_x() + patch.get_width() / 2, height),
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=0,
        )


def _apply_zoom_range(ax, values: pd.Series, axis: str = "x", pad_frac: float = 0.05):
    """
    Zoom the plot to the data range so small decimal differences are visible.
    pad_frac adds a small margin around min/max.
    """
    if values.empty:
        return

    vmin = values.min()
    vmax = values.max()
    if vmin == vmax:
        # Avoid zero-width range
        pad = abs(vmin) * pad_frac if vmin != 0 else pad_frac
        vmin -= pad
        vmax += pad
    else:
        pad = (vmax - vmin) * pad_frac
        vmin -= pad
        vmax += pad

    if axis == "x":
        ax.set_xlim(vmin, vmax)
    elif axis == "y":
        ax.set_ylim(vmin, vmax)


def plot_state_rankings(
    state_df: pd.DataFrame,
    indicator: str,
    top_n: int | None = 15,
    ascending: bool = False,
    zoom: bool = True,
    pad_frac: float = 0.05,
):
    """Horizontal bar chart of state-wise rankings."""
    col = f"{indicator}_state_avg" if f"{indicator}_state_avg" in state_df.columns else indicator
    data = state_df.sort_values(by=col, ascending=ascending)
    if top_n:
        data = data.head(top_n) if ascending else data.tail(top_n)

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(
        data=data,
        y="state",
        x=col,
        palette="RdYlGn_r",
    )
    _annotate_bars(ax)
    if zoom:
        _apply_zoom_range(ax, data[col], axis="x", pad_frac=pad_frac)
    title_order = "Ascending" if ascending else "Descending"
    ax.set_title(f"{indicator} — State Ranking ({title_order})", fontsize=12, weight="bold")
    ax.set_xlabel(f"{indicator} score")
    ax.set_ylabel("State")
    plt.tight_layout()
    return ax


def plot_district_rankings(
    district_df: pd.DataFrame,
    indicator: str,
    state: str | None = None,
    top_n: int | None = 20,
    min_threshold: float | None = None,
    ascending: bool = False,
    zoom: bool = True,
    pad_frac: float = 0.05,
):
    """Sortable district-level bar chart with optional state filter."""
    filtered = filter_districts(
        district_df,
        indicator=indicator,
        state=state,
        min_threshold=min_threshold,
        top_n=top_n,
        ascending=ascending,
    )

    plt.figure(figsize=(12, 7))
    ax = sns.barplot(
        data=filtered,
        x=indicator,
        y="district",
        hue="state" if state is None else None,
        dodge=False,
        palette="RdYlGn_r",
    )
    _annotate_bars(ax)
    if zoom:
        _apply_zoom_range(ax, filtered[indicator], axis="x", pad_frac=pad_frac)
    ax.set_title(
        f"{indicator} — District Ranking" + (f" ({state})" if state else ""),
        fontsize=12,
        weight="bold",
    )
    ax.set_xlabel(f"{indicator} score")
    ax.set_ylabel("District")
    plt.tight_layout()
    return ax


def plot_heatmap(state_df: pd.DataFrame, indicators: list[str] | None = None):
    """Heatmap of states vs indicators."""
    indicators = indicators or INDICATORS
    cols = [f"{col}_state_avg" if f"{col}_state_avg" in state_df.columns else col for col in indicators]

    data = state_df.set_index("state")[cols]
    plt.figure(figsize=(12, max(6, len(data) * 0.25)))
    ax = sns.heatmap(
        data,
        cmap="RdYlGn_r",
        annot=True,
        fmt=".3f",
        cbar_kws={"label": "Score"},
    )
    ax.set_title("State × Indicator Heatmap", fontsize=12, weight="bold")
    ax.set_xlabel("Indicator")
    ax.set_ylabel("State")
    plt.tight_layout()
    return ax


def plot_top_bottom(district_df: pd.DataFrame, indicator: str, n: int = 10):
    """Side-by-side top-N and bottom-N district view."""
    sorted_df = district_df.sort_values(by=indicator, ascending=False)
    top = sorted_df.head(n)
    bottom = sorted_df.tail(n).sort_values(by=indicator, ascending=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharex=False)

    sns.barplot(data=top, x=indicator, y="district", hue="state", palette="RdYlGn_r", ax=axes[0], dodge=False)
    axes[0].set_title(f"Top {n} Districts — {indicator}", weight="bold")
    _annotate_bars(axes[0])

    sns.barplot(data=bottom, x=indicator, y="district", hue="state", palette="RdYlGn", ax=axes[1], dodge=False)
    axes[1].set_title(f"Bottom {n} Districts — {indicator}", weight="bold")
    _annotate_bars(axes[1])

    for ax in axes:
        _apply_zoom_range(ax, top[indicator] if ax is axes[0] else bottom[indicator], axis="x", pad_frac=0.05)
        ax.set_xlabel(f"{indicator} score")
        ax.set_ylabel("District")

    plt.tight_layout()
    return axes


def plot_state_vs_district_peaks(
    combined_df: pd.DataFrame,
    indicator: str,
    top_n: int | None = 15,
    ascending: bool = False,
    zoom: bool = True,
    pad_frac: float = 0.05,
):
    """Compare state averages with their peak districts for an indicator."""
    avg_col = f"{indicator}_state_avg"
    peak_col = f"{indicator}_district_peak"
    data = combined_df.sort_values(by=avg_col, ascending=ascending)
    if top_n:
        data = data.head(top_n) if ascending else data.tail(top_n)

    x = range(len(data))
    plt.figure(figsize=(12, 6))
    plt.bar(x, data[avg_col], width=0.4, label="State Avg", color="#4daf4a")
    plt.bar([i + 0.4 for i in x], data[peak_col], width=0.4, label="District Peak", color="#e41a1c")

    plt.xticks([i + 0.2 for i in x], data["state"], rotation=45, ha="right")
    plt.xlabel("State")
    plt.ylabel(f"{indicator} score")
    plt.title(f"{indicator}: State Average vs District Peak", fontsize=12, weight="bold")
    plt.legend()
    if zoom:
        _apply_zoom_range(plt.gca(), pd.concat([data[avg_col], data[peak_col]]), axis="y", pad_frac=pad_frac)
    plt.tight_layout()
    return plt.gca()


def demo():
    """Example usage to generate core views."""
    sns.set_style("whitegrid")

    district_df, state_df, combined_df = prepare_data()

    plot_state_rankings(state_df, "NPI", top_n=15, ascending=False)
    plot_district_rankings(district_df, "PDI", state=None, top_n=20, ascending=False)
    plot_heatmap(state_df, indicators=["PDI", "Education_PDI", "Transport_PDI", "Utilities_PDI", "Jobs_PDI", "NPI"])
    plot_top_bottom(district_df, "Transport_PDI", n=10)
    plot_state_vs_district_peaks(combined_df, "Education_PDI", top_n=12)

    plt.show()


if __name__ == "__main__":
    demo()

