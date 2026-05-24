"""
Runner for the Matplotlib NPIS dashboard.

This imports the computed indicators from analysis.py (kept untouched) and
renders the core decision-maker views.
"""

import matplotlib.pyplot as plt
import seaborn as sns

import dashboard


def main():
    sns.set_style("whitegrid")

    district_df, state_df, combined_df = dashboard.prepare_data()

    # Core views
    dashboard.plot_state_rankings(state_df, "NPI", top_n=15, ascending=False)
    dashboard.plot_district_rankings(district_df, "PDI", state=None, top_n=20, ascending=False)
    dashboard.plot_heatmap(
        state_df,
        indicators=["PDI", "Education_PDI", "Transport_PDI", "Utilities_PDI", "Jobs_PDI", "NPI"],
    )
    dashboard.plot_top_bottom(district_df, "Transport_PDI", n=10)
    dashboard.plot_state_vs_district_peaks(combined_df, "Education_PDI", top_n=12)

    plt.show()


if __name__ == "__main__":
    main()

