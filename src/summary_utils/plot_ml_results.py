"""Plot ML results visualization module"""

import os
import sys
from pathlib import Path

# Add the necessary directories to sys.path
current_dir = Path(__file__).resolve().parent
src_dir = current_dir.parent
for path in [str(current_dir), str(src_dir)]:
    if path not in sys.path:
        sys.path.append(path)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Tuple
import logging
from plot_style import (
    set_default_style,
    get_figure_size,
    style_ml_metrics_plot,
    set_color_scheme
)

# logger setup
logger = logging.getLogger(__name__)

def plot_time_variance_feature_importance(
    csv_path: Path = Path("output_data/time_variance/time_variance_importance.csv"),
    save_dir: Path = Path("output_data/plots/feature_importance"),
    outcomes: list[str] | None = None,
    n_categories: int = 8,
) -> None:
    """
    Plot feature category importance through time for each outcome and model.
    
    Parameters
    ----------
    csv_path : Path
        Path to the CSV file containing time variance importance data
    save_dir : Path
        Directory to save the generated plots
    outcomes : list[str] | None
        List of outcomes to plot. If None, uses all outcomes except excluded ones
    n_categories : int
        Number of top feature categories to include in plots
    """
    # Create save directory if it doesn't exist
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Read and prepare data
    df = pd.read_csv(csv_path)

    # ── keep only one row per category/year/outcome/model ──
    df = df.drop_duplicates(
        subset=[
            "feature_category",
            "year",
            "outcome",
            "model",
            "category_relative_importance",
        ]
    )
    
    # Drop excluded outcomes
    exclude = {"y_EPSNormalized_surprise", "y_revenue_surprise"}
    df = df[~df["outcome"].isin(exclude)]
    
    if outcomes is not None:
        df = df[df["outcome"].isin(outcomes)]
    
    # Get unique outcomes and models
    outcomes_to_plot = sorted(df["outcome"].unique())
    models = sorted(df["model"].unique())
    
    # Set plot style
    set_default_style()
    
    # Plot for each outcome
    for out in outcomes_to_plot:
        # Get top N categories for this outcome
        top_cats = (
            df[df["outcome"] == out]
            .groupby("feature_category")["feature_relative_importance"]
            .mean()
            .nlargest(n_categories)
            .index
        )
        sub = df[(df["outcome"] == out) & (df["feature_category"].isin(top_cats))]
        
        if sub.empty:
            logger.warning(f"No data to plot for outcome: {out}")
            continue

        # ------------------------------------------------------------------ #
        # NEW — loop over models and create a separate figure for each one
        # ------------------------------------------------------------------ #
        for mdl in models:
            tmp = sub[sub["model"] == mdl].copy()
            if tmp.empty:
                logger.warning(f"No data to plot for outcome {out}, model {mdl}")
                continue

            # ── make sure every category (1-8) has data for each year ──
            tmp["year_numeric"] = tmp["year"].astype(int)
            years_sorted = sorted(tmp["year_numeric"].unique())
            full_year_idx = pd.Index(years_sorted, name="year_numeric")

            plot_vals = {}
            for cat in range(1, 9):
                ser = (
                    tmp[tmp["feature_category"] == cat]
                    .set_index("year_numeric")["feature_relative_importance"]
                    .reindex(full_year_idx, fill_value=0)
                )
                if not ser.empty:  # Only add series if it has values
                    plot_vals[cat] = ser
            
            if not plot_vals:
                logger.warning(f"No valid categories to plot for outcome {out}, model {mdl}")
                continue

            # ── single-axis figure ──
            fig, ax = plt.subplots(figsize=(8, 6), dpi=120)

            for cat, ser in plot_vals.items():
                ax.plot(ser.index, ser.values, marker="o", label=str(cat))

            # dynamic y-limits (±5 %)
            vals = np.concatenate([v.values for v in plot_vals.values()])
            if len(vals) > 0:  # Only set limits if we have values
                y_min, y_max = vals.min(), vals.max()
                if y_min == y_max:  # flat line edge-case
                    y_min = max(0, y_min - 0.05)
                    y_max = y_max + 0.05
                margin = 0.05 * (y_max - y_min) if y_max > y_min else 0.05
                ax.set_ylim(max(0, y_min - margin),
                           y_max + margin)

            ax.set_title(f"{out} · {mdl}")
            ax.set_xlabel("Year")
            ax.set_ylabel("Feature Relative Importance")
            ax.set_xticks(years_sorted)
            ax.set_xticklabels(years_sorted, rotation=45, ha="right")
            ax.legend(title="Category", bbox_to_anchor=(1.02, 0.5), loc="center left")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()

            # ── save ──
            out_path = (
                save_dir / out / f"{out}_{mdl}_feature_importance_time.png"
            )
            out_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"Saving plot to {out_path}")
            fig.savefig(out_path, bbox_inches="tight")
            plt.close(fig)
        # ------------------------------------------------------------------ #

def plot_time_variance_model_performance(
    csv_path: Path = Path("output_data/time_variance/time_variance_performance.csv"),
    save_dir: Path = Path("output_data/plots/time_variance/model_performance"),
    outcomes: list[str] | None = None,
    metrics_cv: tuple[str, ...] = ("r2", "mse", "rmse", "mae"),
    metrics_oos: tuple[str, ...] = ("r2", "rmse"),
) -> None:
    """
    For each *outcome* (excluding EPS- and revenue-surprise by default)
    create time-series plots that track every model's performance.

    • Separate figures for CV and OOS results.
    • Within each dataset, one figure per metric.
    • Only the tree models (LightGBM, CatBoost, XGBoost) are plotted.
    """
    df = pd.read_csv(csv_path)
    df = df.drop_duplicates()   # guarantee one line per (year, outcome, model, dataset)

    # ── keep only tree-based models ──
    tree_models = {"lgbm", "catboost", "xgb"}
    df = df[df["model"].isin(tree_models)]

    # drop outcomes we do NOT want
    exclude = {"y_EPSNormalized_surprise", "y_revenue_surprise"}
    df = df[~df["outcome"].isin(exclude)]

    if outcomes is not None:
        df = df[df["outcome"].isin(outcomes)]

    save_dir.mkdir(parents=True, exist_ok=True)

    # Set plot style for better readability
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['figure.figsize'] = (10, 6)  # Wider figure
    plt.rcParams['figure.dpi'] = 120
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10

    for out in df["outcome"].unique():
        sub_out = df[df["outcome"] == out]

        for dataset, metrics in (("cv", metrics_cv), ("oos", metrics_oos)):
            sub_ds = sub_out[sub_out["dataset"] == dataset]
            if sub_ds.empty:
                continue

            models = sorted(sub_ds["model"].unique())
            years = sorted(sub_ds["year"].astype(int).unique())
            
            # Calculate year spacing
            year_range = max(years) - min(years)
            if year_range > 10:
                # Show every other year if range is large
                xticks = years[::2]
                xtick_labels = [str(y) for y in years[::2]]
            else:
                xticks = years
                xtick_labels = [str(y) for y in years]

            for met in metrics:
                fig, ax = plt.subplots(figsize=(10, 6))

                # Plot lines with different line styles and markers for better distinction
                line_styles = ['-', '--', '-.', ':']
                markers = ['o', 's', '^', 'D', 'v']
                
                for i, mdl in enumerate(models):
                    # make sure the line is monotonic in time
                    grp = (
                        sub_ds[sub_ds["model"] == mdl]
                        .assign(year_numeric=lambda x: x["year"].astype(int))
                        .sort_values("year_numeric")
                    )
                    line_style = line_styles[i % len(line_styles)]
                    marker = markers[i % len(markers)]
                    
                    ax.plot(
                        grp["year_numeric"],
                        grp[met],
                        linestyle=line_style,
                        marker=marker,
                        markersize=6,
                        label=mdl,
                        linewidth=2
                    )

                # Customize title and labels
                ax.set_title(f"{out}\n{met.upper()} ({dataset.upper()})", pad=20, fontsize=14)
                ax.set_xlabel("Year", fontsize=12, labelpad=10)
                ax.set_ylabel(met.upper(), fontsize=12, labelpad=10)
                
                # Set x-axis ticks and rotation
                ax.set_xticks(xticks)
                ax.set_xticklabels(xtick_labels, rotation=45, ha='right')
                
                # Ensure x-axis limits include all years with padding
                ax.set_xlim(min(years) - 0.5, max(years) + 0.5)
                
                # Add grid but make it subtle
                ax.grid(True, alpha=0.3)
                
                # Place legend outside to the right
                ax.legend(
                    bbox_to_anchor=(1.02, 0.5),
                    loc="center left",
                    borderaxespad=0,
                    frameon=True,
                    fancybox=True,
                    shadow=True
                )

                # Adjust layout to prevent label cutoff
                fig.tight_layout()

                # Save the plot
                out_fn = (
                    save_dir
                    / dataset
                    / out
                    / f"{out}_{dataset}_{met}.png"
                )
                out_fn.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(out_fn, bbox_inches="tight", dpi=120)
                plt.close(fig)

if __name__ == "__main__":
    plot_time_variance_feature_importance()
    plot_time_variance_model_performance()

