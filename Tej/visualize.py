# ----------------------------------------------------------------------
#  Simple visualisation utilities  (add this near the top of main.py)
# ----------------------------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _heatmap(df: pd.DataFrame, *, figsize=(8, 6)) -> None:
    """
    Plot a correlation heat-map of all numeric columns in *df*,
    with the correlation coefficients annotated in each cell.
    """
    corr = df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(corr, aspect="auto", vmin=-1, vmax=1)
    ax.set_xticks(np.arange(len(corr.columns)), labels=corr.columns, rotation=90)
    ax.set_yticks(np.arange(len(corr.columns)), labels=corr.columns)
    ax.set_title("Correlation heat-map")
    fig.colorbar(im, ax=ax)

    # ── NEW: write the numbers ────────────────────────────────────
    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            val = corr.iat[i, j]
            ax.text(
                j, i,
                f"{val:+.2f}",               # + sign keeps ±
                ha="center", va="center",
                color="white" if abs(val) > 0.5 else "black",
                fontsize=8,
            )

    fig.tight_layout()

def _histograms(
    df: pd.DataFrame,
    *,
    bins: int = 30,
    figsize=(10, 8),
    ncols: int = 3,
) -> None:
    """
    Draw one histogram per numeric column.

    Parameters
    ----------
    bins : int
        Number of bins for each histogram.
    ncols : int
        How many histogram panes per row.
    """
    num_df = df.select_dtypes(include="number")
    if num_df.empty:
        print("No numeric columns to plot.")
        return

    nrows = -(-num_df.shape[1] // ncols)  # ceiling division
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes = axes.flatten()

    for i, col in enumerate(num_df.columns):
        axes[i].hist(num_df[col].dropna(), bins=bins)
        axes[i].set_title(col)

    # Hide any unused sub-plots
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Histograms")
    fig.tight_layout()


def visualize(df: pd.DataFrame) -> None:
    """
    Convenience wrapper – call once and you get both plots.
    """
    _heatmap(df)
    _histograms(df)
    plt.show()
# ----------------------------------------------------------------------
#  End of visualisation utilities