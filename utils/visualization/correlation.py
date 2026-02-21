import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from ..colors import colors_palette

def plot_correlation_heatmap(
    df,
    numerical_columns,
    figsize=(10, 8),
    dpi=120,
    title="Correlation Matrix"
):
    """
    Plot a styled lower-triangle correlation heatmap.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing numerical variables.

    numerical_columns : list of str
        List of numerical columns to compute correlation.

    figsize : tuple, default=(10, 8)
        Figure size in inches.

    dpi : int, default=120
        Resolution of the figure.

    title : str, default="Correlation Matrix"
        Title of the plot.

    Returns
    -------
    matplotlib.figure.Figure
        The generated matplotlib figure.
    """

    corr = df[numerical_columns].corr()

    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    sns.heatmap(
        corr,
        mask=mask,
        cmap="RdBu_r",         
        vmin=-1,
        vmax=1,
        center=0,
        annot=True,
        linewidths=0, 
        fmt=".2f",
        square=True,
        cbar_kws={"shrink": 0.8},
        ax=ax
    )

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.tick_params(axis='x', rotation=45)
    ax.tick_params(axis='y', rotation=0)

    fig.tight_layout()

    return fig

def plot_high_correlation_scatter(
    df,
    correlation_matrix,
    threshold=0.5,
    n_cols=4,
    figsize_per_plot=4,
    dpi=120,
    hspace=0.6,
    title=None
):
    """
    Plot scatter plots for variable pairs with high correlation.

    Identifies pairs of variables whose absolute correlation
    is greater than or equal to the specified threshold and
    generates a styled grid of scatter plots with regression lines.

    Parameters
    ----------
    df : pandas.DataFrame
        Original DataFrame containing the variables.

    correlation_matrix : pandas.DataFrame
        Precomputed correlation matrix of the variables.

    threshold : float, default=0.5
        Minimum absolute correlation value (|r|) required
        to include a variable pair.

    n_cols : int, default=4
        Number of columns in the subplot grid.

    figsize_per_plot : int or float, default=4
        Size (in inches) allocated to each subplot.

    dpi : int, default=120
        Resolution of the figure in dots per inch.

    hspace : float, default=0.6
        Vertical spacing between subplot rows.

    title : str or None, default=None
        General title for the entire figure.

    Returns
    -------
    list of tuple
        List of tuples (col_x, col_y, r_value).
    """
    
    corr_pairs = []
    cols = correlation_matrix.columns

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            val = correlation_matrix.iloc[i, j]
            if abs(val) >= threshold:
                corr_pairs.append((cols[i], cols[j], val))

    if not corr_pairs:
        print(f"No pairs found with |r| >= {threshold}")
        return []

    n_plots = len(corr_pairs)
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(n_cols * figsize_per_plot, n_rows * figsize_per_plot),
        dpi=dpi
    )

    if n_plots == 1:
        axes = np.array([axes])

    axes = axes.flatten()

    for i, (col_x, col_y, val) in enumerate(corr_pairs):

        sns.regplot(
            data=df,
            x=col_x,
            y=col_y,
            ax=axes[i],
            scatter_kws={"alpha": 0.6, "s": 30, "color": "steelblue"},
            line_kws={"color": "darkred", "linewidth": 1.5},
            ci=None
        )

        axes[i].set_title(
            f"{col_x} vs {col_y}\n$r = {val:.2f}$",
            fontsize=11,
            fontweight="bold"
        )

        axes[i].grid(False)
        for spine in axes[i].spines.values():
            spine.set_visible(False)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.subplots_adjust(hspace=hspace)

    if title is not None:
        fig.suptitle(
            title,
            fontsize=16,
            fontweight="bold",
            y=1.02
        )

    return corr_pairs