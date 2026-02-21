import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from ..colors import colors_palette
import math

def plot_numerical_distributions(
    df,
    numerical_columns,
    n_cols=3,
    figsize=(15, 15),
    dpi=120,
    title=None
):
    """
    Plot numerical distributions with histogram and KDE.

    Generates a grid of styled histograms with kernel density
    estimation overlays for the specified numerical columns.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the numerical variables.

    numerical_columns : list of str
        List of numerical column names to visualize.

    n_cols : int, default=3
        Number of subplot columns.

    figsize : tuple, default=(15, 10)
        Size of the entire figure in inches.

    dpi : int, default=120
        Resolution of the figure in dots per inch.

    title : str or None, default=None
        General title for the entire figure.

    Returns
    -------
    matplotlib.figure.Figure
        The generated matplotlib figure.
    """

    n_rows = int(np.ceil(len(numerical_columns) / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, dpi=dpi)
    axes = axes.flatten()

    for i, col in enumerate(numerical_columns):

        sns.histplot(
            df[col].dropna(),
            bins=30,
            kde=True,
            ax=axes[i],
            color=colors_palette["steel_blue"],
            alpha=0.7
        )

        axes[i].set_title(col, fontsize=13, fontweight="bold")
        axes[i].set_ylabel("Frequency")

        axes[i].grid(False)
        for spine in axes[i].spines.values():
            spine.set_visible(False)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.subplots_adjust(hspace=0.25)

    if title is not None:
        fig.suptitle(title, fontsize=16, fontweight="bold", y=1.02)

    return fig

def plot_dual_histogram_comparison(
    df,
    column_1,
    column_2,
    label_1,
    label_2,
    x_label,
    bins=25,
    figsize=(16, 6),
    dpi=120,
    title=None
):
    """
    Plot side-by-side histogram and KDE comparison between two variables.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing both variables.

    column_1 : str
        First numerical column name.

    column_2 : str
        Second numerical column name.

    label_1 : str
        Display label for the first variable.

    label_2 : str
        Display label for the second variable.

    x_label : str
        Label for the x-axis.

    bins : int, default=25
        Number of histogram bins.

    figsize : tuple, default=(16, 6)
        Figure size in inches.

    dpi : int, default=120
        Figure resolution.

    title : str or None, default=None
        General title for the figure.

    Returns
    -------
    matplotlib.figure.Figure
        The generated matplotlib figure.
    """

    fig, axs = plt.subplots(1, 2, figsize=figsize, dpi=dpi)
    fig.subplots_adjust(wspace=0.35)
    
    axs[0].hist(
        df[column_1].dropna(),
        bins=bins,
        density=True,
        alpha=0.45,
        color=colors_palette["steel_blue"],
        label=label_1
    )

    axs[0].hist(
        df[column_2].dropna(),
        bins=bins,
        density=True,
        alpha=0.45,
        color=colors_palette["coral"],
        label=label_2
    )

    axs[0].set_title("Histogram", fontsize=13, fontweight="bold")
    axs[0].set_xlabel(x_label)
    axs[0].set_ylabel("Density")
    axs[0].legend()

    axs[0].grid(False)
    for spine in axs[0].spines.values():
        spine.set_visible(False)

    sns.kdeplot(
        df[column_1].dropna(),
        ax=axs[1],
        color=colors_palette["steel_blue"],
        linewidth=2,
        label=label_1
    )

    sns.kdeplot(
        df[column_2].dropna(),
        ax=axs[1],
        color=colors_palette["coral"],
        linewidth=2,
        label=label_2
    )

    axs[1].set_title("Kernel Density Estimation", fontsize=13, fontweight="bold")
    axs[1].set_xlabel(x_label)
    axs[1].set_ylabel("Density")
    axs[1].legend()

    axs[1].grid(False)
    for spine in axs[1].spines.values():
        spine.set_visible(False)

    if title is not None:
        fig.suptitle(title, fontsize=16, fontweight="bold", y=1.05)

    return fig


def plot_group_distribution_comparison(
    columns,
    group0,
    label0,
    group1,
    label1,
    bins=20,
    n_cols=3,
    figsize_base=(18, 5),
    dpi=120,
    title=None
):

    n_vars = len(columns)
    n_rows = math.ceil(n_vars / n_cols)

    fig, axs = plt.subplots(
        n_rows,
        n_cols,
        figsize=(figsize_base[0], figsize_base[1] * n_rows),
        dpi=dpi
    )

    # Asegurar que axs siempre sea iterable
    if n_rows == 1 and n_cols == 1:
        axs = [axs]
    elif n_rows == 1 or n_cols == 1:
        axs = np.array(axs).reshape(-1)
    else:
        axs = axs.flatten()

    for i, col in enumerate(columns):

        ax = axs[i]

        data_0 = group0[col].dropna()
        data_1 = group1[col].dropna()

        mean_0 = data_0.mean()
        mean_1 = data_1.mean()

        ax.hist(data_0, bins=bins, alpha=0.6,
                color=colors_palette["steel_blue"], label=label0)

        ax.hist(data_1, bins=bins, alpha=0.6,
                color=colors_palette["coral"], label=label1)

        ax.axvline(mean_0, linestyle="--", linewidth=2,
                   color=colors_palette["steel_blue"])

        ax.axvline(mean_1, linestyle="--", linewidth=2,
                   color=colors_palette["coral"])

        ax.set_title(col, fontsize=13, fontweight="bold")
        ax.set_ylabel("Frequency")
        ax.legend()

        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)

    # Eliminar ejes sobrantes
    for j in range(len(columns), len(axs)):
        fig.delaxes(axs[j])

    if title is not None:
        fig.suptitle(title, fontsize=16, fontweight="bold")

    fig.tight_layout(rect=[0, 0, 1, 0.96])

    return fig