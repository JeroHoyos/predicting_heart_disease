import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
from ..colors import colors_palette
import math
import numpy as np


def plot_binary_donuts(
    df,
    selected_columns,
    n_cols=3,
    figsize=None,
    colors=None,
    dpi=120,
    title=None
):
    """
    Plot multiple binary variables as donut charts.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the binary variables.

    selected_columns : list of str
        List of column names to visualize.

    n_cols : int, default=3
        Number of subplot columns.

    figsize : tuple, optional
        Figure size in inches (width, height).

    colors : list, optional
        Custom colors for the categories.

    dpi : int, default=120
        Resolution of the figure in dots per inch.

    title : str, optional
        Global title of the figure.

    Returns
    -------
    matplotlib.figure.Figure
        The generated matplotlib figure.
    """

    n_vars = len(selected_columns)
    n_rows = math.ceil(n_vars / n_cols)

    if figsize is None:
        figsize = (5 * n_cols, 4 * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, dpi=dpi)

    axes = np.array(axes).reshape(-1)

    if colors is None:
        colors = [colors_palette["steel_blue"], colors_palette["coral"]]

    donut_width = 0.7

    for i, col in enumerate(selected_columns):

        ax = axes[i]

        counts = df[col].value_counts().sort_index()
        categories = counts.index.tolist()

        wedges, _, autotexts = ax.pie(
            counts,
            labels=None,
            autopct='%1.1f%%',
            pctdistance=0.6,
            startangle=90,
            colors=colors[:len(counts)],
            wedgeprops=dict(width=donut_width, edgecolor='white', linewidth=2)
        )

        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_weight('bold')
            autotext.set_fontsize(11)

        handles = [
            mpatches.Patch(color=colors[j], label=str(cat))
            for j, cat in enumerate(categories)
        ]

        ax.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.1),
            ncol=len(categories),
            frameon=False
        )

        ax.set_title(f"{col}", fontsize=13, fontweight='bold')
        ax.set_aspect('equal')

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    if title is not None:
        fig.suptitle(title, fontsize=16, fontweight="bold")

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.5)

    return fig
    
def plot_categorical_bar_charts(
    df,
    selected_columns,
    n_cols=2,
    figsize=None,
    dpi=120,
    title=None
):
    """
    Plot multiple categorical variables as bar charts with percentage labels.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the categorical variables.

    selected_columns : list of str
        List of column names to visualize.

    n_cols : int, default=2
        Number of subplot columns.

    figsize : tuple, optional
        Figure size in inches (width, height).

    dpi : int, default=120
        Resolution of the figure in dots per inch.

    title : str, optional
        Global title of the figure.

    Returns
    -------
    matplotlib.figure.Figure
        The generated matplotlib figure.
    """

    # ---- Configuración general ----
    n_vars = len(selected_columns)
    n_rows = math.ceil(n_vars / n_cols)

    if figsize is None:
        figsize = (7 * n_cols, 4 * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, dpi=dpi)

    axes = np.array(axes).reshape(-1)

    palette_values = list(colors_palette.values())

    # ---- Loop principal ----
    for i, col in enumerate(selected_columns):

        ax = axes[i]

        counts = df[col].value_counts()
        categories = counts.index.tolist()
        n_unique = len(categories)

        # Ajustar paleta dinámicamente
        if n_unique <= len(palette_values):
            current_palette = palette_values[:n_unique]
        else:
            current_palette = (
                palette_values *
                (n_unique // len(palette_values) + 1)
            )[:n_unique]

        # Gráfico de barras
        sns.countplot(
            data=df,
            x=col,
            hue=col,
            legend=False,
            ax=ax,
            palette=current_palette
        )

        total = len(df[col])
        max_height = 0

        # Agregar porcentajes
        for p in ax.patches:
            height = p.get_height()
            max_height = max(max_height, height)

            percentage = 100 * height / total

            ax.text(
                p.get_x() + p.get_width() / 2,
                height,
                f'{percentage:.1f}%',
                ha='center',
                va='bottom',
                fontsize=10,
                fontweight='bold'
            )

        ax.set_ylim(0, max_height * 1.15)

        # Limpiar ejes
        ax.set_xticklabels([])
        ax.set_xlabel("")
        ax.set_title(col, fontsize=12, fontweight="bold")

        # Leyenda inferior
        handles = [
            mpatches.Patch(color=current_palette[j], label=str(cat))
            for j, cat in enumerate(categories)
        ]

        ax.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.08),
            ncol=min(len(categories), 4),
            frameon=False
        )

    # Eliminar subplots vacíos
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # Título global opcional
    if title is not None:
        fig.suptitle(title, fontsize=16, fontweight="bold")

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.6)

    return fig