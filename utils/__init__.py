from .colors import colors_palette
from .preprocessing import OutlierRemoverIQR

# Saving utilities
from .visualization.saving import save_figure

# Distribution plots
from .visualization.distributions import (
    plot_numerical_distributions,
    plot_dual_histogram_comparison,
    plot_group_distribution_comparison,
)

# Categorical plots
from .visualization.categorical import (
    plot_binary_donuts,
    plot_categorical_bar_charts,
)

# Correlation plots
from .visualization.correlation import (
    plot_correlation_heatmap,
    plot_high_correlation_scatter,
)