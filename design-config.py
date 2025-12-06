"""
Design System Configuration for bhuv's notebook
https://bhuvanesh09.github.io

This module provides easy access to the website's design system
for creating consistent data visualizations and plots.

Usage:
    from design_config import COLORS, PLOT_CONFIG, create_plotly_layout

Example:
    import plotly.graph_objects as go
    from design_config import COLORS, create_plotly_layout

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[1, 2, 3],
        y=[4, 5, 6],
        line=dict(color=COLORS['plot']['steel_blue'], width=2),
        marker=dict(color=COLORS['plot']['steel_blue'], size=8)
    ))
    fig.update_layout(create_plotly_layout("My Chart Title"))
"""

# Color Definitions
COLORS = {
    # Background Colors
    'background': {
        'linen': '#FAF0E6',          # Main website background (light mode)
        'cream': '#FAF8F1',          # Plot/chart background
        'dark': 'rgba(0, 0, 0, 0.9)', # Dark mode background
    },

    # Data Visualization Palette (use in this order)
    'plot': {
        'steel_blue': '#4A90C0',      # Primary series
        'sandy_orange': '#FFA54F',    # Secondary series
        'medium_purple': '#9370DB',   # Tertiary series
        'medium_aquamarine': '#66CDAA', # Quaternary series
        'indian_red': '#CD5C5C',      # Quinary series
    },

    # Typography Colors (Light Mode)
    'text': {
        'primary': '#000000',
        'body': '#374151',
        'heading': '#111827',
        'link': '#111827',
        'lead': '#4b5563',
        'code': '#111827',
    },

    # Typography Colors (Dark Mode)
    'text_dark': {
        'primary': '#FFFFFF',
        'body': '#d1d5db',
        'heading': '#FFFFFF',
        'lead': '#9ca3af',
    },

    # UI Elements
    'ui': {
        'border_light': '#e5e7eb',
        'border_medium': '#E0E0E0',
        'counters': '#6b7280',
        'bullets': '#d1d5db',
        'captions': '#6b7280',
        'placeholder': '#9ca3af',
    },

    # Accent Colors
    'accent': {
        'red': '#ff3b2d',
        'focus_blue': 'rgb(59, 130, 246)',
    },
}

# Plot Data Series Colors (as list for easy iteration)
PLOT_COLORS = [
    '#4A90C0',  # Steel Blue
    '#FFA54F',  # Sandy Orange
    '#9370DB',  # Medium Purple
    '#66CDAA',  # Medium Aquamarine
    '#CD5C5C',  # Indian Red
]

# Standard Plot Configuration
PLOT_CONFIG = {
    'background': {
        'plot': '#FAF8F1',
        'paper': '#FAF8F1',
    },
    'grid': {
        'color': '#E0E0E0',
        'width': 1,
    },
    'dimensions': {
        'width': 800,
        'height': 600,
    },
    'marker': {
        'size': 8,
    },
    'line': {
        'width': 2,
    },
}

# Typography Configuration
TYPOGRAPHY = {
    'fonts': {
        'heading': 'JunicodeVF',
        'body': 'FiraCode',
        'monospace': 'FiraCode',
    },
    'sizes': {
        'h1': '2.25em',
        'h2': '1.5em',
        'h3': '1.25em',
        'h4': '1em',
        'body': '1.0rem',
        'large': '1.125rem',
        'small': '0.875rem',
        'code': '0.875rem',
    },
    'line_heights': {
        'h1': 1.111,
        'h2': 1.333,
        'h3': 1.6,
        'h4': 1.5,
        'body': 1.6,
        'large': 1.75,
        'small': 1.25,
        'code': 1.5,
    },
}

# Layout Configuration
LAYOUT = {
    'max_content_width': '48rem',  # 768px
    'prose_width': '65ch',
    'padding': {
        'section_horizontal': '2rem',
        'large_vertical': '4rem',
        'medium_vertical': '2.5rem',
        'small_vertical': '0.5rem',
    },
}


# Helper Functions

def get_plot_color(index):
    """
    Get plot color by index (0-4), cycling through available colors.

    Args:
        index (int): Index of the color (0-4 recommended, will cycle if larger)

    Returns:
        str: Hex color code

    Example:
        >>> get_plot_color(0)
        '#4A90C0'
        >>> get_plot_color(5)  # Cycles back to first color
        '#4A90C0'
    """
    return PLOT_COLORS[index % len(PLOT_COLORS)]


def create_plotly_layout(title="", xaxis_title="", yaxis_title="", **kwargs):
    """
    Create a standard Plotly layout with website styling.

    Args:
        title (str): Chart title
        xaxis_title (str): X-axis label
        yaxis_title (str): Y-axis label
        **kwargs: Additional layout parameters to override defaults

    Returns:
        dict: Plotly layout configuration

    Example:
        >>> layout = create_plotly_layout(
        ...     title="My Chart",
        ...     xaxis_title="Time",
        ...     yaxis_title="Value"
        ... )
    """
    layout = {
        'plot_bgcolor': PLOT_CONFIG['background']['plot'],
        'paper_bgcolor': PLOT_CONFIG['background']['paper'],
        'width': PLOT_CONFIG['dimensions']['width'],
        'height': PLOT_CONFIG['dimensions']['height'],
        'xaxis': {
            'title': {'text': xaxis_title},
            'gridcolor': PLOT_CONFIG['grid']['color'],
            'gridwidth': PLOT_CONFIG['grid']['width'],
        },
        'yaxis': {
            'title': {'text': yaxis_title},
            'gridcolor': PLOT_CONFIG['grid']['color'],
            'gridwidth': PLOT_CONFIG['grid']['width'],
        },
    }

    if title:
        layout['title'] = {'text': title}

    # Override with any custom parameters
    layout.update(kwargs)

    return layout


def create_trace_style(color_index=0, mode='lines+markers'):
    """
    Create standard trace styling for Plotly.

    Args:
        color_index (int): Index of color from PLOT_COLORS (0-4)
        mode (str): Trace mode ('lines', 'markers', 'lines+markers')

    Returns:
        dict: Trace styling configuration

    Example:
        >>> style = create_trace_style(0)
        >>> trace = go.Scatter(x=[1,2,3], y=[4,5,6], **style)
    """
    color = get_plot_color(color_index)

    style = {
        'mode': mode,
        'line': {
            'color': color,
            'width': PLOT_CONFIG['line']['width'],
        },
        'marker': {
            'color': color,
            'size': PLOT_CONFIG['marker']['size'],
        },
    }

    return style


def get_color_palette():
    """
    Get the complete plot color palette.

    Returns:
        list: List of hex color codes

    Example:
        >>> palette = get_color_palette()
        >>> print(palette)
        ['#4A90C0', '#FFA54F', '#9370DB', '#66CDAA', '#CD5C5C']
    """
    return PLOT_COLORS.copy()


# Matplotlib Configuration (if using matplotlib instead of plotly)
MATPLOTLIB_CONFIG = {
    'figure.facecolor': COLORS['background']['cream'],
    'axes.facecolor': COLORS['background']['cream'],
    'axes.edgecolor': COLORS['ui']['border_medium'],
    'axes.grid': True,
    'grid.color': COLORS['ui']['border_medium'],
    'grid.linewidth': 0.5,
    'axes.prop_cycle': f"cycler('color', {PLOT_COLORS})",
    'font.family': 'monospace',
    'font.size': 10,
    'figure.figsize': (10, 6),
}


def configure_matplotlib():
    """
    Configure matplotlib with website design system.

    Usage:
        import matplotlib.pyplot as plt
        from design_config import configure_matplotlib

        configure_matplotlib()
        # Now create your plots with consistent styling
    """
    try:
        import matplotlib.pyplot as plt
        from cycler import cycler

        plt.rcParams.update({
            'figure.facecolor': COLORS['background']['cream'],
            'axes.facecolor': COLORS['background']['cream'],
            'axes.edgecolor': COLORS['ui']['border_medium'],
            'axes.grid': True,
            'grid.color': COLORS['ui']['border_medium'],
            'grid.linewidth': 0.5,
            'axes.prop_cycle': cycler('color', PLOT_COLORS),
            'font.family': 'monospace',
            'font.size': 10,
            'figure.figsize': (10, 6),
        })
        print("Matplotlib configured with website design system")
    except ImportError:
        print("Matplotlib not installed. Skipping configuration.")


# Quick reference for RGB values
COLORS_RGB = {
    'linen': (250, 240, 230),
    'cream': (250, 248, 241),
    'steel_blue': (74, 144, 192),
    'sandy_orange': (255, 165, 79),
    'medium_purple': (147, 112, 219),
    'medium_aquamarine': (102, 205, 170),
    'indian_red': (205, 92, 92),
}


if __name__ == "__main__":
    # Display color palette when run directly
    print("=" * 60)
    print("BHUV'S NOTEBOOK - DESIGN SYSTEM")
    print("=" * 60)
    print("\nPlot Color Palette (in order):")
    print("-" * 60)
    for i, color in enumerate(PLOT_COLORS, 1):
        print(f"  {i}. {color}")

    print("\nBackground Colors:")
    print("-" * 60)
    for name, color in COLORS['background'].items():
        print(f"  {name}: {color}")

    print("\nExample Usage:")
    print("-" * 60)
    print("""
    import plotly.graph_objects as go
    from design_config import create_plotly_layout, create_trace_style

    # Create figure with standard styling
    fig = go.Figure()

    # Add traces with automatic color cycling
    for i, (x, y, name) in enumerate(data_series):
        fig.add_trace(go.Scatter(
            x=x, y=y, name=name,
            **create_trace_style(i)
        ))

    # Apply standard layout
    fig.update_layout(create_plotly_layout(
        title="My Chart",
        xaxis_title="X Axis",
        yaxis_title="Y Axis"
    ))
    """)
    print("=" * 60)
