"""
Shared display settings for Spinfinity figures.

The default mode is ``"fixed"``. Set ``SPINFINITY_PLOT_MODE`` before importing
Spinfinity to choose a different initial mode, or call ``set_display_mode``.
All package plotting functions use these settings. They do not affect explicit
calls to a returned figure's ``show`` method or notebook automatic display.

Functions
---------
set_display_mode
    Set the default display mode for participating plotting functions.
show_figure
    Configure and optionally display a Plotly figure using the shared settings.
"""

import os

import plotly.graph_objects as go

_DISPLAY_MODE = os.environ.get("SPINFINITY_PLOT_MODE", "fixed")


def set_display_mode(mode: str):
    """
    Set the default display mode for participating plotting functions.

    Parameters
    ----------
    mode : {"fixed", "responsive", "no"}
        ``"fixed"`` displays figures at their preferred width.
        ``"responsive"`` fills the available container width.
        ``"no"`` constructs figures without displaying them.
        The setting applies to subsequent calls in the current Python session.
        A plotting function's explicit ``show`` argument overrides this setting.

    Raises
    ------
    ValueError
        If ``mode`` is not one of the supported display modes.
    """
    if mode not in ("fixed", "responsive", "no"):
        raise ValueError("mode must be 'fixed', 'responsive', or 'no'.")

    global _DISPLAY_MODE
    _DISPLAY_MODE = mode


def _resolve_display_mode(show: str | None = None):
    """
    Resolve a per-call override against the global display mode.

    Parameters
    ----------
    show : {"fixed", "responsive", "no"} or None, optional
        Display override. ``None`` uses the current global setting.

    Returns
    -------
    mode : str
        Effective display mode.

    Raises
    ------
    ValueError
        If the selected display mode is not supported.
    """
    mode = _DISPLAY_MODE if show is None else show
    if mode not in ("fixed", "responsive", "no"):
        raise ValueError("show must be None, 'fixed', 'responsive', or 'no'.")
    return mode


def show_figure(fig: go.Figure, preferred_width: int = 800, show: str | None = None):
    """
    Apply the display mode and optionally display a figure.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
        Figure to configure and optionally display.
    preferred_width : int, optional
        Figure width in pixels for fixed or undisplayed figures. Default is 800.
    show : {"fixed", "responsive", "no"} or None, optional
        Display override. ``None`` uses the current global setting.
        ``"no"`` does not display or clear any output. The figure can still
        be displayed explicitly later, or automatically by the notebook.
    """
    mode = _resolve_display_mode(show)
    responsive = mode == "responsive"
    fig.update_layout(
        autosize=responsive,
        width=None if responsive else preferred_width,
        height=450,
    )
    if mode != "no":
        fig.show(config={"responsive": responsive})
