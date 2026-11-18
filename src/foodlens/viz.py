"""Plot helpers: consistent light colours and figure saving (PNG 300 dpi + SVG)."""

from __future__ import annotations

from pathlib import Path

from matplotlib.figure import Figure

LIGHT_COLORS: dict[str, str] = {"RED": "#d62728", "GREEN": "#2ca02c", "BLUE": "#1f77b4"}
FIG_DPI = 300


def save_fig(fig: Figure, name: str, figures_dir: str | Path | None = None) -> Path:
    """Save ``fig`` as ``<figures_dir>/<name>.png`` (300 dpi) and ``.svg``; return the PNG path.

    ``figures_dir`` defaults to ``<paths.reports_dir>/figures`` from the default config.
    """
    if figures_dir is None:
        from foodlens.config import load_config

        figures_dir = Path(load_config().paths.reports_dir) / "figures"
    out = Path(figures_dir)
    out.mkdir(parents=True, exist_ok=True)
    png = out / f"{name}.png"
    fig.savefig(png, dpi=FIG_DPI, bbox_inches="tight")
    fig.savefig(out / f"{name}.svg", bbox_inches="tight")
    return png
