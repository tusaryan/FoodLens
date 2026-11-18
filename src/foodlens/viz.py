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


def plot_session_temperatures(manifest, figures_dir: str | Path) -> Path:
    """Line plot of ``temp_c`` (°C) vs elapsed hours, one line per session; returns the PNG."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    ok = manifest[manifest["temp_c"].notna()]
    for sid, g in ok.groupby("session_id", sort=True):
        g = g.sort_values("elapsed_s")
        ax.plot(g["elapsed_s"] / 3600.0, g["temp_c"], label=sid)
    ax.set_xlabel("elapsed time in session (h)")
    ax.set_ylabel("ambient temperature (°C)")
    ax.set_title("Ambient temperature per session")
    ax.legend(fontsize=8)
    png = save_fig(fig, "temperature_sessions", figures_dir)
    plt.close(fig)
    return png


def contrast_stretch(img, low_pct: float = 1.0, high_pct: float = 99.5):
    """Linearly stretch a dark uint8 image so its [low, high] percentiles map to 0-255."""
    import numpy as np

    lo, hi = np.percentile(img, [low_pct, high_pct])
    scale = 255.0 / max(hi - lo, 1.0)
    return np.clip((img.astype(np.float32) - lo) * scale, 0, 255).astype(np.uint8)


def show_examples(paths: list, titles: list[str], long_side: int, name: str, figures_dir):
    """Contrast-stretched thumbnails side by side; saved via ``save_fig``; returns the figure."""
    import matplotlib.pyplot as plt

    from foodlens.io.images import load_rgb

    fig, axes = plt.subplots(1, max(len(paths), 1), figsize=(4 * max(len(paths), 1), 4))
    axes = [axes] if len(paths) <= 1 else list(axes)
    for ax, p, t in zip(axes, paths, titles, strict=False):
        ax.imshow(contrast_stretch(load_rgb(p, long_side)))
        ax.set_title(t, fontsize=8)
        ax.axis("off")
    save_fig(fig, name, figures_dir)
    return fig
