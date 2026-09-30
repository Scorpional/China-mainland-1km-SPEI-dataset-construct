from __future__ import annotations

import matplotlib as mpl


def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 10,
            "axes.linewidth": 0.8,
            "savefig.dpi": 300,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )
