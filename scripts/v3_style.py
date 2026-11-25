"""Unified Quantitative Finance publication figure style for RoughCert (v3).

Exact print width: main.tex has \\textwidth = 469.755 pt = 6.500 in.
Every figure is drawn at exact print width (zero LaTeX rescaling) with one
consistent editorial palette, Computer Modern mathtext, clean despined axes,
and fixed method identity across all main-text and supplementary figures.
"""
from __future__ import annotations

from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt

# Exact print geometry from pdflatex (\the\textwidth = 469.75502pt)
TEXTWIDTH_IN = 6.50

# Fixed method colour identity across the ENTIRE paper and supplement
PALETTE = {
    "roughcert": "#0F766E",       # Deep editorial teal (primary hero accent)
    "roughcert_fill": "#14B8A6",  # Translucent teal band fill
    "roughcert_light": "#CCFBF1", # Very light teal tint for workflow / badges
    "bdg": "#1D4ED8",             # Cobalt navy (BDG 2023 exact / root-split)
    "bdg_fd": "#60A5FA",          # Muted sky blue (BDG 100-node FD diagnostic)
    "lab": "#7C3AED",             # Royal violet (LAB 2026 direct IV)
    "markov": "#D97706",          # Burnished amber (Bayer-Breneis Markovian)
    "bcj": "#DC2626",             # Crimson vermilion (BCJ 2023 quantiser)
    "ln": "#475569",              # Cool slate (moment-matched lognormal)
    "jmm": "#94A3B8",             # Soft silver-slate (JMM 2018 bounds)
    "tick": "#BE123C",            # Deep rose for 1-Cboe-tick threshold (0.05 pts)
    "tick_fill": "#FFE4E6",       # Soft rose violation shading
    "grid": "#E2E8F0",            # Faint neutral hairline grid
    "ink": "#1E293B",             # Dark charcoal spine / primary ink
    "muted": "#64748B",           # Secondary annotation / reference ink
    "gold_fill": "#FEF3C7",       # Soft amber highlight (500-cell / gap)
}

# Fixed curve colours for multi-H or multi-curve breakdowns (colour-blind safe)
H_COLORS = {
    "0.05": "#DC2626",  # Crimson (ultra-rough)
    ".05":  "#DC2626",
    "0.10": "#1D4ED8",  # Cobalt (calibrated rough)
    ".10":  "#1D4ED8",
    "0.30": "#0F766E",  # Teal (smoother rough)
    ".30":  "#0F766E",
    "0.50": "#D97706",  # Amber (Brownian)
    "0.75": "#7C3AED",  # Violet (persistent)
}

CURVE_COLORS = {
    "quad": "#0F766E",  # Quadratic rising
    "sqrt": "#1D4ED8",  # Square-root mild rise
    "flat": "#D97706",  # Flat
    "stress": "#DC2626",
}


def apply_style(base_size: float = 8.5) -> None:
    """Apply clean editorial Quantitative Finance Matplotlib rcParams."""
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["CMU Serif", "Computer Modern Roman", "DejaVu Serif", "Times New Roman"],
        "mathtext.fontset": "cm",
        "font.size": base_size,
        "axes.titlesize": base_size + 0.3,
        "axes.titleweight": "normal",
        "axes.titlelocation": "left",
        "axes.titlepad": 6.0,
        "axes.labelsize": base_size,
        "axes.labelcolor": PALETTE["ink"],
        "axes.edgecolor": PALETTE["ink"],
        "axes.linewidth": 0.65,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.labelsize": base_size - 0.8,
        "ytick.labelsize": base_size - 0.8,
        "xtick.color": PALETTE["ink"],
        "ytick.color": PALETTE["ink"],
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.minor.width": 0.4,
        "ytick.minor.width": 0.4,
        "xtick.major.size": 3.2,
        "ytick.major.size": 3.2,
        "xtick.minor.size": 1.8,
        "ytick.minor.size": 1.8,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "lines.linewidth": 1.35,
        "lines.markersize": 4.2,
        "legend.fontsize": base_size - 0.8,
        "legend.frameon": False,
        "legend.handlelength": 1.8,
        "legend.handletextpad": 0.45,
        "legend.columnspacing": 1.1,
        "grid.color": PALETTE["grid"],
        "grid.linewidth": 0.55,
        "grid.linestyle": "-",
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "text.color": PALETTE["ink"],
    })


def style_axes(ax, grid_axis: str = "y") -> None:
    """Apply subtle background grid and crisp despined aesthetics."""
    ax.set_axisbelow(True)
    if grid_axis:
        ax.grid(True, axis=grid_axis, which="major", color=PALETTE["grid"], lw=0.55, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(PALETTE["ink"])
    ax.spines["bottom"].set_color(PALETTE["ink"])


def panel_title(ax, letter: str, subtitle: str) -> None:
    """Bold lowercase panel label followed by crisp plain subtitle."""
    ax.set_title(f"$\\mathbf{{({letter})}}$  {subtitle}", loc="left", pad=6.5)


def tex_sci(v: float) -> str:
    """Format scientific notation cleanly as LaTeX math (no 1e-05 strings)."""
    if v == 0:
        return "$0$"
    m, e = f"{v:.1e}".split("e")
    e_int = int(e)
    m_f = float(m)
    if abs(m_f - 1.0) < 1e-9:
        return f"$10^{{{e_int}}}$"
    if abs(m_f - round(m_f)) < 1e-9:
        return f"${int(round(m_f))}\\times 10^{{{e_int}}}$"
    return f"${m_f:.1f}\\times 10^{{{e_int}}}$"


def save_figure(fig, out_dir: Path, stem: str, also_png: bool = True) -> list[Path]:
    """Save vector PDF (for LaTeX) and high-res PNG preview (for visual QA)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / f"{stem}.pdf"
    # Preserve all external legend and tick text; place the vector content at
    # exactly the physical width used by both LaTeX documents.
    from io import BytesIO
    import fitz
    buf=BytesIO()
    fig.savefig(buf,format="pdf",bbox_inches="tight",pad_inches=0.045)
    with fitz.open(stream=buf.getvalue(),filetype="pdf") as src, fitz.open() as out:
        box=src[0].rect
        page=out.new_page(width=72*TEXTWIDTH_IN,height=72*TEXTWIDTH_IN*box.height/box.width)
        page.show_pdf_page(page.rect,src,0)
        out.save(pdf_path,garbage=4,deflate=True,no_new_id=True)
        paths=[pdf_path]
        if also_png:
            png_dir=out_dir.parents[1]/"audit"/"figure_previews"
            png_dir.mkdir(parents=True,exist_ok=True)
            png_path=png_dir/(stem+".png")
            page.get_pixmap(matrix=fitz.Matrix(240/72,240/72)).save(png_path)
            paths.append(png_path)
    plt.close(fig)
    return paths
