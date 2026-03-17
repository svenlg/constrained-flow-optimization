import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


# --------------------
# Defaults / Styling
# --------------------
FULFILLED_COLOR = "#4C72B0"   # blue
VIOLATED_COLOR  = "#DD8452"   # orange
CFO_COLOR = "#8332AC"       # main CFO (purple)


def _safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _pick_cfo_value(df: pd.DataFrame, dipole_col: str, method_col: str = "method") -> float:
    """
    Prefer CFO row if available (method == 'CFO'), otherwise fall back to max dipole row.
    """
    if method_col in df.columns:
        cfo_rows = df[df[method_col].astype(str).str.upper().eq("CFO")]
        if len(cfo_rows) >= 1:
            return float(pd.to_numeric(cfo_rows[dipole_col]).iloc[0])
    # fallback: pick highest dipole
    return float(pd.to_numeric(df[dipole_col]).max())


def plot_histogram_from_pareto_summary(
    in_csv: str,
    out_name: str = "hist_dipole_from_pareto",
    bound: float = -80.0,
    dipole_col: str = "reward_mean",
    constraint_col: str = "constraint_mean",
    bin_width: float = 0.05,
    safe_fig: bool = False,
    dpi: int = 500,
    figsize=(3.2, 2.6),
    font_scale: float = 1.0,
    arrow: bool = True,
):
    # --------------------
    # Load data
    # --------------------
    df = pd.read_csv(in_csv)

    if dipole_col not in df.columns or constraint_col not in df.columns:
        raise ValueError(
            f"CSV must contain columns '{dipole_col}' and '{constraint_col}'. "
            f"Found columns: {list(df.columns)}"
        )

    df[dipole_col] = pd.to_numeric(df[dipole_col], errors="coerce")
    df[constraint_col] = pd.to_numeric(df[constraint_col], errors="coerce")
    df = df.dropna(subset=[dipole_col, constraint_col]).copy()

    if df.empty:
        raise RuntimeError("No valid numeric rows after cleaning. Check your CSV and column names.")

    fulfilled = df[df[constraint_col] < bound]
    violated  = df[df[constraint_col] > bound]

    cfo_dipole = _pick_cfo_value(df, dipole_col=dipole_col)

    # --------------------
    # Histogram bins
    # --------------------
    xmin, xmax = float(df[dipole_col].min()), float(df[dipole_col].max())
    if bin_width <= 0:
        raise ValueError("--bin_width must be > 0")

    # add a tiny epsilon so the max lands inside the last bin
    eps = 1e-12
    bins = np.arange(xmin, xmax + bin_width + eps, bin_width)

    # --------------------
    # Plot
    # --------------------
    base_fs = 10 * font_scale
    rc = {
        "font.size": base_fs,
        "axes.titlesize": base_fs * 1.1,
        "axes.labelsize": base_fs * 1.0,
        "xtick.labelsize": base_fs * 1.0,
        "ytick.labelsize": base_fs * 1.0,
        "legend.fontsize": base_fs * 0.75,
        "axes.linewidth": 1.0,
    }

    with plt.rc_context(rc):
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        ax.tick_params(
            axis="both",
            which="major",
            length=2,
            width=1.0,
            pad=0
        )

        ax.tick_params(
            axis="both",
            which="minor",
            length=1,
            width=0.8
        )

        # --------------------
        # Histogram
        # --------------------
        ax.hist(
            [fulfilled[dipole_col], violated[dipole_col]],
            bins=bins,
            stacked=True,
            color=[FULFILLED_COLOR, VIOLATED_COLOR],
            edgecolor="black",
            linewidth=0.4,
            alpha=0.85,
            label=[
                "Manual sweep:\nConstraint fulfilled",
                "Manual sweep:\nConstraint not fulfilled",
            ],
        )

        # --------------------
        # CFO marker
        # --------------------
        ax.axvline(
            cfo_dipole,
            linestyle="--",
            linewidth=3.0,
            color=CFO_COLOR,
            label="CFO (ours): 1 run",
            zorder=5
        )

        # --------------------
        # Labels
        # --------------------
        ax.set_xlabel(r"Dipole (D) $\uparrow$", labelpad=0)
        ax.set_ylabel("Count", labelpad=0)

        # --------------------
        # Axes styling
        # --------------------
        # for spine in ax.spines.values():
        #     spine.set_color("black")
        # ax.tick_params(axis="both", colors="black")

        # --------------------
        # Maximize arrow
        # --------------------
        if arrow:
            arrow_pos_y = 0.12
            arrow_pos_x = 0.40

            ax.annotate(
                "",
                xy=(arrow_pos_x + 0.18, arrow_pos_y),
                xytext=(arrow_pos_x, arrow_pos_y),
                xycoords="axes fraction",
                textcoords="axes fraction",
                arrowprops=dict(
                    arrowstyle="->",
                    lw=1.1,
                    color="black"
                ),
                annotation_clip=False
            )

            ax.text(
                arrow_pos_x + 0.09,
                arrow_pos_y - 0.02,
                "maximize",
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=base_fs * 0.8
            )

        # --------------------
        # Grid
        # --------------------
        ax.set_axisbelow(True)
        # ax.yaxis.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.35)
        ax.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.35)
        ax.minorticks_on()
        ax.yaxis.grid(True, which="minor", linestyle="--", linewidth=0.4, alpha=0.2)
        ax.yaxis.set_major_formatter(
            FuncFormatter(lambda y, _: "" if y == 0 else f"{int(y)}")
        )

        # --------------------
        # Legend
        # --------------------
        ax.legend(frameon=True, loc="upper center")

        # --------------------
        # Title
        # --------------------
        ax.set_title("Baseline Dipole Histogram", pad=0)

        # --------------------
        # Layout
        # --------------------

        fig.tight_layout()

        # --------------------
        # Save / show
        # --------------------
        if safe_fig:
            out_dir = os.path.dirname(os.path.abspath(in_csv))
            _safe_mkdir(out_dir)

            out_path = os.path.join(out_dir, out_name)
            fig.savefig(f"{out_path}.jpg", bbox_inches="tight")
            fig.savefig(f"{out_path}.pdf", bbox_inches="tight")
            plt.close(fig)
            print(f"Saved histogram to {out_path}.jpg/.pdf")
        else:
            plt.show()


def main():
    ap = argparse.ArgumentParser(
        description="Stacked histogram from Pareto summary CSV (fulfilled vs violated constraint) with CFO marker."
    )
    ap.add_argument("--in_csv", type=str, required=True, help="Path to pareto_*_summary.csv")
    ap.add_argument("--out_name", type=str, default="hist_dipole_from_pareto",
                    help="Base name of saved figure (no extension).")
    ap.add_argument("--bound", type=float, default=-80.0, help="Constraint bound used for fulfilled/violated split.")
    ap.add_argument("--dipole_col", type=str, default="reward_mean", help="Column for dipole values.")
    ap.add_argument("--constraint_col", type=str, default="constraint_mean", help="Column for constraint values.")
    ap.add_argument("--bin_width", type=float, default=0.05, help="Histogram bin width.")
    ap.add_argument("--font_scale", type=float, default=1.0, help="Font size scaling.")
    ap.add_argument("--dpi", type=int, default=500, help="Figure DPI.")
    ap.add_argument("--safe_fig", action="store_true", help="Save PDF/JPG to zz_figures instead of showing.")
    ap.add_argument("--no_arrow", action="store_true", help="Disable the maximize arrow annotation.")

    args = ap.parse_args()

    plot_histogram_from_pareto_summary(
        in_csv=args.in_csv,
        out_name=args.out_name,
        bound=args.bound,
        dipole_col=args.dipole_col,
        constraint_col=args.constraint_col,
        bin_width=args.bin_width,
        safe_fig=args.safe_fig,
        dpi=args.dpi,
        font_scale=args.font_scale,
        arrow=(not args.no_arrow)
    )


if __name__ == "__main__":
    main()


"""
Example:
python utils/plot_hist.py \
  --in_csv /Users/svlg/MasterThesis/v03_geom/aa_experiments/zz_figures/pareto_v3_cfo_vs_fixed_baseline_summary.csv \
  --bound -80 \
  --bin_width 0.05 \
  --out_name hist_pareto_v3 \
  --safe_fig
"""
