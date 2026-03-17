import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import LogNorm
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import LogLocator, LogFormatterSciNotation

# ---------- CONFIG ----------
CFO_COLOR = "#8332AC"       # main CFO (purple)
BOUND_COLOR = "#D62728"     # red bound
FEAS_COLOR = "#2CA02C"      # green feasible region
np.random.seed(0)

from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import LogNorm
from matplotlib.cm import ScalarMappable

def collect_runs_for_sweep(sweep_path):
    groups = {}
    for name in os.listdir(sweep_path):
        run_dir = os.path.join(sweep_path, name)
        if not os.path.isdir(run_dir):
            continue
        if "_" not in name:
            continue
        value_str, seed_str = name.rsplit("_", 1)
        try:
            value_str = str(float(value_str))
        except ValueError:
            continue
        groups.setdefault(value_str, []).append(run_dir)
    return groups


def collect_runs_for_sweep(sweep_path):
    """
    Groups run dirs by 'value' when folders look like:
      <value>_<seed>  e.g. 0.1_0, 0.1_1, 1.0_0 ...
    Returns dict: value_str -> list[run_dir]
    """
    groups = {}
    for name in os.listdir(sweep_path):
        run_dir = os.path.join(sweep_path, name)
        if not os.path.isdir(run_dir):
            continue

        if "_" not in name:
            continue

        value_str, seed_str = name.rsplit("_", 1)
        try:
            value_str = str(float(value_str))  # normalize e.g. "1e-3"
        except ValueError:
            # If a folder isn't numeric (e.g. "debug_0"), skip it
            continue

        groups.setdefault(value_str, []).append(run_dir)

    return groups

def summarize_sweep(sweep_dir):
    """
    Returns list[dict] with one summary per value in the sweep.
    """
    groups = collect_runs_for_sweep(sweep_dir)
    rows = []

    # sort values numerically
    for value_str in sorted(groups.keys(), key=float):
        run_dirs = groups[value_str]

        reward_arr = load_metric_arrays(run_dirs, "reward")
        constraint_arr = load_metric_arrays(run_dirs, "constraint")
        if reward_arr is None or constraint_arr is None:
            continue

        r_final = reward_arr[:, -1]
        c_final = constraint_arr[:, -1]

        r_mean, r_hw = mean_ci_scalar(r_final)
        c_mean, c_hw = mean_ci_scalar(c_final)

        rows.append({
            "param_value": float(value_str),
            "constraint_mean": c_mean,
            "constraint_ci_hw": c_hw,
            "reward_mean": r_mean,
            "reward_ci_hw": r_hw,
            "n_seeds": int(r_final.shape[0]),
        })

    return rows



def load_metric_arrays(run_dirs, metric):
    series_list = []
    for rd in run_dirs:
        csv_path = os.path.join(rd, "full_stats.csv")
        if not os.path.isfile(csv_path):
            print(f"Warning: {csv_path} not found, skipping.")
            continue
        df = pd.read_csv(csv_path)
        if metric not in df.columns:
            raise ValueError(f"Metric '{metric}' not found in {csv_path}")
        series_list.append(df[metric].to_numpy())

    if not series_list:
        return None

    min_len = min(len(s) for s in series_list)
    series_list = [s[:min_len] for s in series_list]
    return np.stack(series_list, axis=0)  # (n_seeds, T)


def mean_ci_scalar(samples):
    samples = np.asarray(samples, dtype=float)
    n = samples.size
    mean = samples.mean()
    if n > 1:
        std = samples.std(ddof=1)
        z = 1.96
        half_width = z * std / np.sqrt(n)
    else:
        half_width = 0.0
    return mean, half_width


def list_seed_dirs(parent_dir):
    return [
        os.path.join(parent_dir, d)
        for d in os.listdir(parent_dir)
        if os.path.isdir(os.path.join(parent_dir, d))
    ]


def summarize_run_group(group_dir):
    run_dirs = list_seed_dirs(group_dir)
    if not run_dirs:
        return None

    reward_arr = load_metric_arrays(run_dirs, "reward")
    constraint_arr = load_metric_arrays(run_dirs, "constraint")
    if reward_arr is None or constraint_arr is None:
        return None

    r_final = reward_arr[:, -1]
    c_final = constraint_arr[:, -1]
    r_mean, r_hw = mean_ci_scalar(r_final)
    c_mean, c_hw = mean_ci_scalar(c_final)

    return {
        "constraint_mean": c_mean,
        "constraint_ci_hw": c_hw,
        "reward_mean": r_mean,
        "reward_ci_hw": r_hw,
        "n_seeds": int(r_final.shape[0]),
    }


def parse_ablations(items):
    """
    Parse: ["eta=path/to/dir", "tau=path/to/dir", ...] -> list of (name, path)
    """
    out = []
    for s in items:
        if "=" not in s:
            raise ValueError(f"Bad --ablations entry '{s}'. Use name=path.")
        name, path = s.split("=", 1)
        out.append((name.strip(), path.strip()))
    return out


def plot_cfo_ablation_pareto(
    root_dir,
    main_cfo_dir,
    ablations,
    bound=None,
    show_front=False,
    flip_constraint_axis=True,
    reward_name=r"Dipole (in D)$\uparrow$",
    constraint_name="Energy (in Ha)",
    out_name="pareto_cfo_ablation",
    safe_fig=False,
    font_scale=2.6,
    xlim=None,
    ylim=None,
    mu_baseline_dir=None,
    mu_name=r"$\mu$",
):
    base_fs = 10 * font_scale
    rc = {
        "font.size": base_fs,
        "axes.titlesize": base_fs * 1.2,
        "axes.labelsize": base_fs * 1.1,
        "xtick.labelsize": base_fs * 1.0,
        "ytick.labelsize": base_fs * 1.0,
        "legend.fontsize": base_fs * 0.8,
        "axes.linewidth": 1.5,
    }

    with plt.rc_context(rc):
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

        ax.tick_params(
            axis="both",
            which="major",
            labelsize=base_fs * 0.85,  # smaller tick labels
            length=4,
            width=1.2,
            pad=2                      # bring numbers closer to axis
        )

        ax.tick_params(
            axis="both",
            which="minor",
            length=2,
            width=0.8
        )

        rows = []
        all_pts = []

        # --- choose styles for ablations (distinct, not purple) ---
        # tab10 gives 10 distinct colors; we skip the purple-ish one by picking from indices.
        tab = plt.cm.tab10.colors
        colors = [tab[i % len(tab)] for i in range(len(ablations))]
        markers = ["o", "D", "^", "v", "P", "X", "*", "<", ">"]

        # ---------- MAIN CFO ----------
        main = summarize_run_group(os.path.join(root_dir, main_cfo_dir))
        if main is None:
            raise RuntimeError(f"No usable runs found in main_cfo_dir={main_cfo_dir}")

        ax.errorbar(
            main["constraint_mean"], main["reward_mean"],
            xerr=main["constraint_ci_hw"], yerr=main["reward_ci_hw"],
            fmt="none", ecolor=CFO_COLOR, elinewidth=2.8, capsize=6, zorder=6,
        )
        ax.scatter(
            main["constraint_mean"], main["reward_mean"],
            marker="s", s=180, color=CFO_COLOR, edgecolor="black",
            zorder=7, label="CFO (main)"
        )

        rows.append({
            "variant": "main",
            "name": "CFO (main)",
            **main,
        })
        all_pts.append((main["constraint_mean"], main["reward_mean"]))

        # ---------- ABLATIONS ----------
        # One fixed color per ablation name (edit these to taste)
        ablation_color = {
            "rho_init": "#D62728",   # red
            "eta":      "#1F77B4",   # blue
            "tau":      "#17BECF",   # cyan
            "lambda_min":"#FF7F0E",  # orange
        }

        markers = ["o", "D", "^", "v", "P", "X", "*", "<", ">"]

        ablation_names = []
        for i, (name, path) in enumerate(ablations):
            ablation_names.append(name)
            sweep_path = os.path.join(root_dir, path)

            sweep_rows = summarize_sweep(sweep_path)
            if not sweep_rows:
                print(f"Warning: no usable runs for ablation '{name}' at {sweep_path}, skipping.")
                continue

            color = ablation_color.get(name, plt.cm.tab10(i % 10))
            marker = markers[i % len(markers)]

            for sr in sweep_rows:
                ax.errorbar(
                    sr["constraint_mean"], sr["reward_mean"],
                    xerr=sr["constraint_ci_hw"], yerr=sr["reward_ci_hw"],
                    fmt="none", ecolor=color, elinewidth=2.0, capsize=4, alpha=0.85, zorder=4
                )
                ax.scatter(
                    sr["constraint_mean"], sr["reward_mean"],
                    marker=marker, s=120, color=color, edgecolor="black",
                    zorder=5,
                    label="_nolegend_",
                )

                rows.append({
                    "variant": "ablation",
                    "name": name,
                    **sr,
                })
                all_pts.append((sr["constraint_mean"], sr["reward_mean"]))

        # if ablation_names:
        #     txt = "Ablations tested:\n" + "\n".join([f"• {n}" for n in ablation_names])
        #     ax.text(
        #         1.02, 0.98, txt,
        #         transform=ax.transAxes,
        #         ha="left", va="top",
        #         bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85, edgecolor="none")
        #     )

        # ---------- Optional: baseline mu sweep overlay ----------
        if mu_baseline_dir is not None:
            mu_path = os.path.join(root_dir, mu_baseline_dir)
            mu_groups = collect_runs_for_sweep(mu_path)
            mu_keys = sorted(mu_groups.keys(), key=float)
            mu_vals = np.array([float(k) for k in mu_keys], dtype=float)

            vmin, vmax = mu_vals.min(), mu_vals.max()

            # Trim viridis to avoid purple region (same trick as before)
            base_cmap = plt.cm.viridis
            cmap_mu = LinearSegmentedColormap.from_list(
                "viridis_no_purple",
                base_cmap(np.linspace(0.25, 1.0, 256))
            )
            norm_mu = LogNorm(vmin=vmin, vmax=vmax)

            for k in mu_keys:
                run_dirs = mu_groups[k]
                reward_arr = load_metric_arrays(run_dirs, "reward")
                constraint_arr = load_metric_arrays(run_dirs, "constraint")
                if reward_arr is None or constraint_arr is None:
                    continue

                r_final = reward_arr[:, -1]
                c_final = constraint_arr[:, -1]
                r_mean, r_hw = mean_ci_scalar(r_final)
                c_mean, c_hw = mean_ci_scalar(c_final)

                color = cmap_mu(norm_mu(float(k)))

                ax.errorbar(
                    c_mean, r_mean, xerr=c_hw, yerr=r_hw,
                    fmt="none", ecolor=color, elinewidth=2.0, capsize=4, alpha=0.75, zorder=1,
                )
                ax.scatter(
                    c_mean, r_mean,
                    marker="o", s=90, color=color, edgecolor="black",
                    alpha=0.8, zorder=2,
                    label="_nolegend_",
                )

            # colorbar for mu
            sm = ScalarMappable(norm=norm_mu, cmap=cmap_mu)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax, pad=0.02)
            cbar.set_label(mu_name, rotation=0, labelpad=5)

        # ---------- BOUND ----------
        if bound is not None:
            ax.axvline(bound, linestyle="--", linewidth=3.5, color=BOUND_COLOR, label="Bound", zorder=3)

        # ---------- LABELS ----------
        ax.set_xlabel(constraint_name)
        ax.set_ylabel(reward_name)
        ax.set_title("CFO Ablation Pareto")

        # Apply manual axis limits if provided
        if xlim is not None:
            ax.set_xlim(xlim[0], xlim[1])
        if ylim is not None:
            ax.set_ylim(ylim[0], ylim[1])

        # Flip axis so "more negative energy" appears to the right (your convention)
        if flip_constraint_axis:
            ax.invert_xaxis()

        # ---------- FEASIBLE REGION SHADING ----------
        if bound is not None:
            ax.autoscale(False)
            xmin, xmax = ax.get_xlim()
            left = min(xmin, bound)
            right = min(xmax, bound)
            ax.axvspan(left, right, color=FEAS_COLOR, alpha=0.12, zorder=0)

        ax.grid(True, alpha=0.25)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))

        # legend (dedupe)
        handles, labels = ax.get_legend_handles_labels()
        keep = {"CFO (main)", "Bound"}
        filtered = [(h, l) for h, l in zip(handles, labels) if l in keep]
        if filtered:
            h2, l2 = zip(*filtered)
            ax.legend(h2, l2)

        # fig.tight_layout()
        fig.tight_layout(rect=[0, 0, 0.82, 1])

        df = pd.DataFrame(rows)

        if safe_fig:
            out_dir = os.path.join(root_dir, "zz_figures")
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, out_name)
            fig.savefig(f"{out_path}.jpg", bbox_inches="tight")
            fig.savefig(f"{out_path}.pdf", bbox_inches="tight")
            df.to_csv(f"{out_path}_summary.csv", index=False)
            plt.close(fig)
            print(f"Saved: {out_path}.jpg/.pdf")
        else:
            plt.show()

        return df


def main():
    ap = argparse.ArgumentParser("Pareto plot comparing main CFO vs CFO ablations (each as one point with CI).")
    ap.add_argument("--root_dir", type=str, help="Root folder containing all runs.")
    ap.add_argument("--main_cfo_dir", type=str, required=True, help="Folder with main CFO runs (seed subfolders).")
    ap.add_argument("--ablations", nargs="*", default=[],
                   help="Ablation entries as name=path (repeatable), e.g. eta=... tau=...")

    ap.add_argument("--bound", type=float, default=None, help="Constraint bound (vertical line + feasible shading).")
    ap.add_argument("--show_front", action="store_true", help="Draw Pareto front across all variants.")
    ap.add_argument("--flip_constraint_axis", action="store_true",
                   help="Invert x-axis so lower energy appears to the right.")
    ap.add_argument("--safe_fig", action="store_true", help="Save figure + CSV instead of showing.")
    ap.add_argument("--font_scale", type=float, default=2.6)

    ap.add_argument("--reward", type=str, default=r"Dipole (in D)$\uparrow$")
    ap.add_argument("--constraint", type=str, default="Energy (in Ha)")
    ap.add_argument("--out_name", type=str, default="pareto_cfo_ablation")

    ap.add_argument("--xlim", nargs=2, type=float, default=None, metavar=("XMIN", "XMAX"),
                help="Set x-axis limits (Energy). Example: --xlim -87 -78")
    ap.add_argument("--ylim", nargs=2, type=float, default=None, metavar=("YMIN", "YMAX"),
                help="Set y-axis limits (Dipole). Example: --ylim 6.5 8.8")
    
    ap.add_argument("--mu_baseline_dir", type=str, default=None,
                    help="Optional: baseline mu sweep folder (subfolders value_seed).")
    ap.add_argument("--mu_name", type=str, default=r"$\mu$", help="Label for mu colorbar.")

    args = ap.parse_args()
    ablations = parse_ablations(args.ablations)

    plot_cfo_ablation_pareto(
        root_dir=args.root_dir,
        main_cfo_dir=args.main_cfo_dir,
        ablations=ablations,
        bound=args.bound,
        show_front=args.show_front,
        flip_constraint_axis=args.flip_constraint_axis,
        safe_fig=args.safe_fig,
        reward_name=args.reward,
        constraint_name=args.constraint,
        out_name=args.out_name,
        font_scale=args.font_scale,
        xlim=args.xlim,
        ylim=args.ylim,
        mu_baseline_dir=args.mu_baseline_dir,
        mu_name=args.mu_name,
    )


if __name__ == "__main__":
    main()


"""
python utils/plots_pareto_cfo_ablation.py \
  --root_dir /Users/svlg/MasterThesis/v03_geom/aa_experiments \
  --main_cfo_dir cfo_final \
  --ablations rho_init=cfo_rho_init eta=cfo_eta tau=cfo_tau lambda_min=cfo_lambda_min \
  --bound -80 \
  --flip_constraint_axis \
  --show_front \
  --out_name pareto_cfo_ablation \
  --safe_fig \
  --mu_baseline_dir aug_lag_baseline
"""

"""
python utils/plots_pareto_cfo_ablation.py \
  --root_dir /Users/svlg/MasterThesis/v03_geom/aa_experiments \
  --main_cfo_dir cfo_final \
  --ablations rho_init=cfo_rho_init eta=cfo_eta tau=cfo_tau lambda_min=cfo_lambda_min \
  --mu_baseline_dir aug_lag_baseline \
  --bound -80 \
  --flip_constraint_axis \
  --xlim -87 -77 \
  --ylim 6.44 8.8 \
  --out_name pareto_cfo_ablation_with_mu \
  --safe_fig
"""