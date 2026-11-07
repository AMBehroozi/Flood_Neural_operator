import torch
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
from datetime import datetime
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm, LightSource
from datetime import datetime, timedelta
from matplotlib.ticker import FormatStrFormatter


def eventwise_csi_for_violin(
    Depth_True,
    Depth_Pred,
    wet_threshold=0.025,
    eps=1e-12,
):
    """
    Compute event-wise CSI for violin plots.

    Depth_True, Depth_Pred: torch tensors [nbatch, nx, ny, nt]

    Returns
    -------
    dict with:
        "CSI": tensor [nbatch]
    """

    assert Depth_True.shape == Depth_Pred.shape, (
        "True and predicted tensors must have the same shape."
    )

    assert Depth_True.ndim == 4, (
        "Expected tensors with shape [nbatch, nx, ny, nt]."
    )

    nb = Depth_True.shape[0]
    csi_all = []

    for b in range(nb):
        O = Depth_True[b]  # [nx, ny, nt]
        P = Depth_Pred[b]

        wet_O = O > wet_threshold
        wet_P = P > wet_threshold

        TP = torch.sum(wet_O & wet_P).float()
        FP = torch.sum(~wet_O & wet_P).float()
        FN = torch.sum(wet_O & ~wet_P).float()

        csi = TP / (TP + FP + FN + eps)

        csi_all.append(csi)

    return {
        "CSI": torch.stack(csi_all)  # [nbatch]
    }


def eventwise_relrmse_nse_for_violin(
    Depth_True,
    Depth_Pred,
    eps=1e-12,
):
    """
    Compute event-wise relRMSE and NSE for violin plots.

    Inputs
    ------
    Depth_True, Depth_Pred : torch.Tensor
        Shape [nbatch, nx, ny, nt]

    Returns
    -------
    dict of tensors, each with shape [nbatch]
    """

    assert Depth_True.shape == Depth_Pred.shape, (
        "True and predicted tensors must have the same shape."
    )

    assert Depth_True.ndim == 4, (
        "Expected tensors with shape [nbatch, nx, ny, nt]."
    )

    nb = Depth_True.shape[0]

    relrmse_all = []
    nse_all = []

    for b in range(nb):
        O = Depth_True[b]  # [nx, ny, nt]
        P = Depth_Pred[b]

        diff = O - P

        rmse = torch.sqrt(torch.mean(diff**2))
        mean_O = torch.mean(O)

        relrmse = rmse / (mean_O + eps)

        nse = 1.0 - torch.sum(diff**2) / (
            torch.sum((O - mean_O) ** 2) + eps
        )

        relrmse_all.append(relrmse)
        nse_all.append(nse)

    return {
        "relRMSE": torch.stack(relrmse_all),  # [nbatch]
        "NSE": torch.stack(nse_all),          # [nbatch]
    }


def plot_test_violin(
    values,
    metric_label="RMSE (m)",
    title="RMSE (m) over test scenarios",
    ylabel=None,
    save_path=None,
    ylim=None,
    violin_x=1.25,  # shift violin to right
    bbox_to_anchor=(0.02, 0.98),
):
    """
    Plot a single violin plot for test values only.
    """

    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()

    values = np.asarray(values, dtype=float).reshape(-1)
    values = values[~np.isnan(values)]

    if len(values) == 0:
        raise ValueError("No valid values to plot.")

    mean_val = np.mean(values)
    median_val = np.median(values)
    min_val = np.min(values)
    max_val = np.max(values)
    q25, q75 = np.percentile(values, [25, 75])

    # -----------------------------
    # Automatic y-axis limits
    # -----------------------------
    if ylim is None:
        pad_frac = 0.10
        data_range = max_val - min_val

        if data_range == 0:
            pad = pad_frac * abs(max_val) if abs(max_val) > 0 else 0.1
        else:
            pad = pad_frac * data_range

        ylim = (min_val - pad, max_val + pad)

    # -----------------------------
    # Figure
    # -----------------------------
    fig, ax = plt.subplots(figsize=(3.5, 5.2))

    parts = ax.violinplot(
        [values],
        positions=[violin_x],
        widths=0.55,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )

    for body in parts["bodies"]:
        body.set_alpha(0.35)

    # Interquartile range
    ax.vlines(
        violin_x,
        q25,
        q75,
        linewidth=6,
        alpha=0.55,
        zorder=4,
    )

    # # Median
    # ax.scatter(
    #     violin_x,
    #     median_val,
    #     marker="o",
    #     s=55,
    #     edgecolor="black",
    #     zorder=5,
    #     label="Median",
    # )

    # Mean
    ax.scatter(
        violin_x,
        mean_val,
        marker="D",
        s=80,
        edgecolor="black",
        zorder=6,
        label="Mean",
    )

    # Min / mean / max guide lines
    guide_vals = {
        "max": max_val,
        "mean": mean_val,
        "min": min_val,
    }

    for name, val in guide_vals.items():
        ax.hlines(
            val,
            violin_x + 0.12,
            violin_x + 0.55,
            linestyles="--",
            linewidth=1.4,
            alpha=0.8,
        )
        ax.text(
            violin_x + 0.60,
            val,
            f"{val:.4f}",
            va="center",
            ha="left",
            fontsize=14,
        )

    # Jittered points
    jitter = np.random.normal(violin_x, 0.025, size=len(values))
    ax.scatter(
        jitter,
        values,
        s=14,
        alpha=0.25,
        edgecolor="none",
        zorder=3,
    )

    ax.set_xticks([violin_x])
    ax.set_xticklabels([""], fontsize=16)

    ax.set_ylabel(
        ylabel if ylabel is not None else metric_label,
        fontsize=18,
    )

    # Wider x-range with free space on both sides
    ax.set_xlim(0.45, 2.0)

    ax.set_ylim(ylim)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.4f"))

    # Larger tick label numbers
    ax.tick_params(axis="both", labelsize=12)

    ax.grid(True, axis="y", alpha=0.25)

    # ax.legend(
    #     frameon=True,
    #     loc="upper left",
    #     bbox_to_anchor=bbox_to_anchor,
    #     fontsize=16
    # )

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()
    return fig, ax


def plot_test_violin_D(
    fine_values,
    coarse_values,
    metric_label="RMSE (m)",
    title="RMSE (m) over test scenarios",
    ylabel=None,
    save_path=None,
    fine_ylim=None,
    coarse_ylim=None,
    bbox_to_anchor=(0.02, 0.98),
    value_format="decimal",
    decimals=5,
):
    """
    Plot Fine and Coarse violin plots together with independent y-axes.

    Fine   = left y-axis
    Coarse = right y-axis
    """

    # -----------------------------
    # Convert inputs
    # -----------------------------
    if isinstance(fine_values, torch.Tensor):
        fine_values = fine_values.detach().cpu().numpy()

    if isinstance(coarse_values, torch.Tensor):
        coarse_values = coarse_values.detach().cpu().numpy()

    fine_values = np.asarray(fine_values, dtype=float).reshape(-1)
    coarse_values = np.asarray(coarse_values, dtype=float).reshape(-1)

    fine_values = fine_values[~np.isnan(fine_values)]
    coarse_values = coarse_values[~np.isnan(coarse_values)]

    if len(fine_values) == 0:
        raise ValueError("No valid fine values to plot.")

    if len(coarse_values) == 0:
        raise ValueError("No valid coarse values to plot.")

    # -----------------------------
    # Positions
    # -----------------------------
    fine_x = 0.5
    coarse_x = 1.3

    # -----------------------------
    # Independent automatic y-limits
    # -----------------------------
    def get_ylim(values):
        min_val = np.min(values)
        max_val = np.max(values)

        pad_frac = 0.10
        data_range = max_val - min_val

        if data_range == 0:
            pad = pad_frac * abs(max_val) if abs(max_val) > 0 else 0.1
        else:
            pad = pad_frac * data_range

        return min_val - pad, max_val + pad

    if fine_ylim is None:
        fine_ylim = get_ylim(fine_values)

    if coarse_ylim is None:
        coarse_ylim = get_ylim(coarse_values)

    # -----------------------------
    # Figure and two y-axes
    # -----------------------------
    fig, ax_fine = plt.subplots(figsize=(4.5, 5.2))
    ax_coarse = ax_fine.twinx()

    # -----------------------------
    # Fine violin - LEFT axis
    # -----------------------------
    mean_val = np.mean(fine_values)
    min_val = np.min(fine_values)
    max_val = np.max(fine_values)
    q25, q75 = np.percentile(fine_values, [25, 75])

    parts = ax_fine.violinplot(
        [fine_values],
        positions=[fine_x],
        widths=0.15,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )

    for body in parts["bodies"]:
        body.set_alpha(0.35)

    ax_fine.vlines(
        fine_x,
        q25,
        q75,
        linewidth=6,
        alpha=0.55,
        zorder=4,
    )

    ax_fine.scatter(
        fine_x,
        mean_val,
        marker="D",
        s=80,
        edgecolor="black",
        zorder=6,
    )




    for val in [max_val, mean_val, min_val]:
        ax_fine.hlines(
            val,
            fine_x + 0.12,
            fine_x + 0.42,
            linestyles="--",
            linewidth=1.4,
            alpha=0.8,
        )

        if value_format == "percent":
            label = f"{val * 100:.{decimals}f}%"
        elif value_format == "decimal":
            label = f"{val:.{decimals}f}"
        else:
            label = str(val)


        ax_fine.text(
            fine_x + 0.12,
            val,
            f"\n{label}",
            va="center",
            ha="left",
            fontsize=14,
        )

    jitter = np.random.normal(
        fine_x,
        0.025,
        size=len(fine_values),
    )

    ax_fine.scatter(
        jitter,
        fine_values,
        s=14,
        alpha=0.25,
        edgecolor="none",
        zorder=3,
    )

    # -----------------------------
    # Coarse violin - RIGHT axis
    # -----------------------------
    mean_val = np.mean(coarse_values)
    min_val = np.min(coarse_values)
    max_val = np.max(coarse_values)
    q25, q75 = np.percentile(coarse_values, [25, 75])

    parts = ax_coarse.violinplot(
        [coarse_values],
        positions=[coarse_x],
        widths=0.15,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )

    for body in parts["bodies"]:
        body.set_alpha(0.35)

    ax_coarse.vlines(
        coarse_x,
        q25,
        q75,
        linewidth=6,
        alpha=0.55,
        zorder=4,
    )

    ax_coarse.scatter(
        coarse_x,
        mean_val,
        marker="D",
        s=80,
        edgecolor="black",
        zorder=6,
    )

    for val in [max_val, mean_val, min_val]:
        ax_coarse.hlines(
            val,
            coarse_x + 0.12,
            coarse_x + 0.42,
            linestyles="--",
            linewidth=1.4,
            alpha=0.8,
        )

        if value_format == "percent":
            label = f"{val * 100:.{decimals}f}%"
        elif value_format == "decimal":
            label = f"{val:.{decimals}f}"
        else:
            label = str(val)


        ax_coarse.text(
            coarse_x + 0.12,
            val,
            f"\n{label}",
            va="center",
            ha="left",
            fontsize=14,
        )

    jitter = np.random.normal(
        coarse_x,
        0.025,
        size=len(coarse_values),
    )

    ax_coarse.scatter(
        jitter,
        coarse_values,
        s=14,
        alpha=0.25,
        edgecolor="none",
        zorder=3,
    )

    # -----------------------------
    # X-axis
    # -----------------------------
    ax_fine.set_xticks([fine_x, coarse_x])
    ax_fine.set_xticklabels(
        ["Fine (FNO)", "Coarse (FNO)"],
        fontsize=16,
    )

    ax_fine.set_xlim(0.35, 1.9)
    ax_coarse.set_xlim(0.35, 1.9)

    # -----------------------------
    # Independent Y-axes
    # -----------------------------
    axis_label = ylabel if ylabel is not None else metric_label

    ax_fine.set_ylabel(
        f"{axis_label}",
        fontsize=18,
    )

    # ax_coarse.set_ylabel(
    #     f"Coarse {axis_label}",
    #     fontsize=18,
    # )

    ax_fine.set_ylim(fine_ylim)
    ax_coarse.set_ylim(coarse_ylim)

    ax_fine.yaxis.set_major_formatter(
        FormatStrFormatter("%.5f")
    )

    ax_coarse.yaxis.set_major_formatter(
        FormatStrFormatter("%.5f")
    )

    # X-axis tick labels
    ax_fine.tick_params(axis="x", labelsize=12)

    # Hide y-axis tick numbers
    ax_fine.tick_params(axis="y", labelleft=False)
    ax_coarse.tick_params(axis="y", labelright=False)


    # ax_fine.tick_params(
    #     axis="both",
    #     labelsize=12,
    # )

    # ax_coarse.tick_params(
    #     axis="y",
    #     labelsize=12,
    # )

    # Grid follows Fine / left axis
    ax_fine.grid(
        True,
        axis="y",
        alpha=0.25,
    )

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
        )

    plt.show()
    return fig, ax_fine





def plot_ever_inundation_confusion(
    u_true,
    u_pred,
    bed=None,
    print_results=False,                # NEW: bed topo tensor/array (nx,ny) or (N,nx,ny)
    sample_idx=None,         # if int -> plot that sample; if None -> plot across all samples (mode per pixel)
    inund_th=0.01,
    stride_t=None,           # e.g. 4 to use [..., ::4]; None = no striding
    extent=None,             # e.g. [0, X_range, 0, Y_range]; None = pixel coords
    title_prefix="",
    figsize=(8, 6),

    # --- NEW hillshade controls ---
    hillshade=True,
    hill_azdeg=315,
    hill_altdeg=45,
    hill_vert_exag=2.5,
    hill_dx=1.0,
    hill_dy=1.0,
    bed_alpha=1.0,
    overlay_alpha=1.0,      # make overlay slightly transparent to see relief
):
    """
    Plot TN/FP/FN/TP confusion map for ever-inundated (max over time > inund_th),
    optionally overlaid on 3D-looking hillshaded bed topography.

    u_true/u_pred can be torch tensors or numpy arrays.
    Expected shapes:
      - single sample: (nx, ny, nt)
      - many samples:  (N, nx, ny, nt)

    bed (optional):
      - single bed: (nx, ny)
      - per-sample: (N, nx, ny)
    """

    # ---- helpers ----
    def as_tensor(x):
        if x is None:
            return None
        if isinstance(x, torch.Tensor):
            return x
        return torch.from_numpy(np.asarray(x))

    def fmt_int(n: int) -> str:
        return f"{n:,}"

    ut = as_tensor(u_true)
    up = as_tensor(u_pred)
    bd = as_tensor(bed)

    # ---- optional time stride ----
    if stride_t is not None:
        ut = ut[..., ::stride_t]
        up = up[..., ::stride_t]

    # ---- ensure shapes match ----
    if ut.shape != up.shape:
        raise ValueError(f"Shape mismatch: true {tuple(ut.shape)} vs pred {tuple(up.shape)}")

    # ---- colormap ----
    cmap = ListedColormap(["#d9d9d9", "#ff0000", "#66d9ff", "#08306b"])  # TN, FP, FN, TP
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)

    legend_patches = [
        mpatches.Patch(color="#d9d9d9", label="Correct dry"),
        mpatches.Patch(color="#ff0000", label="False alarm"),
        mpatches.Patch(color="#66d9ff", label="Missed flood"),
        mpatches.Patch(color="#08306b", label="Correct wet"),
    ]

    # ---- compute & plot ----
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # =========================
    # 0) Background: hillshade bed (optional)
    # =========================
    if bd is not None and hillshade:
        # pick correct bed slice
        if bd.ndim == 3:
            if sample_idx is None:
                # across-samples mode map: pick mean bed for visualization
                bed_plot = bd.float().mean(dim=0)
            else:
                bed_plot = bd[sample_idx]
        elif bd.ndim == 2:
            bed_plot = bd
        else:
            raise ValueError(f"Unsupported bed ndim: {bd.ndim}. Expected (nx,ny) or (N,nx,ny).")

        bed_np = bed_plot.detach().cpu().numpy()

        # hillshade
        ls = LightSource(azdeg=hill_azdeg, altdeg=hill_altdeg)
        hill = ls.hillshade(bed_np, vert_exag=hill_vert_exag, dx=hill_dx, dy=hill_dy)  # [0..1]

                # --- make extent be in km if provided in meters ---
        if extent is not None:
            extent = [extent[0] / 1000, extent[1] / 1000, extent[2] / 1000, extent[3] / 1000]


        # plot hillshade as grayscale relief
        ax.imshow(
            hill.T, extent=extent, origin="lower",
            cmap="gist_yarg", vmin=0, vmax=1,
            alpha=bed_alpha, interpolation="bilinear", zorder=1
        )

    # =========================
    # 1) Confusion map overlay
    # =========================
    if sample_idx is not None:
        if ut.ndim == 4:
            ut_s = ut[sample_idx]
            up_s = up[sample_idx]
        elif ut.ndim == 3:
            ut_s = ut
            up_s = up
        else:
            raise ValueError(f"Unsupported ndim for sample plot: {ut.ndim}")

        true_max = torch.max(ut_s, dim=-1).values
        pred_max = torch.max(up_s, dim=-1).values

        true_wet = true_max > inund_th
        pred_wet = pred_max > inund_th

        tn = (~true_wet) & (~pred_wet)
        fp = (~true_wet) & ( pred_wet)
        fn = ( true_wet) & (~pred_wet)
        tp = ( true_wet) & ( pred_wet)

        # map codes
        codes = torch.zeros_like(true_wet, dtype=torch.uint8)
        codes[fp] = 1
        codes[fn] = 2
        codes[tp] = 3
        img = codes.detach().cpu().numpy()

        ax.imshow(
            img.T, extent=extent, origin="lower",
            cmap=cmap, norm=norm, interpolation="nearest",
            alpha=overlay_alpha, zorder=2
        )
        ax.set_title(f"{title_prefix}Ever-inundation confusion (sample={sample_idx}, th={inund_th} m)", fontsize=14)

        TN = int(tn.sum().item())
        FP = int(fp.sum().item())
        FN = int(fn.sum().item())
        TP = int(tp.sum().item())

        header = f"Confusion counts (sample={sample_idx}, th={inund_th} m)"
        total = TN + FP + FN + TP

    else:
        if ut.ndim != 4:
            raise ValueError("For across-samples plot, expected shape (N, nx, ny, nt).")

        true_max = torch.max(ut, dim=-1).values
        pred_max = torch.max(up, dim=-1).values

        true_wet = true_max > inund_th
        pred_wet = pred_max > inund_th

        tn = (~true_wet) & (~pred_wet)
        fp = (~true_wet) & ( pred_wet)
        fn = ( true_wet) & (~pred_wet)
        tp = ( true_wet) & ( pred_wet)

        TN = int(tn.sum().item())
        FP = int(fp.sum().item())
        FN = int(fn.sum().item())
        TP = int(tp.sum().item())

        header = f"Confusion counts (ALL samples + pixels, th={inund_th} m)"
        total = TN + FP + FN + TP

        # mode map
        counts = torch.stack([tn.sum(0), fp.sum(0), fn.sum(0), tp.sum(0)], dim=0)
        mode_map = torch.argmax(counts, dim=0).to(torch.uint8).detach().cpu().numpy()

        ax.imshow(
            mode_map.T, extent=extent, origin="lower",
            cmap=cmap, norm=norm, interpolation="nearest",
            alpha=overlay_alpha, zorder=2
        )
        ax.set_title(f"{title_prefix}Across-samples confusion (MODE per pixel, th={inund_th} m)", fontsize=14)

    # ---- legend ----
    ax.legend(
        handles=legend_patches,
        loc="upper right",
        frameon=True,
        fontsize=10,
        title="Confusion classes",
        title_fontsize=10,
    )
    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")

    # ---- metrics printout ----
    cm = np.array([[TN, FP],
                   [FN, TP]], dtype=np.int64)

    def pct(num, den):
        return 100.0 * num / den if den > 0 else float("nan")

    true_wet_ct = TP + FN
    true_dry_ct = TN + FP
    pred_wet_ct = TP + FP
    pred_dry_ct = TN + FN
    total_ct    = TP + TN + FP + FN

    wet_recall     = TP / true_wet_ct if true_wet_ct > 0 else float("nan")
    wet_precision  = TP / pred_wet_ct if pred_wet_ct > 0 else float("nan")
    dry_recall     = TN / true_dry_ct if true_dry_ct > 0 else float("nan")
    fpr            = FP / true_dry_ct if true_dry_ct > 0 else float("nan")
    fnr            = FN / true_wet_ct if true_wet_ct > 0 else float("nan")
    csi            = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else float("nan")

    f1 = (2 * wet_precision * wet_recall / (wet_precision + wet_recall)
          if np.isfinite(wet_precision) and np.isfinite(wet_recall) and (wet_precision + wet_recall) > 0
          else float("nan"))
    if print_results:
        print("\n" + header)
        print("-" * len(header))
        print(f"TP (Correct wet)      : {fmt_int(TP)}")
        print(f"TN (Correct dry)      : {fmt_int(TN)}")
        print(f"FP (False alarm wet)  : {fmt_int(FP)}")
        print(f"FN (Missed wet)       : {fmt_int(FN)}")
        print(f"True wet  (TP+FN)     : {fmt_int(true_wet_ct)}")
        print(f"True dry  (TN+FP)     : {fmt_int(true_dry_ct)}")
        print(f"Pred wet  (TP+FP)     : {fmt_int(pred_wet_ct)}")
        print(f"Pred dry  (TN+FN)     : {fmt_int(pred_dry_ct)}")
        print(f"Total pixels          : {fmt_int(total_ct)}")
        print()
        print(f"Wet Recall / POD  (TP/(TP+FN)) : {wet_recall:.4f}  ({pct(TP, true_wet_ct):6.2f}%)")
        print(f"Wet Precision (TP/(TP+FP))    : {wet_precision:.4f}  ({pct(TP, pred_wet_ct):6.2f}%)")
        print(f"Dry Recall / TNR  (TN/(TN+FP)) : {dry_recall:.4f}  ({pct(TN, true_dry_ct):6.2f}%)")
        print(f"\nOveral performance\n")
        print(f"False Alarm Rate  (FP/(TN+FP)) : {fpr:.4f}  ({pct(FP, true_dry_ct):6.2f}%)")
        print(f"Miss Rate         (FN/(TP+FN)) : {fnr:.4f}  ({pct(FN, true_wet_ct):6.2f}%)")
        print(f"CSI / IoU         (TP/(TP+FP+FN)) : {csi:.4f}")
        print(f"F1-score (wet)     (2PR/(P+R))   : {f1:.4f}")

    return fig, ax





import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from datetime import timedelta


def plot_detection_skill_time(
    dynamic_metrics,
    start_datetime,
    end_datetime,
    figsize=(6, 4),
    legend_loc=(0.95, 0.6),
    time_unit="auto",
):
    """
    Plot CSI, POD, and FAR over time.

    Automatically selects hourly or daily x-axis formatting.

    Parameters
    ----------
    dynamic_metrics : dict
        Contains 'CSI_t', 'POD_t', and 'FAR_t'.
    start_datetime : datetime
        Start of the time period.
    end_datetime : datetime
        End of the time period.
    figsize : tuple
        Figure size.
    legend_loc : tuple
        Legend anchor position.
    time_unit : str
        'auto', 'hourly', or 'daily'.

    Returns
    -------
    fig, ax
    """

    # ========================================================
    # 1. PREPARE DATA
    # ========================================================

    def to_numpy(x):
        if hasattr(x, "detach"):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    CSI = to_numpy(dynamic_metrics["CSI_t"]) * 100
    POD = to_numpy(dynamic_metrics["POD_t"]) * 100
    FAR = to_numpy(dynamic_metrics["FAR_t"]) * 100

    nt = len(CSI)

    if nt == 0 or len(POD) != nt or len(FAR) != nt:
        raise ValueError("CSI, POD, and FAR must have the same nonzero length.")

    if end_datetime <= start_datetime:
        raise ValueError("end_datetime must be after start_datetime.")

    # ========================================================
    # 2. CREATE TIME AXIS
    # ========================================================

    duration = end_datetime - start_datetime

    time_axis = np.linspace(
        mdates.date2num(start_datetime),
        mdates.date2num(end_datetime),
        nt,
    )

    if time_unit == "auto":
        time_unit = "hourly" if duration <= timedelta(days=2) else "daily"

    if time_unit not in ("hourly", "daily"):
        raise ValueError("time_unit must be 'auto', 'hourly', or 'daily'.")

    # ========================================================
    # 3. PLOT METRICS
    # ========================================================

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(
        time_axis, CSI,
        linewidth=2,
        label="CSI",
    )

    ax.plot(
        time_axis, POD,
        linewidth=2,
        label="POD",
    )

    ax.fill_between(
        time_axis, FAR,
        alpha=0.4,
        color="red",
        label="FAR",
    )

    ax.set_ylabel("Detection Skill (%)")
    ax.set_ylim(0, 100)
    ax.set_xlim(start_datetime, end_datetime)

    # ========================================================
    # 4. AUTOMATIC DATE FORMATTING
    # ========================================================

    if time_unit == "hourly":

        total_hours = duration.total_seconds() / 3600

        if total_hours <= 6:
            hour_interval = 1
        elif total_hours <= 12:
            hour_interval = 2
        elif total_hours <= 24:
            hour_interval = 4
        else:
            hour_interval = 6

        ax.xaxis.set_major_locator(
            mdates.HourLocator(interval=hour_interval)
        )

        ax.xaxis.set_major_formatter(
            mdates.DateFormatter("%d %b\n%H:%M")
        )

        ax.set_xlabel("Time")

    else:

        total_days = duration.total_seconds() / 86400
        day_interval = max(1, int(np.ceil(total_days / 8)))

        ax.xaxis.set_major_locator(
            mdates.DayLocator(interval=day_interval)
        )

        ax.xaxis.set_major_formatter(
            mdates.DateFormatter("%d %b\n%Y")
        )

        ax.set_xlabel("Date")

    # ========================================================
    # 5. FINAL FORMATTING
    # ========================================================

    ax.grid(alpha=0.3)

    ax.legend(
        bbox_to_anchor=legend_loc,
        ncol=3,
    )

    ax.tick_params(axis="x", labelsize=9, rotation=0)

    fig.tight_layout()

    return fig, ax





import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from datetime import timedelta


def plot_inundation_extent_time(
    extent_true,
    extent_pred,
    idx,
    start_datetime,
    end_datetime,
    figsize=(6, 4),
    legend_loc=(0.95, 0.6),
    time_unit="auto",
):
    """
    Plot inundation extent and absolute relative error over time.

    Automatically selects hourly or daily x-axis formatting.

    Parameters
    ----------
    extent_true : torch.Tensor or np.ndarray
        Shape [nb, nt].
    extent_pred : torch.Tensor or np.ndarray
        Shape [nb, nt].
    idx : int
        Batch index to visualize.
    start_datetime : datetime
        Start of the time period.
    end_datetime : datetime
        End of the time period.
    figsize : tuple
        Figure size.
    legend_loc : tuple
        Legend anchor position.
    time_unit : str
        'auto', 'hourly', or 'daily'.

    Returns
    -------
    fig, ax1, ax2
    """

    # ========================================================
    # 1. PREPARE DATA
    # ========================================================

    def to_numpy(x):
        if hasattr(x, "detach"):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    true = to_numpy(extent_true)[idx] / 1e6  # km²
    pred = to_numpy(extent_pred)[idx] / 1e6  # km²

    nt = len(true)

    if nt == 0 or len(pred) != nt:
        raise ValueError(
            "Reference and predicted extents must have the same nonzero length."
        )

    if end_datetime <= start_datetime:
        raise ValueError(
            "end_datetime must be after start_datetime."
        )

    # ========================================================
    # 2. CREATE TIME AXIS
    # ========================================================

    duration = end_datetime - start_datetime

    time_axis = np.linspace(
        mdates.date2num(start_datetime),
        mdates.date2num(end_datetime),
        nt,
    )

    if time_unit == "auto":
        time_unit = "hourly" if duration <= timedelta(days=2) else "daily"

    if time_unit not in ("hourly", "daily"):
        raise ValueError(
            "time_unit must be 'auto', 'hourly', or 'daily'."
        )

    # ========================================================
    # 3. CALCULATE RELATIVE ERROR
    # ========================================================

    rel_error = (
        np.abs(pred - true) / (true + 1e-8) * 100
    )

    # ========================================================
    # 4. PLOT INUNDATION EXTENT
    # ========================================================

    fig, ax1 = plt.subplots(figsize=figsize)

    ax1.plot(
        time_axis,
        true,
        color="black",
        linewidth=2,
        label="Reference",
    )

    ax1.plot(
        time_axis,
        pred,
        color="tab:blue",
        linewidth=2,
        label="Predicted",
    )

    ax1.set_ylabel("Inundation Extent (km$^2$)")

    # ========================================================
    # 5. PLOT RELATIVE ERROR (SECONDARY AXIS)
    # ========================================================

    ax2 = ax1.twinx()

    ax2.plot(
        time_axis,
        rel_error,
        linestyle="--",
        color="red",
        linewidth=2,
        label="Absolute Relative Error",
    )

    ax2.set_ylabel("Absolute Relative Error (%)")

    # ========================================================
    # 6. AUTOMATIC DATE FORMATTING
    # ========================================================

    ax1.set_xlim(start_datetime, end_datetime)

    if time_unit == "hourly":

        total_hours = duration.total_seconds() / 3600

        if total_hours <= 6:
            hour_interval = 1
        elif total_hours <= 12:
            hour_interval = 2
        elif total_hours <= 24:
            hour_interval = 4
        else:
            hour_interval = 6

        ax1.xaxis.set_major_locator(
            mdates.HourLocator(interval=hour_interval)
        )

        ax1.xaxis.set_major_formatter(
            mdates.DateFormatter("%d %b\n%H:%M")
        )

        ax1.set_xlabel("Time")

    else:

        total_days = duration.total_seconds() / 86400
        day_interval = max(1, int(np.ceil(total_days / 8)))

        ax1.xaxis.set_major_locator(
            mdates.DayLocator(interval=day_interval)
        )

        ax1.xaxis.set_major_formatter(
            mdates.DateFormatter("%d %b\n%Y")
        )

        ax1.set_xlabel("Date")

    # ========================================================
    # 7. FINAL FORMATTING
    # ========================================================

    ax1.grid(alpha=0.3)

    ax1.tick_params(
        axis="x",
        labelsize=9,
        rotation=0,
    )

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()

    ax1.legend(
        lines1 + lines2,
        labels1 + labels2,
        bbox_to_anchor=legend_loc,
    )

    fig.tight_layout()

    return fig, ax1, ax2


    


import matplotlib.pyplot as plt
import numpy as np


def plot_metrics(metrics, figsize=(15, 4), plot=False):

    groups = {
        "Relative RMSE": [
            "relRMSE",
            "relRMSE_shallow",
            "relRMSE_deep",
        ],
        "NSE": [
            "NSE",
            "NSE_shallow",
            "NSE_deep",
        ],
        "Flood Detection": [
            "POD",
            "FAR",
            "CSI",
        ],
    }

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    for ax, (title, names) in zip(axes, groups.items()):

        means = [metrics[f"{name}_mean"] for name in names]
        stds = [metrics[f"{name}_std"] for name in names]

        x = np.arange(len(names))

        ax.bar(
            x,
            means,
            yerr=stds,
            capsize=5,
            alpha=0.8,
        )

        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=25, ha="right")
        ax.set_title(title)
        ax.set_ylabel("Metric value")
        ax.grid(axis="y", alpha=0.3)

        for i, (mean, std) in enumerate(zip(means, stds)):
            ax.annotate(
                f"{mean:.4f} ± {std:.4f}",
                (i, mean + std),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )

        ax.margins(y=0.15)
    if plot:
        plt.tight_layout()  
        plt.show()
    return fig, axes


import torch

def evaluate_flood_metrics(
    Depth_True,
    Depth_Pred,
    wet_threshold=0.025,
    peak_percent=0.05,
    eps=1e-12
):
    """
    Depth_True, Depth_Pred: torch tensors [nbatch, nx, ny, nt]
    Returns:
        {
            "static": {...},
            "dynamic": {...}
        }
    """

    nb, nx, ny, nt = Depth_True.shape

    # ------------------------------------------
    # Containers
    # ------------------------------------------
    relrmse_batch = []
    nse_batch = []

    # NEW: depth-stratified static metrics
    relrmse_shallow_batch = []  # h < 0.5
    nse_shallow_batch = []
    relrmse_deep_batch = []     # h >= 0.5
    nse_deep_batch = []

    # NEW: static binary metrics (aggregated over all space-time per sample)
    pod_batch = []
    far_batch = []
    csi_batch = []

    peak_val_err_batch = []
    peak_time_err1_batch = []
    peak_time_err2_batch = []

    relrmse_t = torch.zeros(nb, nt, device=Depth_True.device)
    nse_t = torch.zeros(nb, nt, device=Depth_True.device)
    pod_t = torch.zeros(nb, nt, device=Depth_True.device)
    far_t = torch.zeros(nb, nt, device=Depth_True.device)
    csi_t = torch.zeros(nb, nt, device=Depth_True.device)

    # ------------------------------------------
    # Loop over batch
    # ------------------------------------------
    for b in range(nb):

        O = Depth_True[b]   # [nx, ny, nt]
        P = Depth_Pred[b]

        # --------------------------------------
        # Global relRMSE & NSE
        # --------------------------------------
        diff = O - P
        rmse = torch.sqrt(torch.mean(diff**2))
        mean_O = torch.mean(O)

        relrmse_batch.append(rmse / (mean_O + eps))

        nse = 1 - torch.sum(diff**2) / (torch.sum((O - mean_O)**2) + eps)
        nse_batch.append(nse)

        # --------------------------------------
        # NEW: Depth-stratified static relRMSE & NSE
        # --------------------------------------
        shallow_mask = O < 0.5
        deep_mask = O >= 0.5

        # Shallow
        n_sh = shallow_mask.sum()
        if n_sh > 0:
            O_sh = O[shallow_mask]
            P_sh = P[shallow_mask]
            diff_sh = O_sh - P_sh

            rmse_sh = torch.sqrt(torch.mean(diff_sh**2))
            mean_O_sh = torch.mean(O_sh)
            relrmse_shallow_batch.append(rmse_sh / (mean_O_sh + eps))

            nse_sh = 1 - torch.sum(diff_sh**2) / (torch.sum((O_sh - mean_O_sh)**2) + eps)
            nse_shallow_batch.append(nse_sh)
        else:
            relrmse_shallow_batch.append(torch.tensor(float("nan"), device=Depth_True.device))
            nse_shallow_batch.append(torch.tensor(float("nan"), device=Depth_True.device))

        # Deep
        n_dp = deep_mask.sum()
        if n_dp > 0:
            O_dp = O[deep_mask]
            P_dp = P[deep_mask]
            diff_dp = O_dp - P_dp

            rmse_dp = torch.sqrt(torch.mean(diff_dp**2))
            mean_O_dp = torch.mean(O_dp)
            relrmse_deep_batch.append(rmse_dp / (mean_O_dp + eps))

            nse_dp = 1 - torch.sum(diff_dp**2) / (torch.sum((O_dp - mean_O_dp)**2) + eps)
            nse_deep_batch.append(nse_dp)
        else:
            relrmse_deep_batch.append(torch.tensor(float("nan"), device=Depth_True.device))
            nse_deep_batch.append(torch.tensor(float("nan"), device=Depth_True.device))

        # --------------------------------------
        # NEW: Static binary inundation metrics (aggregate over all time)
        # --------------------------------------
        wet_O_all = O > wet_threshold
        wet_P_all = P > wet_threshold

        TP_all = torch.sum(wet_O_all & wet_P_all).float()
        FP_all = torch.sum(~wet_O_all & wet_P_all).float()
        FN_all = torch.sum(wet_O_all & ~wet_P_all).float()

        pod_batch.append(TP_all / (TP_all + FN_all + eps))
        far_batch.append(FP_all / (TP_all + FP_all + eps))
        csi_batch.append(TP_all / (TP_all + FP_all + FN_all + eps))

        # --------------------------------------
        # Dynamic metrics over time
        # --------------------------------------
        for t in range(nt):

            Ot = O[:, :, t]
            Pt = P[:, :, t]

            diff_t = Ot - Pt
            rmse_t_val = torch.sqrt(torch.mean(diff_t**2))
            mean_Ot = torch.mean(Ot)

            relrmse_t[b, t] = rmse_t_val / (mean_Ot + eps)
            nse_t[b, t] = 1 - torch.sum(diff_t**2) / (
                torch.sum((Ot - mean_Ot)**2) + eps
            )

            # Binary inundation
            wet_O = Ot > wet_threshold
            wet_P = Pt > wet_threshold

            TP = torch.sum(wet_O & wet_P).float()
            FP = torch.sum(~wet_O & wet_P).float()
            FN = torch.sum(wet_O & ~wet_P).float()

            pod_t[b, t] = TP / (TP + FN + eps)
            far_t[b, t] = FP / (TP + FP + eps)
            csi_t[b, t] = TP / (TP + FP + FN + eps)

        # --------------------------------------
        # Peak metrics (volume-based)
        # --------------------------------------
        H_true = torch.sum(O, dim=(0, 1))  # [nt]
        H_pred = torch.sum(P, dim=(0, 1))

        n_peak = max(1, int(nt * peak_percent))

        peak_indices = torch.topk(H_true, n_peak).indices

        H_true_peak = torch.mean(H_true[peak_indices])
        H_pred_peak = torch.mean(H_pred[peak_indices])

        peak_val_err_batch.append(
            (H_pred_peak - H_true_peak) / (H_true_peak + eps)
        )

        t_pred_peak = torch.mean(torch.topk(H_pred, n_peak).indices.float())
        t_ref_peak = torch.mean(peak_indices.float())

        # relPeakTimeErr-1
        peak_duration = (
            torch.max(peak_indices.float())
            - torch.min(peak_indices.float())
            + eps
        )

        peak_time_err1_batch.append(
            (t_pred_peak - t_ref_peak) / peak_duration
        )

        # relPeakTimeErr-2
        H_min = torch.min(H_true)
        rise_threshold = H_min + 0.10 * (H_true_peak - H_min)

        rise_indices = torch.where(H_true >= rise_threshold)[0]
        t_rise = rise_indices[0].float() if len(rise_indices) > 0 else torch.tensor(0.0, device=Depth_True.device)

        rise_duration = (t_ref_peak - t_rise) + eps

        peak_time_err2_batch.append(
            (t_pred_peak - t_ref_peak) / rise_duration
        )

    # ------------------------------------------
    # Stack batch metrics
    # ------------------------------------------
    relrmse_batch = torch.stack(relrmse_batch)
    nse_batch = torch.stack(nse_batch)

    # NEW
    relrmse_shallow_batch = torch.stack(relrmse_shallow_batch)
    nse_shallow_batch = torch.stack(nse_shallow_batch)
    relrmse_deep_batch = torch.stack(relrmse_deep_batch)
    nse_deep_batch = torch.stack(nse_deep_batch)

    # NEW: static binary
    pod_batch = torch.stack(pod_batch)
    far_batch = torch.stack(far_batch)
    csi_batch = torch.stack(csi_batch)

    peak_val_err_batch = torch.stack(peak_val_err_batch)
    peak_time_err1_batch = torch.stack(peak_time_err1_batch)
    peak_time_err2_batch = torch.stack(peak_time_err2_batch)

    # Helpers to ignore NaNs (in case a sample has no shallow/deep pixels)
    def nanmean(x):
        mask = ~torch.isnan(x)
        return x[mask].mean().item() if mask.any() else float("nan")

    def nanstd(x):
        mask = ~torch.isnan(x)
        return x[mask].std().item() if mask.any() else float("nan")

    # ------------------------------------------
    # Static (mean ± std over batch)
    # ------------------------------------------
    static_metrics = {
        "relRMSE_mean": relrmse_batch.mean().item(),
        "relRMSE_std": relrmse_batch.std().item(),

        "NSE_mean": nse_batch.mean().item(),
        "NSE_std": nse_batch.std().item(),

        # NEW: shallow (<0.5m)
        "relRMSE_shallow_mean": nanmean(relrmse_shallow_batch),
        "relRMSE_shallow_std": nanstd(relrmse_shallow_batch),
        "NSE_shallow_mean": nanmean(nse_shallow_batch),
        "NSE_shallow_std": nanstd(nse_shallow_batch),

        # NEW: deep (>=0.5m)
        "relRMSE_deep_mean": nanmean(relrmse_deep_batch),
        "relRMSE_deep_std": nanstd(relrmse_deep_batch),
        "NSE_deep_mean": nanmean(nse_deep_batch),
        "NSE_deep_std": nanstd(nse_deep_batch),

        # NEW: static binary inundation metrics
        "POD_mean": pod_batch.mean().item(),
        "POD_std": pod_batch.std().item(),
        "FAR_mean": far_batch.mean().item(),
        "FAR_std": far_batch.std().item(),
        "CSI_mean": csi_batch.mean().item(),
        "CSI_std": csi_batch.std().item(),

        "relPeakValErr_mean": peak_val_err_batch.mean().item(),
        "relPeakValErr_std": peak_val_err_batch.std().item(),

        "relPeakTimeErr1_mean": peak_time_err1_batch.mean().item(),
        "relPeakTimeErr1_std": peak_time_err1_batch.std().item(),

        "relPeakTimeErr2_mean": peak_time_err2_batch.mean().item(),
        "relPeakTimeErr2_std": peak_time_err2_batch.std().item(),
    }

    # ------------------------------------------
    # Dynamic (mean over batch)
    # ------------------------------------------
    dynamic_metrics = {
        "relRMSE_t": relrmse_t.mean(dim=0),
        "NSE_t": nse_t.mean(dim=0),
        "POD_t": pod_t.mean(dim=0),
        "FAR_t": far_t.mean(dim=0),
        "CSI_t": csi_t.mean(dim=0),
    }

    return {
        "static": static_metrics,
        "dynamic": dynamic_metrics
    }




def inundation_extent_timeseries(
    depth: torch.Tensor,
    Lx: float,
    Ly: float,
    wet_threshold: float = 0.025,
) -> torch.Tensor:
    """
    Inundation extent (area) over time from depth.

    Supports:
      - depth: [nx, ny, nt]      -> returns [nt]
      - depth: [nb, nx, ny, nt]  -> returns [nb, nt]

    Units: m^2 (uniform grid assumed with dx=Lx/nx, dy=Ly/ny)
    """
    if depth.ndim == 3:
        nx, ny, nt = depth.shape
        dx, dy = Lx / nx, Ly / ny
        cell_area = dx * dy
        wet_cells_t = (depth > wet_threshold).sum(dim=(0, 1)).float()  # [nt]
        return wet_cells_t * cell_area

    if depth.ndim == 4:
        nb, nx, ny, nt = depth.shape
        dx, dy = Lx / nx, Ly / ny
        cell_area = dx * dy
        wet_cells_bt = (depth > wet_threshold).sum(dim=(1, 2)).float()  # [nb, nt]
        return wet_cells_bt * cell_area

    raise ValueError(f"Expected depth with 3 or 4 dims, got shape {tuple(depth.shape)}")
