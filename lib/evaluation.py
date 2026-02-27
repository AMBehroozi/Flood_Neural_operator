import torch

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
from datetime import datetime
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm, LightSource
from datetime import datetime, timedelta

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




def plot_detection_skill_time(
    dynamic_metrics,
    start_datetime,
    end_datetime,
    figsize=(6,4),
    legend_loc=(0.95, 0.6)
):
    """
    Plot CSI, POD, and FAR over time with date axis.

    Parameters
    ----------
    dynamic_metrics : dict
        Should contain:
            'CSI_t', 'POD_t', 'FAR_t'
    start_datetime : datetime
    end_datetime   : datetime
    figsize : tuple
        Figure size
    legend_loc : tuple
        bbox_to_anchor location for legend

    Returns
    -------
    fig, ax
    """

    # Convert to numpy safely (works for torch or numpy)
    def to_numpy(x):
        if hasattr(x, "detach"):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    CSI = to_numpy(dynamic_metrics['CSI_t']) * 100
    POD = to_numpy(dynamic_metrics['POD_t']) * 100
    FAR = to_numpy(dynamic_metrics['FAR_t']) * 100

    nt = len(CSI)

    # Create datetime axis
    time_axis = np.linspace(
        mdates.date2num(start_datetime),
        mdates.date2num(end_datetime),
        nt
    )

    # Plot
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot_date(time_axis, CSI, '-', linewidth=2, label='CSI')
    ax.plot_date(time_axis, POD, '-', linewidth=2, label='POD')
    ax.fill_between(time_axis, FAR, alpha=0.4, color='red', label='FAR')

    ax.set_ylabel('Detection Skill (%)')
    ax.set_ylim(0, 100)

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b\n%Y'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())

    ax.set_xlabel('Date')
    ax.grid(alpha=0.3)

    ax.legend(bbox_to_anchor=legend_loc, ncol=3)

    fig.autofmt_xdate()
    plt.tight_layout()

    return fig, ax






def plot_inundation_extent_time(
    extent_true,
    extent_pred,
    idx,
    start_datetime,
    end_datetime,
    figsize=(6,4),
    legend_loc=(0.95, 0.6)
):
    """
    Plot inundation extent and relative error over time.

    Parameters
    ----------
    extent_true : torch.Tensor or np.ndarray
        Shape [nb, nt]
    extent_pred : torch.Tensor or np.ndarray
        Shape [nb, nt]
    idx : int
        Batch index to visualize
    start_datetime : datetime
    end_datetime   : datetime
    figsize : tuple
    legend_loc : tuple

    Returns
    -------
    fig, ax1, ax2
    """

    # --- Safe conversion to numpy ---
    def to_numpy(x):
        if hasattr(x, "detach"):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    true = to_numpy(extent_true)[idx] / 1e6  # km^2
    pred = to_numpy(extent_pred)[idx] / 1e6  # km^2

    nt = len(true)

    # Create datetime axis
    time_axis = np.linspace(
        mdates.date2num(start_datetime),
        mdates.date2num(end_datetime),
        nt
    )

    rel_error = np.abs(pred - true) / (true + 1e-8) * 100

    # --- Plot ---
    fig, ax1 = plt.subplots(figsize=figsize)

    ax1.plot_date(time_axis, true, '-', color='black',
                  linewidth=2, label='Reference')
    ax1.plot_date(time_axis, pred, '-', color='tab:blue',
                  linewidth=2, label='Predicted')

    ax1.set_ylabel('Inundation Extent (km$^2$)')

    ax2 = ax1.twinx()
    ax2.plot_date(time_axis, rel_error, '--',
                  color='red', linewidth=2,
                  label='Absolute Relative Error')
    ax2.set_ylabel('Absolute Relative Error (%)')

    ax1.set_xlabel('Date')
    ax1.grid(alpha=0.3)

    # Date formatting
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d %b\n%Y'))
    ax1.xaxis.set_major_locator(mdates.AutoDateLocator())

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               bbox_to_anchor=legend_loc)

    fig.autofmt_xdate()
    plt.tight_layout()

    return fig, ax1, ax2




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
