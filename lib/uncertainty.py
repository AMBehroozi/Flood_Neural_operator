import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import torch

def compute_inundation_probability(u_true, u_pred, h0=0.025):
    """
    Now returns probabilities over ALL times + original t_peak
    """
    H = u_true.sum(dim=(1, 2))
    t_peak = int(torch.argmax(H.mean(dim=0)).item())

    wet_true = (u_true > h0).float()
    wet_pred = (u_pred > h0).float()

    P_solver = wet_true.mean(dim=0).cpu().numpy()   # [nx, ny, nt]
    P_model  = wet_pred.mean(dim=0).cpu().numpy()

    # → [nt, nx, ny]
    P_solver = P_solver.transpose(2, 0, 1)
    P_model  = P_model.transpose(2, 0, 1)

    return P_solver, P_model, t_peak




def reliability_barplot_calibration_gap(
    P_solver,                   # [nx, ny] or flattened array in [0,1]
    P_model,                    # [nx, ny] or flattened array in [0,1]
    Lx, Ly,                     # domain size: either scalar length (m) or tuple (xmin, xmax)
    nbins=10,
    title="Reliability Diagram",
    figsize=(6, 6),
    show_plot=True,
    fontsize=13,
):
    """
    Plots a reliability diagram with colored calibration gaps:
      - Red hatched  = overconfident (model > observed)
      - Green hatched = underconfident (model < observed)
    Uses area-weighted Expected Calibration Error (ECE).
    """
    # --- Domain area calculation ---
    if isinstance(Lx, (tuple, list, np.ndarray)) and len(Lx) == 2:
        Lx_len = float(Lx[1] - Lx[0])
    else:
        Lx_len = float(Lx)

    if isinstance(Ly, (tuple, list, np.ndarray)) and len(Ly) == 2:
        Ly_len = float(Ly[1] - Ly[0])
    else:
        Ly_len = float(Ly)

    total_area_km2 = (Lx_len * Ly_len) / 1e6

    # --- Flatten & clean ---
    P_solver = np.asarray(P_solver).ravel()
    P_model  = np.asarray(P_model).ravel()

    valid = np.isfinite(P_solver) & np.isfinite(P_model)
    P_solver = P_solver[valid]
    P_model  = P_model[valid]

    if len(P_solver) == 0:
        raise ValueError("No valid data points after removing NaNs/inf.")

    npts = len(P_solver)
    weights = np.full(npts, total_area_km2 / npts)

    # --- Binning ---
    bins = np.linspace(0, 1, nbins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_ids = np.digitize(P_model, bins) - 1
    bin_ids = np.clip(bin_ids, 0, nbins - 1)

    conf = np.full(nbins, np.nan)
    acc  = np.full(nbins, np.nan)
    area = np.zeros(nbins)

    for b in range(nbins):
        mask = (bin_ids == b)
        if not mask.any():
            continue
        w_bin = weights[mask]
        area[b] = w_bin.sum()
        conf[b] = np.average(P_model[mask], weights=w_bin)
        acc[b]  = np.average(P_solver[mask], weights=w_bin)

    # --- Area-weighted ECE ---
    area = np.nan_to_num(area, nan=0.0)
    total_area = area.sum()
    if total_area > 1e-12:
        area_frac = area / total_area
        ece = np.nansum(area_frac * np.abs(acc - conf))
    else:
        ece = np.nan

    # --- Gap ---
    gap = conf - acc           # >0 = overconfident, <0 = underconfident
    gap_abs = np.abs(gap)
    colors = np.where(gap > 0, 'red', 'green')

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=figsize)

    width = 0.9 / nbins

    # Main bars: observed frequency
    ax.bar(bin_centers, acc, width=width, align='center',
           color='steelblue', alpha=0.92, label='Observed frequency')

    # Gap bars (hatched)
    for i in range(nbins):
        if np.isnan(gap[i]) or gap_abs[i] < 1e-9:
            continue
        ax.bar(bin_centers[i], gap_abs[i], width=width, align='center',
               bottom=acc[i], alpha=0.45, hatch='///',
               color=colors[i], edgecolor=colors[i], lw=0.8)

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', lw=1.6, label='Perfect calibration')

    # Legend entries for gaps using Rectangle patches
    legend_elements = [
        ax.bar(0, 0, color='red',   hatch='///', alpha=0.45, label='Overconfident (model > obs)')[0],
        ax.bar(0, 0, color='green', hatch='///', alpha=0.45, label='Underconfident (model < obs)')[0],
    ]

    ax.set_xlabel(r"Predicted probability $P_{\mathrm{model}}$ (wet)", fontsize=fontsize)
    ax.set_ylabel(r"Observed frequency $P_{\mathrm{solver}}$ (wet)", fontsize=fontsize)
    ax.set_title(f"{title}\nArea-weighted ECE = {ece*100:.2f}%", fontsize=fontsize+2, pad=12)

    ax.set_xlim(0, 1.04)
    ax.set_ylim(0, 1.04)
    ax.set_xticks(np.arange(0, 1.1, 0.2))
    ax.set_yticks(np.arange(0, 1.1, 0.2))
    ax.grid(True, alpha=0.18, linestyle='--')

    ax.legend(handles=[Rectangle((0,0),1,1,facecolor='steelblue',label='Observed frequency'),
                       Rectangle((0,0),1,1,facecolor='k',lw=1.6,ls='--',label='Perfect calibration'),
                       Rectangle((0,0),1,1,facecolor='red', hatch='///', alpha=0.45, label='Overconfident'),
                       Rectangle((0,0),1,1,facecolor='green', hatch='///', alpha=0.45, label='Underconfident')],
              loc='upper left', fontsize=fontsize-1, framealpha=0.95)

    plt.tight_layout()

    if show_plot:
        plt.show()

    return {
        'bin_centers': bin_centers,
        'acc': acc,
        'conf': conf,
        'gap': gap,
        'gap_abs': gap_abs,
        'area_km2': area,
        'ECE': ece,
        'fig': fig,
        'ax': ax,
    }

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def reliability_barplot_with_side_box_fit01(
    P_solver,
    P_model,
    Lx, Ly,
    nbins=10,
    title="Area-weighted Reliability Diagram",
    figsize=(7.8, 5.8),
    show_plot=True,
    fontsize=13,
):
    # ── Area calculation ────────────────────────────────────────────────────
    Lx_len = Lx[1] - Lx[0] if isinstance(Lx, (tuple, list)) else float(Lx)
    Ly_len = Ly[1] - Ly[0] if isinstance(Ly, (tuple, list)) else float(Ly)
    total_area_km2 = (Lx_len * Ly_len) / 1e6

    # ── Flatten & clean ─────────────────────────────────────────────────────
    P_solver = np.asarray(P_solver).ravel()
    P_model  = np.asarray(P_model).ravel()
    valid = np.isfinite(P_solver) & np.isfinite(P_model)
    P_solver = P_solver[valid]
    P_model  = P_model[valid]

    if len(P_solver) == 0:
        raise ValueError("No valid data.")

    weights = np.full(len(P_solver), total_area_km2 / len(P_solver))

    # ── Binning ─────────────────────────────────────────────────────────────
    bins       = np.linspace(0, 1, nbins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_ids    = np.clip(np.digitize(P_model, bins) - 1, 0, nbins - 1)

    conf = np.full(nbins, np.nan)
    acc  = np.full(nbins, np.nan)
    area = np.zeros(nbins)

    for b in range(nbins):
        m = bin_ids == b
        if not m.any():
            continue
        wb     = weights[m]
        area[b] = wb.sum()
        conf[b] = np.average(P_model[m], weights=wb)
        acc[b]  = np.average(P_solver[m], weights=wb)

    # ── ECE ─────────────────────────────────────────────────────────────────
    area_frac = area / area.sum() if area.sum() > 1e-12 else np.zeros(nbins)
    ece       = np.nansum(area_frac * np.abs(conf - acc))

    # ── Gaps ────────────────────────────────────────────────────────────────
    gap       = conf - acc
    gap_abs   = np.abs(gap)
    gap_valid = gap_abs[~np.isnan(gap_abs)]   # ← now absolute values
    # ── Main figure ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=figsize)
    width   = 0.89 / nbins

    # Bars
    ax.bar(bin_centers, acc, width=width, color='steelblue', alpha=0.92,
           label='Observed frequency')

    # Gap overlays
    colors = np.where(gap > 0, 'red', 'green')
    for i in range(nbins):
        if np.isnan(gap[i]) or gap_abs[i] < 1e-9:
            continue
        ax.bar(bin_centers[i], gap_abs[i], width=width,
               bottom=acc[i], alpha=0.45, hatch='///',
               color=colors[i], edgecolor=colors[i], lw=0.8)

    # Diagonal
    ax.plot([0, 1], [0, 1], 'k--', lw=1.8, label='Perfect calibration')

    # ── Inset axes for boxplot (own y-scale, right side, inside main plot) ──
    if len(gap_valid) > 1:
        # Position inset: [left, bottom, width, height] in axes-fraction coords
        # Placed in the right portion of the main axes
        ax_box = ax.inset_axes([1.0, 0.0, 0.1, 1.0])   # tweak as needed

        ax_box.boxplot(
            gap_valid * 100,
            positions=[0.5],
            widths=0.35,
            patch_artist=True,
            showfliers=False,
            boxprops=dict(facecolor='#fff5e6', edgecolor='#cc5500', lw=1.4),
            medianprops=dict(color='#cc5500', lw=2.2),
            whiskerprops=dict(color='#cc5500', lw=1.2),
            capprops=dict(color='#cc5500', lw=1.4),
            flierprops=dict(marker='o', ms=4, mfc='none', mec='#cc5500'),
        )

        # ── Inset axes styling ───────────────────────────────────────────────
        ax_box.set_xlim(0, 1)
        ax_box.set_xticks([])                          # hide x ticks
        ax_box.set_title("Gap(%)", fontsize=fontsize - 3, pad=3, color='#cc5500')

        # Y-axis ticks on the RIGHT, own scale
        ax_box.yaxis.set_label_position('right')
        ax_box.yaxis.tick_right()
        ax_box.tick_params(axis='y', labelsize=fontsize - 4, colors='#cc5500')

        # Subtle background so it sits visually inside the plot
        ax_box.set_facecolor('#fffdf8')
        for spine in ax_box.spines.values():
            spine.set_edgecolor('#cc5500')
            spine.set_linewidth(0.9)
            spine.set_alpha(0.6)

        # Zero reference line
        # ax_box.axhline(0, color='gray', lw=0.9, ls='--', alpha=0.7)
        # ax_box.axhline(ece * 100, color='gray', lw=0.9, ls='--', alpha=0.7)




    # ── Main axes styling ────────────────────────────────────────────────────
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(np.arange(0, 1.01, 0.2))
    ax.set_xlabel(r"Predicted probability $P_{\mathrm{model}}$ (wet)", fontsize=fontsize)
    ax.set_ylabel(r"Observed frequency $P_{\mathrm{solver}}$ (wet)", fontsize=fontsize)
    # ax.set_title(f"{title}\nECE = {ece*100:.2f}%", fontsize=fontsize + 3, pad=15)
    ax.grid(True, alpha=0.18, ls='--')
    ax.text(0.23, 0.75, f"{title}\nECE = {ece*100:.2f}%", fontsize=fontsize + 3, color='red', ha='center', va='center')
    legend_elements = [
        Rectangle((0, 0), 1, 1, facecolor='steelblue', label='Observed frequency'),
        Rectangle((0, 0), 1, 1, ls='--', lw=1.8, edgecolor='k',
                  facecolor='none', label='Perfect calibration'),
        Rectangle((0, 0), 1, 1, facecolor='red',   hatch='///', alpha=0.45, label='Overconfident'),
        Rectangle((0, 0), 1, 1, facecolor='green', hatch='///', alpha=0.45, label='Underconfident'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=fontsize - 1)

    plt.tight_layout()
    if show_plot:
        plt.show()

    return {'ECE': ece, 'gap': gap_valid, 'fig': fig, 'ax': ax}