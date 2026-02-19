import shutil
import subprocess
import traceback
from multiprocessing import Pool

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LogNorm
from scipy.ndimage import gaussian_filter
import os
import matplotlib.ticker as ticker
import numpy as np
from matplotlib.gridspec import GridSpec

def weighted_quantile(values, quantiles, weights):
    sorter = np.argsort(values)
    v, w = values[sorter], weights[sorter]
    cumw = np.cumsum(w)
    cumw /= cumw[-1]
    return np.interp(quantiles, cumw, v)

def format_with_error(med, lo, hi):
    err = max(hi - med, med - lo)
    if err == 0 or np.isnan(err):
        return f"{med:.2f}", f"+{(hi-med):.2f}/-{(med-lo):.2f}"
    sig = 2
    decimals = max(-int(np.floor(np.log10(err))) + (sig - 1), 0)
    fmt = f"{{:.{decimals}f}}"
    med_str = fmt.format(med)
    err_hi = fmt.format(hi - med)
    err_lo = fmt.format(med - lo)
    return med_str, f"+{err_hi}/-{err_lo}"

def sync_y_to_x(ax_src, ax_dst):
    def on_ylim_change(event_ax):
        if getattr(ax_dst, '_updating', False):
            return
        ax_dst._updating = True
        try:
            ylim = ax_src.get_ylim()
            ax_dst.set_xlim(ylim)
            ax_dst.figure.canvas.draw_idle()
        finally:
            ax_dst._updating = False
    return on_ylim_change

def sync_x_to_y(ax_src, ax_dst):
    def on_xlim_change(event_ax):
        if getattr(ax_dst, '_updating', False):
            return
        ax_dst._updating = True
        try:
            xlim = ax_src.get_xlim()
            ax_dst.set_ylim(xlim)
            ax_dst.figure.canvas.draw_idle()
        finally:
            ax_dst._updating = False
    return on_xlim_change

def get_quantile_bounds(values, weights, quantile=0.995, margin=0.05):
    """Get bounds covering the specified quantile of data with margin."""
    q_low = (1 - quantile) / 2
    q_high = 1 - q_low
    low_val, high_val = weighted_quantile(values, [q_low, q_high], weights)
    range_size = high_val - low_val
    margin_size = range_size * margin
    return low_val - margin_size, high_val + margin_size

def param_to_name(p):
    """Convert parameter names to formatted labels."""
    param_namedict = {
        "period": r"$P$ [d]",
        "amplitude": r"$K$ [km s$^{-1}$]",
        "offset": r"$\gamma$ [km s$^{-1}$]",
        "phase": r"$t_0$ [phase]",
        "eccentricity": r"$e$",
        "omega": r"$\omega$ [rad]",
    }
    return param_namedict.get(p, p.capitalize())

def candidate_plot(gaia_id, eccentric=False, smoothing=0, limits="", base_dir=None, output_dir="plots"):
    # --- hard-coded limits ---
    amp_lim    = 500.0
    offset_lim = 500.0
    phase_min, phase_max = -0.5, 0.5
    ecc_min, ecc_max = 0.0, 1.0
    omega_min, omega_max = 0.0, 2*np.pi

    # Parse limits string
    try:
        if limits != "":
            limits_dict = {}
            for l in limits.split(";"):
                parts = l.strip().split()
                if len(parts) >= 3:
                    param = parts[0].strip()
                    vmin = float(parts[1].strip())
                    vmax = float(parts[2].strip())
                    limits_dict[param] = (vmin, vmax)
            limits = limits_dict
        else:
            limits = {}
    except:
        limits = {}

    # --- load histograms from specified directory ---
    if base_dir is None:
        base = os.path.join(
            "/home/fabian",
            "Projects/subdwarf-rv-simulation/build",
            "out",
            str(gaia_id)
        )
    else:
        base = os.path.join(base_dir, str(gaia_id))

    H = {}
    def load2d(name):
        filepath = os.path.join(base, f"{name}.csv")
        if os.path.exists(filepath):
            return np.loadtxt(filepath, delimiter=',')
        return None
    
    # Load base combinations
    base_combinations = [
        ('period','amplitude'),
        ('period','offset'),
        ('period','phase'),
        ('amplitude','offset'),
        ('amplitude','phase'),
        ('offset','phase'),
    ]
    
    # Add eccentric combinations if flag is set
    if eccentric:
        eccentric_combinations = [
            ('period','eccentricity'),
            ('period','omega'),
            ('amplitude','eccentricity'),
            ('amplitude','omega'),
            ('offset','eccentricity'),
            ('offset','omega'),
            ('phase','eccentricity'),
            ('phase','omega'),
            ('eccentricity','omega'),
        ]
        all_combinations = base_combinations + eccentric_combinations
    else:
        all_combinations = base_combinations
    
    for a,b in all_combinations:
        data = load2d(f"{a}_vs_{b}")
        if data is not None:
            H[(a,b)] = data

    # --- get bin counts from one histogram ---
    Nx, Ny = H['period','amplitude'].shape

    # --- rebuild edges exactly as in sampler ---
    pgram = np.loadtxt(os.path.join(base, "pgram.csv"), delimiter=',')
    pgram_x, _ = pgram.T
    edges = {
        'period':    np.geomspace(pgram_x.min(), pgram_x.max(), Nx+1),
        'amplitude': np.linspace(0,           amp_lim,    Ny+1),
        'offset':    np.linspace(-offset_lim, offset_lim, Ny+1),
        'phase':     np.linspace(phase_min,   phase_max,  Ny+1),
    }
    
    if eccentric:
        edges['eccentricity'] = np.linspace(ecc_min, ecc_max, Ny+1)
        edges['omega'] = np.linspace(omega_min, omega_max, Ny+1)

    # --- compute 1D marginals correctly ---
    names = ['period','amplitude','offset','phase']
    if eccentric:
        names.extend(['eccentricity', 'omega'])
    
    marg = {}
    for α in names:
        w = np.zeros(len(edges[α]) - 1)
        for β in names:
            if α == β:
                continue
            key = (α,β) if (α,β) in H else (β,α)
            if key not in H:
                continue
            arr = H[key]
            
            if key[0] == α:
                w += arr.sum(axis=0)
            else:
                w += arr.sum(axis=1)
        marg[α] = w

    # --- Compute bounds for zooming ---
    bounds = {}
    for α in names:
        if α in limits:
            bounds[α] = limits[α]
        else:
            centers = 0.5*(edges[α][1:] + edges[α][:-1])
            wts = marg[α] / marg[α].sum()
            lo, hi = get_quantile_bounds(centers, wts, quantile=0.995, margin=0.05)
            bounds[α] = (lo, hi)

    # --- print medians ± errors ---
    print(f"\nParameter estimates for GAIA ID {gaia_id}:")
    print("-" * 60)
    for α in names:
        centers = 0.5*(edges[α][1:] + edges[α][:-1])
        wts     = marg[α] / marg[α].sum()
        med, lo, hi = weighted_quantile(centers, [0.5,0.16,0.84], wts)
        s, err = format_with_error(med, lo, hi)
        print(f"{param_to_name(α):>15s} = {s:>10s} ({err})")

    # --- prepare corner grid ---
    n = len(names)
    fig = plt.figure(figsize=(2.5 * n, 1.75 * n))
    gs = GridSpec(n, n, left=0.1, right=0.95, bottom=0.1, top=0.95,
                  wspace=0.05, hspace=0.05)
    
    axes = {}
    
    # Create all axes
    for i, α in enumerate(names):
        for j, β in enumerate(names):
            ax = fig.add_subplot(gs[i,j])
            axes[(i,j)] = ax

    # --- draw all panels ---
    for i, α in enumerate(names):
        for j, β in enumerate(names):
            ax = axes[(i,j)]
            if i > j:
                # off-diag: 2D histogram
                key = (α,β) if (α,β) in H else (β,α)
                if key not in H:
                    ax.axis('off')
                    continue
                    
                arr = H[key]
                
                if key == (α,β):
                    arr = arr.T
                else:
                    pass
                
                # Apply smoothing if requested
                if smoothing > 0:
                    arr = gaussian_filter(arr, smoothing)
                
                # mask zeros & nans, dynamic vmin/vmax
                mask = (arr <= 0) | np.isnan(arr)
                arrm = np.ma.masked_array(arr, mask=mask)

                if arrm.count() > 0:
                    vmin = float(arrm.compressed().min())
                    vmax = float(arrm.compressed().max())
                else:
                    vmin, vmax = 1e-2, 1.0

                ax.pcolormesh(
                    edges[β], edges[α], arrm,
                    cmap='Blues',
                    vmin=vmin,
                    vmax=vmax,
                    shading='auto',
                    rasterized=True
                )

                # only period is log
                if β == 'period':
                    ax.set_xscale('log')
                if α == 'period':
                    ax.set_yscale('log')
                
                # Handle tick labels
                if i < n-1:
                    ax.tick_params(axis='x', labelbottom=False)
                    # Also set locator to reduce ticks even when labels are hidden
                    if β == 'period':
                        ax.xaxis.set_major_locator(ticker.LogLocator(base=10, numticks=3))
                    else:
                        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=3))
                else:
                    ax.set_xlabel(param_to_name(β))
                    if β == 'period':
                        # Use fewer ticks for log scale to prevent overlap
                        ax.xaxis.set_major_locator(ticker.LogLocator(base=10, numticks=3))
                    else:
                        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=3))
                    for lbl in ax.get_xticklabels():
                        lbl.set_rotation(45)
                
                if j > 0:
                    ax.tick_params(axis='y', labelleft=False)
                    # Also set locator to reduce ticks even when labels are hidden
                    if α == 'period':
                        ax.yaxis.set_major_locator(ticker.LogLocator(base=10, numticks=3))
                    else:
                        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=3))
                else:
                    ax.set_ylabel(param_to_name(α))
                    if α == 'period':
                        # Use fewer ticks for log scale to prevent overlap
                        ax.yaxis.set_major_locator(ticker.LogLocator(base=10, numticks=3))
                    else:
                        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=3))
                        
            elif i == j:
                # diag: independent 1D marginal
                centers = 0.5*(edges[α][1:] + edges[α][:-1])
                
                # Apply smoothing to the histogram values
                smoothed_weights = gaussian_filter(marg[α], sigma=smoothing) if smoothing > 0 else marg[α]
                
                ax.hist(
                    centers, bins=edges[α], weights=smoothed_weights,
                    histtype='stepfilled',
                    edgecolor='black', facecolor='lightblue', alpha=0.6
                )
                if α == 'period':
                    ax.set_xscale('log')
                
                # Handle tick labels for diagonal plots
                if i < n-1:
                    ax.tick_params(axis='x', labelbottom=False)
                    # Set locator even when labels are hidden to reduce tick density
                    if α == 'period':
                        ax.xaxis.set_major_locator(ticker.LogLocator(base=10, numticks=3))
                    else:
                        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=3))
                else:
                    ax.set_xlabel(param_to_name(α))
                    if α == 'period':
                        # Use fewer ticks for log scale to prevent overlap
                        ax.xaxis.set_major_locator(ticker.LogLocator(base=10, numticks=3))
                        for lbl in ax.get_xticklabels():
                            lbl.set_rotation(45)
                    else:
                        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=3))
                        for lbl in ax.get_xticklabels():
                            lbl.set_rotation(45)
                
                ax.tick_params(axis='y', labelleft=False)
                        
            else:
                # Upper triangle - turn off
                ax.axis('off')

    # Set up axis synchronization using callbacks
    for k, param in enumerate(names):
        # Get all axes where this parameter appears on x-axis
        x_axes = [axes[(i, k)] for i in range(k, n)]
        
        # Get all axes where this parameter appears on y-axis
        y_axes = [axes[(k, j)] for j in range(k)]
        
        # Share x-axes among themselves
        if len(x_axes) > 1:
            master_x = x_axes[0]
            for ax in x_axes[1:]:
                ax.sharex(master_x)
        
        # Share y-axes among themselves
        if len(y_axes) > 1:
            master_y = y_axes[0]
            for ax in y_axes[1:]:
                ax.sharey(master_y)
        
        # Connect x and y representations bidirectionally
        for ax_x in x_axes:
            for ax_y in y_axes:
                ax_x.callbacks.connect("xlim_changed", sync_x_to_y(ax_x, ax_y))
                ax_y.callbacks.connect("ylim_changed", sync_y_to_x(ax_y, ax_x))

    # Apply initial zoom to bounds
    for i, param in enumerate(names):
        diag_ax = axes[(i,i)]
        diag_ax.set_xlim(bounds[param])

    # ------------------------------------------------------------------
    # FINAL TIDY-UP  ––  run AFTER sharex()/sharey() has been called
    # ------------------------------------------------------------------
    for (i, j), ax in axes.items():

        # --------------------------------------------------  x-axis
        if i == n - 1:                     # show labels only in bottom row
            ax.tick_params(axis='x', labelbottom=True)

            # fewer ticks – works for both linear and log axes
            if (names[j] == 'period'):
                ax.xaxis.set_major_locator(ticker.LogLocator(base=10, numticks=4))
            else:
                ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4))

            # rotate for readability
            for lab in ax.get_xticklabels():
                lab.set_rotation(45)
                lab.set_ha('right')
        else:                              # hide everywhere else
            ax.tick_params(axis='x', labelbottom=False)

        # --------------------------------------------------  y-axis
        if j == 0:                         # only first column keeps labels
            ax.tick_params(axis='y', labelleft=True)

            if (names[i] == 'period'):
                ax.yaxis.set_major_locator(ticker.LogLocator(base=10, numticks=4))
            else:
                ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
        else:
            ax.tick_params(axis='y', labelleft=False)
    plt.tight_layout(pad=0.4)
    
    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"rv_cornerplot_{gaia_id}.pdf")
    plt.savefig(output_path, dpi=300)
    plt.show()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create corner plot for RV fitting results")
    parser.add_argument("gaia_id", type=int, help="GAIA source ID")
    parser.add_argument("--eccentric", action="store_true", 
                        help="Include eccentricity and omega parameters in the corner plot")
    parser.add_argument("--smoothing", type=float, default=0,
                        help="Smoothing radius for Gaussian blur of the histograms (default: 0)")
    parser.add_argument("--limits", type=str, default="",
                        help="Max and min bounds for specific parameters (example: 'period 1 100; amplitude 0 50')")
    parser.add_argument("--base-dir", type=str, default=None,
                        help="Base directory containing the histogram data (default: /home/fabian/Projects/subdwarf-rv-simulation/build/out)")
    parser.add_argument("--output-dir", type=str, default="plots",
                        help="Output directory for plots (default: plots)")
    args = parser.parse_args()

    candidate_plot(args.gaia_id, eccentric=args.eccentric, smoothing=args.smoothing, 
                   limits=args.limits, base_dir=args.base_dir, output_dir=args.output_dir)  