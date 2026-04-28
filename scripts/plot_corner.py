#!/usr/bin/env python3
"""
Interactive corner plot from MCMC chain.

- Adaptive Freedman–Diaconis binning per cluster (skips empty gaps)
- Zoom on any marginal → filters chain for all other panels
- Zoom out / Home button → resets to full chain
- Phase parameter is converted to T₀ [BJD] for display
- Saves two PDFs: full view + zoomed to dominant period peak
"""

import argparse
import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from matplotlib.gridspec import GridSpec
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks

# ====================================================================
#  I/O
# ====================================================================

def load_chain(base_dir, gaia_id):
    base = os.path.join(base_dir, str(gaia_id))
    meta_path = os.path.join(base, "chain_meta.txt")
    bin_path  = os.path.join(base, "chain.bin")
    if not os.path.exists(meta_path) or not os.path.exists(bin_path):
        raise FileNotFoundError(
            f"Chain files not found in {base}.  "
            "Re-run the MCMC with the updated C++ code.")
    with open(meta_path) as f:
        n_params    = int(f.readline().strip())
        param_names = f.readline().strip().split()
    raw = np.fromfile(bin_path, dtype=np.float64)
    n_samples = len(raw) // n_params
    if n_samples == 0:
        raise ValueError("Chain file is empty")
    chain = raw[: n_samples * n_params].reshape(n_samples, n_params)
    print(f"Loaded chain: {n_samples} samples × {n_params} params from {bin_path}")
    return chain, list(param_names)


def load_t_ref(base_dir, gaia_id):
    p = os.path.join(base_dir, str(gaia_id), "t_ref.txt")
    if os.path.exists(p):
        return float(open(p).read().strip())
    print("  Warning: t_ref.txt not found — T₀ will be relative to t=0")
    return 0.0


# ====================================================================
#  Adaptive edge builder
# ====================================================================

def build_adaptive_edges(values, name, resolution=3, coarsen=1,
                         min_bins_per_cluster=30, max_total_bins=25000,
                         obs_times=None):
    """
    Build histogram edges that are *fine* inside density peaks and
    *coarse* (single bin) across empty gaps.

    If *obs_times* is given and *name* == 'period', the scout-histogram
    resolution is derived from the periodogram sampling algorithm so
    that fine alias peaks are correctly separated.
    """
    if len(values) < 2:
        lo, hi = float(values.min()) - 1, float(values.max()) + 1
        return np.linspace(lo, hi, 20)

    is_log = (name == "period")
    work = np.log10(np.clip(values, 1e-10, None)) if is_log else values.copy()
    sv = np.sort(work)
    n  = len(sv)
    total_range = sv[-1] - sv[0]

    if total_range == 0:
        e = np.linspace(sv[0] - 1, sv[0] + 1, 20)
        return 10 ** e if is_log else e

    # --- period resolution from observations (if available) ----------------
    period_res = None
    if is_log and obs_times is not None and len(obs_times) >= 2:
        try:
            period_res = compute_period_resolution(
                obs_times, min_p=10 ** sv[0], max_p=10 ** sv[-1],
                sample_factor=3.0)
        except Exception:
            period_res = None

    # ------------------------------------------------------------------
    # Step 1 — scout histogram (fine enough to see inter-alias valleys)
    # ------------------------------------------------------------------
    if is_log:
        if period_res is not None:
            scout_bw = period_res["alias_log_at_min"] / 3.0
            scout_nb = max(500, int(total_range / scout_bw))
            scout_nb = min(scout_nb, 200_000)
        else:
            scout_bw = 0.0002                               # in log₁₀
            scout_nb = max(500, int(total_range / scout_bw))
            scout_nb = min(scout_nb, 60000)
    else:
        q75, q25 = np.percentile(work, [75, 25])
        iqr = q75 - q25
        if iqr <= 0:
            iqr = total_range
        fd_w = 2.0 * iqr * n ** (-1.0 / 3.0)
        scout_nb = max(100, min(5000, int(total_range / fd_w * 3)))

    scout_hist, scout_edges = np.histogram(work, bins=scout_nb)
    scout_bw_actual = scout_edges[1] - scout_edges[0]

    # light smoothing — just enough to suppress Poisson noise
    if is_log:
        if period_res is not None:
            sigma = np.clip(
                period_res["alias_log_at_min"] / (3.0 * scout_bw_actual),
                0.5, 5.0)
        else:
            sigma = np.clip(0.0004 / scout_bw_actual, 0.5, 5.0)
    else:
        sigma = 1.5
    scout_s = gaussian_filter(scout_hist.astype(float), sigma=sigma)

    # ------------------------------------------------------------------
    # Step 2 — find peaks and troughs in the scout histogram
    # ------------------------------------------------------------------
    if is_log:
        if period_res is not None:
            min_dist = max(1, int(
                period_res["alias_log_at_min"] / (2.0 * scout_bw_actual)))
        else:
            min_dist = max(1, int(0.0003 / scout_bw_actual))
    else:
        min_dist = 3
    pks, _ = find_peaks(scout_s,
                        height=scout_s.max() * 0.008,
                        prominence=scout_s.max() * 0.004,
                        distance=min_dist)
    if len(pks) == 0:
        pks = np.array([int(np.argmax(scout_s))])

    pks_sorted = np.sort(pks)

    # troughs = minima between consecutive peaks (+ histogram boundaries)
    troughs = [0]
    for k in range(len(pks_sorted) - 1):
        seg = scout_s[pks_sorted[k]:pks_sorted[k + 1] + 1]
        troughs.append(pks_sorted[k] + int(np.argmin(seg)))
    troughs.append(len(scout_s))

    # ------------------------------------------------------------------
    # Step 3 — fine edges inside each peak region, single bin for gaps
    # ------------------------------------------------------------------
    segments   = []
    total_bins = 0

    for k in range(len(troughs) - 1):
        lo_idx = troughs[k]
        hi_idx = troughs[k + 1]
        region_lo = scout_edges[lo_idx]
        region_hi = scout_edges[min(hi_idx, len(scout_edges) - 1)]

        if region_hi <= region_lo:
            continue

        # check whether this region contains significant density
        seg_max = scout_s[lo_idx:hi_idx].max() if hi_idx > lo_idx else 0
        if seg_max < scout_s.max() * 0.008:
            # sparse gap → single bin
            segments.append(np.array([region_lo, region_hi]))
            total_bins += 1
            continue

        # dense cluster → Freedman–Diaconis binning
        cl_mask = (work >= region_lo) & (work <= region_hi)
        cl_vals = work[cl_mask]
        nc = len(cl_vals)

        if nc < 3:
            segments.append(np.array([region_lo, region_hi]))
            total_bins += 1
            continue

        q75, q25 = np.percentile(cl_vals, [75, 25])
        iqr = q75 - q25
        if iqr <= 0:
            iqr = cl_vals.max() - cl_vals.min()
        if iqr <= 0:
            iqr = abs(cl_vals.mean()) * 0.01 + 1e-10

        fd_w = 2.0 * iqr * nc ** (-1.0 / 3.0)
        bw   = fd_w / max(resolution, 1) * max(coarsen, 1)

        margin = bw * 0.5
        lo = region_lo - margin
        hi = region_hi + margin
        nb = max(int(np.ceil((hi - lo) / bw)), min_bins_per_cluster)
        remaining = max_total_bins - total_bins
        if nb > remaining:
            nb = max(remaining, min_bins_per_cluster)
        segments.append(np.linspace(lo, hi, nb + 1))
        total_bins += nb

    if not segments:
        e = np.linspace(sv[0] - 1, sv[-1] + 1, 50)
        return 10 ** e if is_log else e

    combined = np.unique(np.concatenate(segments))
    return 10 ** combined if is_log else combined
# ====================================================================
#  Peak finding
# ====================================================================

def load_obs_times(base_dir, gaia_id, rv_dir=None):
    """
    Load observation times (BJD) for period-resolution computation.
    Checks chain dir for obs_times.txt, then falls back to RV CSV.
    """
    base = os.path.join(base_dir, str(gaia_id))
    obs_path = os.path.join(base, "obs_times.txt")
    if os.path.exists(obs_path):
        try:
            t = np.atleast_1d(np.loadtxt(obs_path))
            if len(t) >= 2:
                return t
        except Exception:
            pass

    if rv_dir is not None:
        rv_file = os.path.join(rv_dir, str(gaia_id), "RV_variation.csv")
    else:
        rv_file = os.path.expanduser(
            f"~/Projects/RVVD_refit_2025/output/{gaia_id}/RV_variation.csv")

    if os.path.exists(rv_file):
        try:
            with open(rv_file) as f:
                hdr = [h.strip().upper() for h in f.readline().split(",")]
            if "BJD" in hdr:
                data = np.genfromtxt(rv_file, delimiter=",",
                                     skip_header=1,
                                     usecols=(hdr.index("BJD"),))
                data = data[np.isfinite(data)]
                if len(data) >= 2:
                    return data
        except Exception as e:
            print(f"  Warning: could not load obs times from {rv_file}: {e}")
    return None

def find_period_peaks(chain, col, obs_times=None, sample_factor=5.0,
                      n_bins=80000):
    """Return list of dicts sorted by prominence (most prominent first).

    Uses troughs between adjacent peaks as boundaries so that masks
    are non-overlapping and each sample is assigned to exactly one peak.

    If *obs_times* is supplied, the histogram resolution is derived from
    the periodogram sampling algorithm (bin width = alias_spacing /
    sample_factor in log₁₀ P) so that even the finest alias peaks are
    resolved.  Smoothing and minimum peak distance also scale with the
    alias spacing.
    """
    periods = chain[:, col["period"]]
    log_p   = np.log10(periods)
    log_range = log_p.max() - log_p.min()

    # ---- determine histogram parameters ----------------------------------
    if obs_times is not None and len(obs_times) >= 2:
        res = compute_period_resolution(
            obs_times,
            min_p=float(periods.min()),
            max_p=float(periods.max()),
            sample_factor=sample_factor,
        )
        target_bw = res["log_bw_at_min"]
        alias_log = res["alias_log_at_min"]

        nb = int(np.ceil(log_range / target_bw)) if target_bw > 0 else n_bins
        nb = int(np.clip(nb, 5000, 1_000_000))

        edges = np.linspace(log_p.min() - 1e-10,
                            log_p.max() + 1e-10, nb + 1)
        bw = edges[1] - edges[0]

        # smoothing ≈ 1/3 alias spacing (in bins) — resolves adjacent aliases
        sigma_bins = np.clip(alias_log / (3.0 * bw), 0.5, 8.0)
        # minimum peak distance ≈ 1/2 alias spacing (in bins)
        min_dist = max(1, int(alias_log / (2.0 * bw)))

        print(f"  [Peak detection] observation-based: {nb} bins, "
              f"bw={bw:.2e} log₁₀(d), σ={sigma_bins:.1f} bins, "
              f"min_dist={min_dist} bins, "
              f"alias spacing={alias_log:.2e} log₁₀(d), "
              f"baseline={res['x_ptp']:.1f} d")
    else:
        # fallback heuristic (original behaviour)
        target_bw = 0.00005
        nb = int(np.clip(log_range / target_bw, 5000, n_bins))
        nb = max(nb, min(n_bins, int(np.sqrt(len(chain)) * 15)))

        edges = np.linspace(log_p.min() - 1e-10,
                            log_p.max() + 1e-10, nb + 1)
        bw = edges[1] - edges[0]
        sigma_bins = np.clip(0.00025 / bw, 0.5, 4.0)
        min_dist   = max(1, int(0.00015 / bw))

    # ---- build histogram --------------------------------------------------
    indices = np.searchsorted(edges, log_p, side="right") - 1
    indices = np.clip(indices, 0, nb - 1)
    hist    = np.bincount(indices, minlength=nb)[:nb]
    centers = 0.5 * (edges[1:] + edges[:-1])

    hist_s = gaussian_filter(hist.astype(float), sigma=sigma_bins)

    # ---- find peaks -------------------------------------------------------
    pks, props = find_peaks(hist_s,
                            height=hist_s.max() * 0.01,
                            prominence=hist_s.max() * 0.005,
                            distance=min_dist)
    if len(pks) == 0:
        pk = int(np.argmax(hist_s))
        pks   = np.array([pk])
        props = {"prominences":  np.array([hist_s[pk]]),
                 "peak_heights": np.array([hist_s[pk]])}

    # ---- non-overlapping masks via troughs --------------------------------
    pos_order = np.argsort(pks)
    s_pks = pks[pos_order]
    n_p   = len(s_pks)

    trough_bins = []
    for k in range(n_p - 1):
        seg = hist_s[s_pks[k]:s_pks[k + 1] + 1]
        trough_bins.append(s_pks[k] + int(np.argmin(seg)))

    masks_sorted = []
    for k in range(n_p):
        lo_bin = trough_bins[k - 1] if k > 0      else 0
        hi_bin = trough_bins[k]     if k < n_p - 1 else len(hist_s)
        lp_lo  = edges[lo_bin]
        lp_hi  = edges[hi_bin]
        masks_sorted.append((log_p >= lp_lo) & (log_p < lp_hi))

    # ---- sort by prominence -----------------------------------------------
    order = np.argsort(props["prominences"])[::-1]

    peaks = []
    for rank, oi in enumerate(order):
        sorted_pos = int(np.where(pos_order == oi)[0][0])
        mask = masks_sorted[sorted_pos]
        pk   = pks[oi]
        peaks.append(dict(
            rank=rank + 1,
            period=10 ** centers[pk],
            prom=props["prominences"][oi],
            n=int(mask.sum()),
            mask=mask,
            log_p_range=(
                edges[trough_bins[sorted_pos - 1]]
                    if sorted_pos > 0      else edges[0],
                edges[trough_bins[sorted_pos]]
                    if sorted_pos < n_p - 1 else edges[-1]),
        ))
    return peaks

# ====================================================================
#  Histogram helpers
# ====================================================================

MAX_BINS_1D = 1000
MAX_BINS_2D = 500


def _make_edges(values, name, view_lo, view_hi, max_bins):
    """
    Uniform edges in display space within [view_lo, view_hi].
    Bin count = min(Freedman–Diaconis optimal for data in view, max_bins).
    For 'period', edges are log-spaced (returned in linear space for
    matplotlib log-scale axes).
    """
    is_log = (name == "period")
    if is_log:
        d_lo = np.log10(max(view_lo, 1e-10))
        d_hi = np.log10(max(view_hi, 1e-10))
        work = np.log10(np.clip(values, 1e-10, None))
    else:
        d_lo, d_hi = float(view_lo), float(view_hi)
        work = np.asarray(values, dtype=float)

    view_range = d_hi - d_lo
    if view_range <= 0:
        nb = min(20, max_bins)
    else:
        in_view = work[(work >= d_lo) & (work <= d_hi)]
        if len(in_view) < 2:
            nb = min(20, max_bins)
        else:
            q75, q25 = np.percentile(in_view, [75, 25])
            iqr = q75 - q25
            if iqr <= 0:
                iqr = float(in_view.max() - in_view.min())
            if iqr <= 0:
                iqr = 1e-10
            fd_w = 2.0 * iqr * len(in_view) ** (-1.0 / 3.0)
            nb = (int(np.ceil(view_range / fd_w)) if fd_w > 0
                  else max_bins)
            nb = int(np.clip(nb, 10, max_bins))

    disp = np.linspace(d_lo, d_hi, nb + 1)
    return 10 ** disp if is_log else disp


def _hist1d(values, name, edges):
    """1-D histogram; period is binned in log space."""
    if name == "period":
        h, _ = np.histogram(
            np.log10(np.clip(values, 1e-10, None)),
            bins=np.log10(np.clip(edges, 1e-10, None)))
    else:
        h, _ = np.histogram(values, bins=edges)
    return h.astype(float)


def _hist2d(vx, nx, ex, vy, ny, ey):
    """2-D histogram; period axes are binned in log space."""
    hx = (np.log10(np.clip(vx, 1e-10, None)) if nx == "period"
          else np.asarray(vx, dtype=float))
    bx = (np.log10(np.clip(ex, 1e-10, None)) if nx == "period"
          else np.asarray(ex, dtype=float))
    hy = (np.log10(np.clip(vy, 1e-10, None)) if ny == "period"
          else np.asarray(vy, dtype=float))
    by = (np.log10(np.clip(ey, 1e-10, None)) if ny == "period"
          else np.asarray(ey, dtype=float))
    hh, _, _ = np.histogram2d(hx, hy, bins=[bx, by])
    return hh.astype(float)


def _step_xy(edges, h):
    """Stepped-histogram line coordinates."""
    return np.repeat(edges, 2)[1:-1], np.repeat(h, 2)

# ====================================================================
#  Formatting helpers
# ====================================================================

PARAM_LABELS = {
    "period":       r"$P$ [d]",
    "amplitude":    r"$K$ [km s$^{-1}$]",
    "offset":       r"$\gamma$ [km s$^{-1}$]",
    "phase":        r"$t_0$ [phase]",
    "t0":           r"$T_0$ [BJD]",
    "eccentricity": r"$e$",
    "omega":        r"$\omega$ [deg]",
}

def param_to_name(p):
    return PARAM_LABELS.get(p, p.capitalize())

def format_with_error(med, lo, hi):
    err = max(hi - med, med - lo)
    if err == 0 or np.isnan(err):
        return f"{med:.2f}", f"+{hi - med:.2f}/-{med - lo:.2f}"
    dec = max(-int(np.floor(np.log10(err))) + 1, 0)
    f = f"{{:.{dec}f}}"
    return f.format(med), f"+{f.format(hi - med)}/-{f.format(med - lo)}"


def compute_period_resolution(obs_times, min_p=None, max_p=None,
                               sample_factor=1.0):
    """
    Compute optimal period resolution from observation timestamps,
    mirroring the periodogram sampling algorithm.

    At period P with baseline T, adjacent aliases are separated by
    R_p ≈ P²/T in period, or ≈ P/(T·ln10) in log₁₀(P).
    The frequency step is df ≈ 1/(T · sample_factor).

    Returns dict with df, n_points, min_p, max_p, x_ptp, R_p,
    alias_log_at_min, log_bw_at_min.
    """
    t = np.sort(np.asarray(obs_times, dtype=float))
    x_ptp = t[-1] - t[0]
    if x_ptp <= 0:
        raise ValueError("Observation time baseline is zero or negative.")

    if min_p is None or min_p <= 0:
        diffs = np.diff(t)
        diffs = diffs[diffs > 0]
        min_p = float(np.min(diffs)) * 2 if len(diffs) > 0 else 0.01
    if max_p is None or max_p <= 0:
        max_p = x_ptp / 2.0

    n = np.ceil(x_ptp / min_p)
    if n <= 1:
        n = 2.0
    R_p = x_ptp / (n - 1.0) - x_ptp / n

    df = (1.0 / min_p - 1.0 / (min_p + R_p)) / sample_factor
    if not np.isfinite(df) or df <= 0:
        n_points = 100000
        df = (1.0 / min_p - 1.0 / max_p) / n_points
    else:
        n_points = int(np.ceil((1.0 / min_p - 1.0 / max_p) / df))

    ln10 = np.log(10.0)
    alias_log = R_p / (min_p * ln10)
    log_bw = alias_log / max(sample_factor, 1.0)

    return dict(df=df, n_points=n_points,
                min_p=min_p, max_p=max_p, x_ptp=x_ptp, R_p=R_p,
                alias_log_at_min=alias_log, log_bw_at_min=log_bw)

# ====================================================================
#  Static corner-plot builder  (no interactive callbacks)
# ====================================================================

def build_corner(chain, param_names, names, col, smoothing=0,
                 bounds=None, title=None):
    n = len(names)

    # view bounds — log-space margin for period
    view = {}
    for nm in names:
        v = chain[:, col[nm]]
        if bounds and nm in bounds:
            view[nm] = bounds[nm]
        elif nm == "period":
            lo, hi = float(v.min()), float(v.max())
            log_lo = np.log10(max(lo, 1e-10))
            log_hi = np.log10(max(hi, 1e-10))
            log_margin = (log_hi - log_lo) * 0.03 if log_hi > log_lo else 0.1
            view[nm] = (10 ** (log_lo - log_margin),
                        10 ** (log_hi + log_margin))
        else:
            lo, hi = float(v.min()), float(v.max())
            margin = (hi - lo) * 0.03 if hi > lo else 0.1
            view[nm] = (lo - margin, hi + margin)

    fig = plt.figure(figsize=(2.5 * n, 2.0 * n))
    gs  = GridSpec(n, n, left=0.09, right=0.97, bottom=0.09, top=0.92,
                   wspace=0.04, hspace=0.04)
    if title:
        fig.suptitle(title, fontsize=10)

    cmap = plt.cm.Blues.copy()
    cmap.set_bad("white", alpha=0)

    axes = {}
    for i in range(n):
        for j in range(n):
            ax = fig.add_subplot(gs[i, j])
            axes[(i, j)] = ax
            if j > i:
                ax.axis("off")
                continue
            ax.set_autoscalex_on(False)
            ax.set_autoscaley_on(False)

            nm_i, nm_j = names[i], names[j]

            if i == j:
                lo, hi = view[nm_i]
                edges = _make_edges(chain[:, col[nm_i]], nm_i,
                                    lo, hi, MAX_BINS_1D)
                h = _hist1d(chain[:, col[nm_i]], nm_i, edges)
                if smoothing > 0:
                    h = gaussian_filter(h, sigma=smoothing)
                sx, sy = _step_xy(edges, h)
                ax.fill_between(sx, sy, step="mid",
                                facecolor="lightblue", alpha=0.6)
                ax.plot(sx, sy, color="black", linewidth=0.5)
                if nm_i == "period":
                    ax.set_xscale("log")
                ax.set_xlim(lo, hi)
                ax.set_ylim(0, h.max() * 1.15 if h.max() > 0 else 1)
                ax.tick_params(axis="y", labelleft=False)
            else:
                lo_j, hi_j = view[nm_j]
                lo_i, hi_i = view[nm_i]
                ej = _make_edges(chain[:, col[nm_j]], nm_j,
                                 lo_j, hi_j, MAX_BINS_2D)
                ei = _make_edges(chain[:, col[nm_i]], nm_i,
                                 lo_i, hi_i, MAX_BINS_2D)
                hh = _hist2d(chain[:, col[nm_j]], nm_j, ej,
                             chain[:, col[nm_i]], nm_i, ei)
                if smoothing > 0:
                    hh = gaussian_filter(hh, sigma=smoothing)
                dat = hh.T.copy()
                dat[dat <= 0] = np.nan
                mesh = ax.pcolormesh(ej, ei, dat, cmap=cmap,
                                     shading="flat", rasterized=True)
                v = dat[~np.isnan(dat)]
                if len(v):
                    mesh.set_clim(v.min(), v.max())
                if nm_j == "period":
                    ax.set_xscale("log")
                if nm_i == "period":
                    ax.set_yscale("log")
                # --- FIX: explicit limits on off-diagonal panels ---
                ax.set_xlim(lo_j, hi_j)
                ax.set_ylim(lo_i, hi_i)

            # labels / ticks
            is_log_x = (nm_j == "period")
            is_log_y = (nm_i == "period") if i != j else False
            if i == n - 1:
                ax.set_xlabel(param_to_name(nm_j))
                ax.xaxis.set_major_locator(
                    ticker.LogLocator(base=10, numticks=5) if is_log_x
                    else ticker.MaxNLocator(nbins=4))
                for lb in ax.get_xticklabels():
                    lb.set_rotation(45); lb.set_ha("right")
            else:
                ax.tick_params(axis="x", labelbottom=False)
            if j == 0 and i != j:
                ax.set_ylabel(param_to_name(nm_i))
                ax.yaxis.set_major_locator(
                    ticker.LogLocator(base=10, numticks=5) if is_log_y
                    else ticker.MaxNLocator(nbins=4))
            else:
                ax.tick_params(axis="y", labelleft=False)

    # synchronise shared axes
    for k in range(n):
        x_axes = [axes[(i, k)] for i in range(k, n)]
        for ax in x_axes[1:]:
            ax.sharex(x_axes[0])
        y_axes = [axes[(k, j)] for j in range(k)]
        for ax in y_axes[1:]:
            ax.sharey(y_axes[0])

    # --- FIX: re-apply after sharex (sharing can widen limits) ---
    for k, nm in enumerate(names):
        axes[(k, k)].set_xlim(view[nm])

    fig.tight_layout(pad=0.4, rect=[0, 0, 1, 0.95] if title else None)
    return fig
# ====================================================================
#  Interactive corner (for the on-screen window)
# ====================================================================

class InteractiveCorner:
    """
    Interactive corner plot that recomputes histogram bins on every
    zoom / pan so that the resolution adapts to the current view.

    1-D panels: up to MAX_BINS_1D bins  (Freedman–Diaconis, capped)
    2-D panels: up to MAX_BINS_2D bins per axis
    """

    def __init__(self, chain, all_param_names, plot_names,
                 smoothing=0, limits=None):
        self.chain     = chain
        self.all_names = all_param_names
        self.names     = plot_names
        self.n         = len(plot_names)
        self.col       = {nm: all_param_names.index(nm) for nm in plot_names}
        self.smoothing = smoothing
        self.limits    = limits or {}

        self._guard          = False
        self.active_filters  = {}
        self.original_bounds = {}

        # ---- figure & axes ----
        self.fig = plt.figure(figsize=(2.5 * self.n, 2.0 * self.n))
        self.gs  = GridSpec(self.n, self.n,
                            left=0.09, right=0.97, bottom=0.09, top=0.94,
                            wspace=0.04, hspace=0.04)
        self.axes      = {}
        self.diag_fill = {}
        self.diag_line = {}
        self.meshes    = {}

        self._create_axes()
        self._setup_labels()

        # initial view = full data range
        for i, nm in enumerate(self.names):
            v = chain[:, self.col[nm]]
            if nm == "period":
                lo, hi = float(v.min()), float(v.max())
                log_lo = np.log10(max(lo, 1e-10))
                log_hi = np.log10(max(hi, 1e-10))
                log_margin = (log_hi - log_lo) * 0.03 if log_hi > log_lo else 0.1
                self.axes[(i, i)].set_xlim(10 ** (log_lo - log_margin),
                                           10 ** (log_hi + log_margin))
            else:
                lo, hi = float(v.min()), float(v.max())
                margin = (hi - lo) * 0.03 if hi > lo else 0.1
                self.axes[(i, i)].set_xlim(lo - margin, hi + margin)

        # disable autoscale so pcolormesh doesn't fight us
        for key, ax in self.axes.items():
            i, j = key
            if j <= i:
                ax.set_autoscalex_on(False)
                ax.set_autoscaley_on(False)

        self._draw_all()

        for i, nm in enumerate(self.names):
            self.original_bounds[nm] = self.axes[(i, i)].get_xlim()

        self._status = self.fig.text(
            0.98, 0.98, f"Chain: {len(chain)} samples",
            ha="right", va="top", fontsize=6.5,
            family="monospace", color="0.45",
            transform=self.fig.transFigure)

        self._connect_callbacks()

        # apply user limits (triggers callbacks → redraw)
        for nm, (lo, hi) in self.limits.items():
            if nm in self.names:
                idx = self.names.index(nm)
                self.axes[(idx, idx)].set_xlim(lo, hi)

    # ---- axes setup -------------------------------------------------------

    def _create_axes(self):
        for i in range(self.n):
            for j in range(self.n):
                ax = self.fig.add_subplot(self.gs[i, j])
                self.axes[(i, j)] = ax
                if j > i:
                    ax.axis("off")

    def _setup_labels(self):
        n = self.n
        for i, nm_i in enumerate(self.names):
            ax = self.axes[(i, i)]
            if nm_i == "period":
                ax.set_xscale("log")
            ax.tick_params(axis="y", labelleft=False)
            if i == n - 1:
                ax.set_xlabel(param_to_name(nm_i))
                ax.xaxis.set_major_locator(
                    ticker.LogLocator(base=10, numticks=5)
                    if nm_i == "period"
                    else ticker.MaxNLocator(nbins=4))
                for lb in ax.get_xticklabels():
                    lb.set_rotation(45); lb.set_ha("right")
            else:
                ax.tick_params(axis="x", labelbottom=False)

            for j in range(i):
                nm_j = self.names[j]
                ax2 = self.axes[(i, j)]
                if nm_j == "period":
                    ax2.set_xscale("log")
                if nm_i == "period":
                    ax2.set_yscale("log")
                if i == n - 1:
                    ax2.set_xlabel(param_to_name(nm_j))
                    ax2.xaxis.set_major_locator(
                        ticker.LogLocator(base=10, numticks=5)
                        if nm_j == "period"
                        else ticker.MaxNLocator(nbins=4))
                    for lb in ax2.get_xticklabels():
                        lb.set_rotation(45); lb.set_ha("right")
                else:
                    ax2.tick_params(axis="x", labelbottom=False)
                if j == 0:
                    ax2.set_ylabel(param_to_name(nm_i))
                    ax2.yaxis.set_major_locator(
                        ticker.LogLocator(base=10, numticks=5)
                        if nm_i == "period"
                        else ticker.MaxNLocator(nbins=4))
                else:
                    ax2.tick_params(axis="y", labelleft=False)

    # ---- filtered chain ---------------------------------------------------

    def _get_filtered_chain(self):
        if not self.active_filters:
            return self.chain
        mask = np.ones(len(self.chain), dtype=bool)
        for nm, (lo, hi) in self.active_filters.items():
            v = self.chain[:, self.col[nm]]
            mask &= (v >= lo) & (v <= hi)
        return self.chain[mask]

    # ---- draw / redraw ----------------------------------------------------

    def _draw_all(self):
        fc = self._get_filtered_chain()
        if len(fc) < 2:
            self.fig.canvas.draw_idle()
            return

        cmap = plt.cm.Blues.copy()
        cmap.set_bad("white", alpha=0)

        for i, nm_i in enumerate(self.names):
            ax = self.axes[(i, i)]
            lo_i, hi_i = ax.get_xlim()

            # ---- 1-D diagonal ----
            edges_1d = _make_edges(fc[:, self.col[nm_i]], nm_i,
                                   lo_i, hi_i, MAX_BINS_1D)
            h = _hist1d(fc[:, self.col[nm_i]], nm_i, edges_1d)
            if self.smoothing > 0:
                h = gaussian_filter(h, sigma=self.smoothing)

            if i in self.diag_fill:
                self.diag_fill[i].remove()
            if i in self.diag_line:
                self.diag_line[i].remove()

            sx, sy = _step_xy(edges_1d, h)
            self.diag_fill[i] = ax.fill_between(
                sx, sy, alpha=0.6, facecolor="lightblue", step="mid")
            self.diag_line[i], = ax.plot(
                sx, sy, color="black", linewidth=0.5)
            ax.set_ylim(0, h.max() * 1.15 if h.max() > 0 else 1)

            # ---- 2-D off-diagonal ----
            for j in range(i):
                nm_j = self.names[j]
                ax2  = self.axes[(i, j)]
                lo_j, hi_j = self.axes[(j, j)].get_xlim()

                ej = _make_edges(fc[:, self.col[nm_j]], nm_j,
                                 lo_j, hi_j, MAX_BINS_2D)
                ei = _make_edges(fc[:, self.col[nm_i]], nm_i,
                                 lo_i, hi_i, MAX_BINS_2D)

                hh = _hist2d(fc[:, self.col[nm_j]], nm_j, ej,
                             fc[:, self.col[nm_i]], nm_i, ei)
                if self.smoothing > 0:
                    hh = gaussian_filter(hh, sigma=self.smoothing)
                dat = hh.T.copy()
                dat[dat <= 0] = np.nan

                if (i, j) in self.meshes:
                    self.meshes[(i, j)].remove()

                mesh = ax2.pcolormesh(ej, ei, dat, cmap=cmap,
                                      shading="flat", rasterized=True)
                v = dat[~np.isnan(dat)]
                if len(v):
                    mesh.set_clim(v.min(), v.max())
                self.meshes[(i, j)] = mesh

                # keep off-diagonal limits in sync with diagonal
                ax2.set_xlim(lo_j, hi_j)
                ax2.set_ylim(lo_i, hi_i)

        self.fig.canvas.draw_idle()

    # ---- callbacks --------------------------------------------------------

    def _connect_callbacks(self):
        for i in range(self.n):
            for j in range(self.n):
                if j > i:
                    continue
                ax = self.axes[(i, j)]
                ax.callbacks.connect("xlim_changed",
                                     self._make_xlim_cb(j))
                if i != j:
                    ax.callbacks.connect("ylim_changed",
                                         self._make_ylim_cb(i))

    def _make_xlim_cb(self, pj):
        def cb(ax):
            if self._guard:
                return
            self._guard = True
            try:
                xl = ax.get_xlim()
                for i in range(pj, self.n):
                    self.axes[(i, pj)].set_xlim(xl)
                for jj in range(pj):
                    self.axes[(pj, jj)].set_ylim(xl)
                self._check_filter(pj, xl)
                self._draw_all()
            finally:
                self._guard = False
        return cb

    def _make_ylim_cb(self, pi):
        def cb(ax):
            if self._guard:
                return
            self._guard = True
            try:
                yl = ax.get_ylim()
                self.axes[(pi, pi)].set_xlim(yl)
                for jj in range(pi):
                    self.axes[(pi, jj)].set_ylim(yl)
                for ii in range(pi + 1, self.n):
                    self.axes[(ii, pi)].set_xlim(yl)
                self._check_filter(pi, yl)
                self._draw_all()
            finally:
                self._guard = False
        return cb

    def _check_filter(self, pidx, lim):
        nm   = self.names[pidx]
        orig = self.original_bounds[nm]
        if nm == "period":
            try:
                o_r = np.log10(orig[1]) - np.log10(orig[0])
                c_r = (np.log10(max(lim[1], 1e-10))
                       - np.log10(max(lim[0], 1e-10)))
            except (ValueError, FloatingPointError):
                return
        else:
            o_r = orig[1] - orig[0]
            c_r = lim[1] - lim[0]
        if o_r <= 0:
            return

        if c_r / o_r >= 0.85:
            if nm in self.active_filters:
                del self.active_filters[nm]
        else:
            self.active_filters[nm] = (float(lim[0]), float(lim[1]))

        nf = len(self._get_filtered_chain())
        if self.active_filters:
            parts = [f"Filtered: {nf}/{len(self.chain)}"]
            for nm2, (lo, hi) in self.active_filters.items():
                parts.append(f"  {nm2}: [{lo:.5g}, {hi:.5g}]")
            self._status.set_text("\n".join(parts))
        else:
            self._status.set_text(f"Chain: {len(self.chain)} samples")

    def show(self):
        plt.show()

# ====================================================================
#  Entry point
# ====================================================================

def candidate_plot(gaia_id, eccentric=False, smoothing=0, limits="",
                   base_dir=None, rv_dir=None, output_dir="plots"):

    if base_dir is None:
        base_dir = os.path.join(os.path.expanduser("~"),
                                "Projects/subdwarf-rv-simulation/build/out")

    chain_orig, param_names = load_chain(base_dir, gaia_id)
    chain = chain_orig.copy()

    names = ["period", "amplitude", "offset", "phase"]
    if eccentric:
        names.extend(["eccentricity", "omega"])
    names = [n for n in names if n in param_names]

    # phase → T₀
    t_ref = load_t_ref(base_dir, gaia_id)
    t0_offset = int(np.floor(t_ref))

    if "phase" in param_names:
        pi = param_names.index("phase")
        pp = param_names.index("period")
        t0_full = t_ref + chain[:, pi] * chain[:, pp]
        chain[:, pi] = t0_full - t0_offset
        param_names[pi] = "t0"
        for k, nm in enumerate(names):
            if nm == "phase":
                names[k] = "t0"
        PARAM_LABELS["t0"] = rf"$T_0$ [BJD $-$ {t0_offset}]"

    col = {n: param_names.index(n) for n in names}

    lim_dict = {}
    try:
        if limits and limits.strip():
            for tok in limits.split(";"):
                p = tok.strip().split()
                if len(p) >= 3:
                    lim_dict[p[0].strip()] = (float(p[1]), float(p[2]))
    except Exception:
        pass

    # --- load observation times for period resolution ----------------------
    obs_times = load_obs_times(base_dir, gaia_id, rv_dir=rv_dir)
    if obs_times is not None:
        print(f"Loaded {len(obs_times)} observation times "
              f"(baseline: {obs_times.max() - obs_times.min():.1f} d)")
    else:
        print("  Warning: could not load observation times — "
              "using heuristic peak detection resolution")

    # parameter estimates
    print(f"\nParameter estimates for GAIA ID {gaia_id}:")
    print("-" * 60)
    for nm in names:
        v = chain[:, col[nm]]
        med = np.median(v)
        lo16, hi84 = np.percentile(v, 16), np.percentile(v, 84)
        if nm == "t0":
            s, err = format_with_error(med + t0_offset,
                                       lo16 + t0_offset, hi84 + t0_offset)
            print(f"{'T₀ [BJD]':>20s} = {s:>14s} ({err})")
        else:
            s, err = format_with_error(med, lo16, hi84)
            print(f"{param_to_name(nm):>20s} = {s:>14s} ({err})")

    # period peaks
    peaks = (find_period_peaks(chain, col, obs_times=obs_times)
             if "period" in names else [])
    if peaks:
        print("\nTop period peaks:")
        for pi in peaks[:10]:
            print(f"  #{pi['rank']}  P = {pi['period']:.6f} d   "
                  f"(prominence {pi['prom']:.0f},  {pi['n']} samples)")

    os.makedirs(output_dir, exist_ok=True)

    # 1) full corner
    fig_full = build_corner(chain, param_names, names, col,
                            smoothing=smoothing,
                            bounds=lim_dict if lim_dict else None,
                            title=f"Full posterior — Gaia {gaia_id}")
    path_full = os.path.join(output_dir, f"rv_cornerplot_{gaia_id}_full.pdf")
    fig_full.savefig(path_full, dpi=300)
    print(f"\nFull corner plot  → {path_full}")
    plt.close(fig_full)

    # 2) peak corner
    if peaks:
        best = peaks[0]
        sub_chain = chain[best["mask"]]
        print(f"\nPeak corner: P = {best['period']:.6f} d  "
              f"({best['n']} / {len(chain)} samples)")

        peak_bounds = {}
        for nm in names:
            v = sub_chain[:, col[nm]]
            lo, hi = float(v.min()), float(v.max())
            margin = (hi - lo) * 0.08
            if margin == 0:
                margin = 0.01
            peak_bounds[nm] = (lo - margin, hi + margin)

        fig_peak = build_corner(
            sub_chain, param_names, names, col,
            smoothing=smoothing, bounds=peak_bounds,
            title=(f"Peak #1 — P = {best['period']:.6f} d  "
                   f"({best['n']} samples)"))
        path_peak = os.path.join(output_dir,
                                 f"rv_cornerplot_{gaia_id}_peak.pdf")
        fig_peak.savefig(path_peak, dpi=300)
        print(f"Peak corner plot  → {path_peak}")
        plt.close(fig_peak)

        print(f"\nPeak #1 parameter estimates:")
        print("-" * 60)
        for nm in names:
            v = sub_chain[:, col[nm]]
            med = np.median(v)
            lo16, hi84 = np.percentile(v, 16), np.percentile(v, 84)
            if nm == "t0":
                s, err = format_with_error(med + t0_offset,
                                           lo16 + t0_offset,
                                           hi84 + t0_offset)
                print(f"{'T₀ [BJD]':>20s} = {s:>14s} ({err})")
            else:
                s, err = format_with_error(med, lo16, hi84)
                print(f"{param_to_name(nm):>20s} = {s:>14s} ({err})")

    # 3) interactive
    ic = InteractiveCorner(chain, param_names, names,
                           smoothing=smoothing, limits=lim_dict)
    ic.show()
# ====================================================================

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Interactive corner plot from MCMC chain")
    p.add_argument("gaia_id", type=int)
    p.add_argument("--eccentric", action="store_true")
    p.add_argument("--smoothing", type=float, default=0)
    p.add_argument("--limits", type=str, default="")
    p.add_argument("--base-dir", type=str, default=None)
    p.add_argument("--rv-dir", type=str, default=None)
    p.add_argument("--output-dir", type=str, default="plots")
    a = p.parse_args()
    candidate_plot(a.gaia_id, eccentric=a.eccentric, smoothing=a.smoothing,
                   limits=a.limits, base_dir=a.base_dir, rv_dir=a.rv_dir,
                   output_dir=a.output_dir)