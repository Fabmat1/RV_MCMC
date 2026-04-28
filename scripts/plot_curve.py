#!/usr/bin/env python3
"""
Plot phase-folded RV curve (and lightcurves).

--use-peak : interactively choose among the top period aliases.
Phase is converted to T₀ [BJD] for all printed output.
"""

import argparse, os, sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from matplotlib.gridspec import GridSpec
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks
from scipy.stats import binned_statistic

mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["Arial"]

PAUL_TOL_COLORS = {
    "bright":  {"blue": "#4477AA", "cyan": "#66CCEE", "green": "#228833",
                "yellow": "#CCBB44", "red": "#EE6677", "purple": "#AA3377",
                "grey": "#BBBBBB"},
    "vibrant": {"blue": "#0077BB", "cyan": "#33BBEE", "teal": "#009988",
                "orange": "#EE7733", "red": "#CC3311", "magenta": "#EE3377",
                "grey": "#BBBBBB"},
    "muted":   {"rose": "#CC6677", "indigo": "#332288", "sand": "#DDCC77",
                "green": "#117733", "cyan": "#88CCEE", "wine": "#882255",
                "teal": "#44AA99", "olive": "#999933", "purple": "#AA4499"},
}

# ====================================================================
#  Chain I/O
# ====================================================================

def load_chain(base_dir, gaia_id):
    base = os.path.join(base_dir, str(gaia_id))
    meta = os.path.join(base, "chain_meta.txt")
    binf = os.path.join(base, "chain.bin")
    if not os.path.exists(meta) or not os.path.exists(binf):
        return None, None
    with open(meta) as f:
        np_ = int(f.readline().strip())
        pn  = f.readline().strip().split()
    raw = np.fromfile(binf, dtype=np.float64)
    ns  = len(raw) // np_
    if ns == 0: return None, None
    return raw[: ns * np_].reshape(ns, np_), pn

# ====================================================================
#  Parameter extraction
# ====================================================================

def get_best_fit_params(gaia_id, eccentric=True, base_dir=None):
    if base_dir is None:
        base_dir = os.path.expanduser("~/Projects/subdwarf_rv_simulation/out")
    chain, pn = load_chain(base_dir, gaia_id)
    if chain is None: return None
    col = {n: i for i, n in enumerate(pn)}
    names = ["period", "amplitude", "offset", "phase"]
    if eccentric:
        for x in ("eccentricity", "omega"):
            if x in col: names.append(x)
    return {n: float(np.median(chain[:, col[n]])) for n in names}


def get_peak_params_interactive(gaia_id, eccentric=True, base_dir=None,
                                n_period_bins=80000, rv_dir=None,
                                sample_factor=5.0):
    if base_dir is None:
        base_dir = os.path.expanduser("~/Projects/subdwarf_rv_simulation/out")
    chain, pn = load_chain(base_dir, gaia_id)
    if chain is None:
        print("No chain data found."); return None
    col = {n: i for i, n in enumerate(pn)}
    periods = chain[:, col["period"]]
    log_p   = np.log10(periods)
    log_range = log_p.max() - log_p.min()

    # --- load observation times for resolution computation -----------------
    obs_times = None
    if rv_dir is not None:
        rv_file = os.path.join(rv_dir, str(gaia_id), "RV_variation.csv")
    else:
        rv_file = os.path.expanduser(
            f"~/Projects/RVVD_refit_2025/output/{gaia_id}/RV_variation.csv")
    if os.path.exists(rv_file):
        try:
            rv = pd.read_csv(rv_file)
            obs_times = rv["BJD"].values
            obs_times = obs_times[np.isfinite(obs_times)]
            if len(obs_times) < 2:
                obs_times = None
        except Exception:
            obs_times = None

    # --- determine histogram parameters ------------------------------------
    if obs_times is not None:
        res = compute_period_resolution(
            obs_times,
            min_p=float(periods.min()),
            max_p=float(periods.max()),
            sample_factor=sample_factor,
        )
        target_bw = res["log_bw_at_min"]
        alias_log = res["alias_log_at_min"]

        nb = int(np.ceil(log_range / target_bw)) if target_bw > 0 else n_period_bins
        nb = int(np.clip(nb, 5000, 1_000_000))

        edges = np.linspace(log_p.min() - 1e-10,
                            log_p.max() + 1e-10, nb + 1)
        bw = edges[1] - edges[0]

        sigma_bins = np.clip(alias_log / (3.0 * bw), 0.5, 8.0)
        min_dist = max(1, int(alias_log / (2.0 * bw)))

        print(f"\n  [Peak detection] observation-based: {nb} bins, "
              f"bw={bw:.2e} log₁₀(d), σ={sigma_bins:.1f} bins, "
              f"min_dist={min_dist} bins, "
              f"alias spacing={alias_log:.2e} log₁₀(d), "
              f"baseline={res['x_ptp']:.1f} d")
    else:
        target_bw = 0.00005
        nb = int(np.clip(log_range / target_bw, 5000, n_period_bins))
        nb = max(nb, min(n_period_bins, int(np.sqrt(len(chain)) * 20)))

        edges = np.linspace(log_p.min() - 1e-10,
                            log_p.max() + 1e-10, nb + 1)
        bw = edges[1] - edges[0]
        sigma_bins = np.clip(0.00025 / bw, 0.5, 4.0)
        min_dist = max(1, int(0.00015 / bw))

    # --- build & smooth histogram ------------------------------------------
    indices = np.searchsorted(edges, log_p, side='right') - 1
    indices = np.clip(indices, 0, nb - 1)
    hist = np.bincount(indices, minlength=nb)[:nb]
    centers = 0.5 * (edges[1:] + edges[:-1])

    hist_s = gaussian_filter(hist.astype(float), sigma=sigma_bins)

    # --- find peaks --------------------------------------------------------
    pks, props = find_peaks(hist_s,
                            height=hist_s.max() * 0.01,
                            prominence=hist_s.max() * 0.005,
                            distance=min_dist)
    if len(pks) == 0:
        pk_idx = int(np.argmax(hist_s))
        pks = np.array([pk_idx])
        props = {"prominences": np.array([hist_s[pk_idx]]),
                 "peak_heights": np.array([hist_s[pk_idx]])}

    # --- non-overlapping masks via troughs ---------------------------------
    pos_order = np.argsort(pks)
    s_pks = pks[pos_order]
    n_p = len(s_pks)

    trough_bins = []
    for k in range(n_p - 1):
        seg = hist_s[s_pks[k]:s_pks[k + 1] + 1]
        trough_bins.append(s_pks[k] + int(np.argmin(seg)))

    masks_sorted = []
    for k in range(n_p):
        lo_bin = trough_bins[k - 1] if k > 0 else 0
        hi_bin = trough_bins[k] if k < n_p - 1 else len(hist_s)
        lp_lo, lp_hi = edges[lo_bin], edges[hi_bin]
        masks_sorted.append((log_p >= lp_lo) & (log_p < lp_hi))

    # --- sort by prominence ------------------------------------------------
    order = np.argsort(props["prominences"])[::-1]

    peaks_info = []
    for rank, oi in enumerate(order):
        sorted_pos = int(np.where(pos_order == oi)[0][0])
        mask = masks_sorted[sorted_pos]
        pk = pks[oi]
        peaks_info.append(dict(rank=rank + 1, period=10 ** centers[pk],
                               prom=props["prominences"][oi],
                               n=int(mask.sum()), mask=mask))

    print(f"\n  Histogram: {nb} bins, bw={bw:.6f} log₁₀(d), "
          f"σ={sigma_bins:.1f} bins, min_dist={min_dist} bins")
    print(f"\n{'#':>3s}  {'Period [d]':>14s}  {'Prominence':>11s}  {'Samples':>9s}")
    print("  " + "-" * 42)
    for pi in peaks_info[:15]:
        print(f"{pi['rank']:>3d}  {pi['period']:>14.6f}  "
              f"{pi['prom']:>11.0f}  {pi['n']:>9d}")

    if sys.stdin.isatty():
        try:
            raw_input = input(f"\nSelect peak number [1]: ").strip()
            sel = 0 if raw_input == "" else int(raw_input) - 1
            if sel < 0 or sel >= len(peaks_info):
                print("Invalid, using #1"); sel = 0
        except (ValueError, EOFError):
            sel = 0
    else:
        sel = 0

    chosen = peaks_info[sel]
    print(f"\n→ Peak #{chosen['rank']}: P = {chosen['period']:.6f} d  "
          f"({chosen['n']} samples)")
    sub = chain[chosen["mask"]]
    names = ["period", "amplitude", "offset", "phase"]
    if eccentric:
        for x in ("eccentricity", "omega"):
            if x in col: names.append(x)
    return {n: float(np.median(sub[:, col[n]])) for n in names}
    
        
# ====================================================================
#  RV model
# ====================================================================

def rv_curve(t, amplitude, gamma, period, t_zero_point, omega, eccentricity):
    M = 2 * np.pi * (t - t_zero_point * period) / period
    M = np.fmod(M + np.pi, 2 * np.pi) - np.pi
    E = M.copy()
    for _ in range(100):
        f = E - eccentricity * np.sin(E) - M
        fp = 1 - eccentricity * np.cos(E)
        d = f / fp; E -= d
        if np.all(np.abs(d) < 1e-6): break
    cosE, sinE = np.cos(E), np.sin(E)
    nu = np.arctan2(np.sqrt(1 - eccentricity**2) * sinE, cosE - eccentricity)
    wr = np.deg2rad(omega)
    return gamma + amplitude * (np.cos(nu + wr) + eccentricity * np.cos(wr))

def sinusoid(t, amplitude, period, offset, phase):
    return amplitude * np.sin(2 * np.pi * (t / period + phase)) + offset

def phase_fold(times, period, t0):
    ph = ((times - t0) / period) % 1; ph[ph > 0.5] -= 1; return ph

def bin_data(x, y, yerr=None, nbins=None):
    if nbins is None: nbins = int(np.cbrt(len(x)))
    si = np.argsort(x); xs, ys = x[si], y[si]
    be = np.linspace(xs.min(), xs.max(), nbins + 1)
    bc = 0.5 * (be[:-1] + be[1:])
    yb, _, _ = binned_statistic(xs, ys, statistic="mean", bins=be)
    ct, _, _ = binned_statistic(xs, ys, statistic="count", bins=be)
    if yerr is not None:
        es = yerr[si]
        eq, _, _ = binned_statistic(xs, es**2, statistic="sum", bins=be)
        eb = np.sqrt(eq) / ct
    else: eb = None
    m = ~np.isnan(yb)
    return bc[m], yb[m], eb[m] if eb is not None else None


# ====================================================================
#  LC helpers
# ====================================================================

def get_filter_color(tel, filt):
    p = PAUL_TOL_COLORS["muted"]
    cm = {"GAIA": {"BP": p["indigo"], "G": p["sand"], "RP": p["rose"]},
          "ZTF":  {"zg": p["green"], "zr": p["rose"], "zi": p["wine"]},
          "ATLAS": {"c": p["cyan"], "o": PAUL_TOL_COLORS["vibrant"]["orange"]},
          "TESS": PAUL_TOL_COLORS["bright"]["blue"],
          "BLACKGEM": p["olive"]}
    if tel in cm:
        return cm[tel].get(filt, p["purple"]) if isinstance(cm[tel], dict) else cm[tel]
    return "#808080"

def load_lightcurve_multifilter(fp, tc, fc, ec, fltc, toff=0):
    if not os.path.exists(fp): return {}
    try:
        try:
            d = pd.read_csv(fp, sep=",", header=None)
            t = d.iloc[:, tc].values + toff; f = d.iloc[:, fc].values
            e = d.iloc[:, ec].values if ec < d.shape[1] else None
            fl = d.iloc[:, fltc].values if fltc is not None and fltc < d.shape[1] else None
        except Exception:
            d = np.loadtxt(fp, delimiter=",")
            if d.ndim == 1: return {}
            t = d[:, tc] + toff; f = d[:, fc]
            e = d[:, ec] if ec < d.shape[1] else None; fl = None
        
        # Ensure numeric types
        t = np.asarray(t, dtype=np.float64)
        f = np.asarray(f, dtype=np.float64)
        if e is not None:
            e = np.asarray(e, dtype=np.float64)
        
        # Filter out non-finite values
        valid = np.isfinite(t) & np.isfinite(f)
        if e is not None:
            valid &= np.isfinite(e)
        if not np.any(valid):
            return {}
        
        t, f = t[valid], f[valid]
        if e is not None:
            e = e[valid]
        if fl is not None:
            fl = fl[valid]
        
        if t.mean() > 2400000: t -= 2400000.5
        if fl is None: return {"default": (t, f, e)}
        r = {}
        for u in np.unique(fl):
            m = fl == u; r[str(u)] = (t[m], f[m], e[m] if e is not None else None)
        return r
    except Exception as ex:
        print(f"Error loading {fp}: {ex}"); return {}

def calculate_lightcurve_chis(phases, fluxes, errors, mp, mf):
    im = interp1d(mp, mf, kind="linear", bounds_error=False, fill_value=np.nan)
    return (fluxes - im(phases)) / errors

def parse_binning_config(s):
    if s is None: return {}
    cfg = {}
    for item in s.split(","):
        item = item.strip()
        if ":" in item:
            t, v = item.split(":", 1)
            t = t.strip().upper(); v = v.strip().lower()
            if v in ("none", "no"):   cfg[t] = None
            elif v == "auto":         cfg[t] = "auto"
            else:
                try:    cfg[t] = int(v)
                except: cfg[t] = "auto"
    return cfg


# ====================================================================
#  Main plot
# ====================================================================

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


def plot_rv_and_lightcurves(
        gaia_id, params=None, eccentric=True, use_peak=False,
        lc_bins=None, lc_model_path=None, lc_model_telescope=None,
        lc_model_t0_offset=0.0, show_lightcurves=True,
        selected_lightcurves=None, binning_config=None,
        show_rv_legend=True, show_lc_legend=True,
        legend_loc="upper right", figsize=None, fontsize=None,
        output_path=None, dpi=300,
        base_dir=None, rv_dir=None, lc_dir=None, output_dir="plots",
        cli_args=None):

    # --- resolve params ---
    if params is None:
        if use_peak:
            print("Extracting parameters from period peaks …")
            params = get_peak_params_interactive(
                gaia_id, eccentric=eccentric, base_dir=base_dir)
        else:
            params = get_peak_params_interactive(
            gaia_id, eccentric=eccentric, base_dir=base_dir, rv_dir=rv_dir)
        if params is None:
            print(f"Could not determine parameters for {gaia_id}."); return
        tag = "(peak)" if use_peak else "(median)"
        print(f"Using auto-determined parameters {tag}:")
    else:
        print("Using user-provided parameters:")

    if cli_args is not None:
        for key in ("period", "offset", "amplitude", "phase"):
            v = getattr(cli_args, key, None)
            if v is not None: params[key] = v
        if eccentric:
            if getattr(cli_args, "omega", None) is not None:
                params["omega"] = cli_args.omega
            if getattr(cli_args, "ecc", None) is not None:
                params["eccentricity"] = cli_args.ecc

    period    = params["period"]
    amplitude = params["amplitude"]
    offset    = params["offset"]
    phase     = params["phase"]           # kept for model evaluation
    omega     = params.get("omega", 0)    if eccentric else 0
    ecc_val   = params.get("eccentricity", 0) if eccentric else 0

    # --- load RV ---
    if rv_dir is None:
        rv_file = os.path.expanduser(
            f"~/Projects/RVVD_refit_2025/output/{gaia_id}/RV_variation.csv")
    else:
        rv_file = os.path.join(rv_dir, str(gaia_id), "RV_variation.csv")
    if not os.path.exists(rv_file):
        print(f"RV file not found: {rv_file}"); return

    rv = pd.read_csv(rv_file)
    rv_t   = rv["BJD"].values
    rv_v   = rv["culum_fit_RV"].values
    rv_e   = rv["u_culum_fit_RV"].values
    rv_tel = rv["TEL"].values

    t_min = rv_t.min()
    ts    = rv_t - t_min

    # ------------------------------------------------------------------
    #  Compute T₀ from phase for display
    # ------------------------------------------------------------------
    t0_bjd = t_min + phase * period

    # ------------------------------------------------------------------
    #  Print parameters — show T₀ instead of phase
    # ------------------------------------------------------------------
    print(f"  {'period':<15}: {period:.6f} d")
    print(f"  {'amplitude':<15}: {amplitude:.4f} km/s")
    print(f"  {'offset':<15}: {offset:.4f} km/s")
    print(f"  {'T₀':<15}: {t0_bjd:.6f} BJD")
    if eccentric:
        print(f"  {'eccentricity':<15}: {ecc_val:.4f}")
        print(f"  {'omega':<15}: {omega:.4f} deg")

    # --- phase-fold & model ---
    ph_off = phase * period if eccentric else 0
    rv_ph  = phase_fold(ts, period, ph_off)

    pg = np.linspace(-1, 1, 1000)
    tg = pg * period + ph_off
    if eccentric:
        mrv  = rv_curve(tg, amplitude, offset, period, phase, omega, ecc_val)
        omrv = rv_curve(ts, amplitude, offset, period, phase, omega, ecc_val)
    else:
        mrv  = sinusoid(tg, amplitude, period, offset, phase)
        omrv = sinusoid(ts, amplitude, period, offset, phase)

    res  = rv_v - omrv
    chi2 = np.sum((res / rv_e) ** 2)
    dof  = len(rv_v) - (6 if eccentric else 4)
    chi2r = chi2 / dof if dof > 0 else float("inf")
    print(f"  {'chi2/dof':<15}: {chi2r:.2f}")

    # --- lightcurves ---
    if show_lightcurves:
        lc_base = (os.path.expanduser(
            f"~/workspace/lightcurvequery/lightcurves/{gaia_id}")
            if lc_dir is None else os.path.join(lc_dir, str(gaia_id)))
        all_lc = {
            "TESS":     load_lightcurve_multifilter(os.path.join(lc_base, "tess_lc.txt"),  0,1,2, None, 2457000-2400000.5),
            "BLACKGEM": load_lightcurve_multifilter(os.path.join(lc_base, "bg_lc.txt"),    0,1,2, None, 0),
            "ATLAS":    load_lightcurve_multifilter(os.path.join(lc_base, "atlas_lc.txt"), 0,1,2, 3, 0),
            "ZTF":      load_lightcurve_multifilter(os.path.join(lc_base, "ztf_lc.txt"),   0,1,2, 3, 0),
            "GAIA":     load_lightcurve_multifilter(os.path.join(lc_base, "gaia_lc.txt"),  0,1,2, 3, 2455197.5-2400000.5),
        }
        all_lc = {k: v for k, v in all_lc.items() if v}
        if selected_lightcurves is not None:
            sel = {s.upper() for s in selected_lightcurves}
            lightcurves = {k: v for k, v in all_lc.items() if k.upper() in sel}
        else:
            lightcurves = all_lc
    else:
        lightcurves = {}

    npan = len(lightcurves)

    lc_model_data = None
    if lc_model_path and os.path.exists(lc_model_path):
        try:
            mr = np.loadtxt(lc_model_path)
            mp = mr[:, 0]; mf = mr[:, 2] / np.nanmedian(mr[:, 2])
            mp = (mp - lc_model_t0_offset) % 1.0; mp[mp > 0.5] -= 1.0
            si = np.argsort(mp); lc_model_data = (mp[si], mf[si])
        except Exception as e:
            print(f"LC model error: {e}")

    # --- figure ---
    dfs = {"general": 8, "labels": 8, "legend": 6, "ticks": 7}
    if fontsize: dfs.update(fontsize)
    if figsize is None: figsize = (7, 3 + 1.5 * npan)
    plt.rcParams.update({
        "figure.figsize": figsize, "font.size": dfs["general"],
        "axes.labelsize": dfs["labels"], "legend.fontsize": dfs["legend"],
        "xtick.labelsize": dfs["ticks"], "ytick.labelsize": dfs["ticks"],
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
        "axes.linewidth": 0.8, "lines.linewidth": 1.0, "patch.linewidth": 0.8,
    })
    fig = plt.figure(figsize=figsize)
    hr = [2, 0.7]; nr = 2
    for tel in lightcurves:
        hr.append(2); nr += 1
        if lc_model_data and lc_model_telescope == tel:
            hr.append(0.7); nr += 1
    gs = GridSpec(nr, 1, height_ratios=hr, hspace=0)
    ax_rv  = fig.add_subplot(gs[0])
    ax_res = fig.add_subplot(gs[1], sharex=ax_rv)

    icol = {0: PAUL_TOL_COLORS["muted"]["rose"],
            1: PAUL_TOL_COLORS["muted"]["indigo"],
            2: PAUL_TOL_COLORS["muted"]["green"],
            3: PAUL_TOL_COLORS["muted"]["wine"],
            4: PAUL_TOL_COLORS["muted"]["cyan"],
            5: PAUL_TOL_COLORS["muted"]["teal"],
            6: PAUL_TOL_COLORS["muted"]["olive"],
            7: PAUL_TOL_COLORS["muted"]["olive"]}
    inam = {0: "LAMOST", 1: "SDSS", 2: "SOAR", 3: "LAMOST",
            4: "EFOSC", 5: "ALFOSC", 6: "UVES", 7: "NOT"}

    for idx in np.unique(rv_tel):
        m = rv_tel == idx
        for sh in (-1, 0, 1):
            ax_rv.errorbar(rv_ph[m]+sh, rv_v[m], yerr=rv_e[m], fmt=".",
                           color=icol.get(idx,"k"), markersize=10, elinewidth=1.2,
                           label=inam.get(idx,f"Inst {idx}") if sh==0 else "",
                           zorder=5, markeredgecolor="white", markeredgewidth=0.4)
    ax_rv.plot(pg, mrv, "k-", lw=1.2, label="Model", zorder=3)
    ax_rv.set_ylabel("RV (km/s)")
    if show_rv_legend:
        ax_rv.legend(loc=legend_loc, framealpha=0.9).set_zorder(99)
    plt.setp(ax_rv.get_xticklabels(), visible=False)

    chi_r = res / rv_e
    for idx in np.unique(rv_tel):
        m = rv_tel == idx
        for sh in (-1, 0, 1):
            ax_res.errorbar(rv_ph[m]+sh, chi_r[m], yerr=1, fmt=".",
                            color=icol.get(idx,"k"), markersize=8, elinewidth=1.2,
                            zorder=5, markeredgecolor="white", markeredgewidth=0.4)
    ax_res.axhline(0, color="grey", ls="--", lw=0.8, zorder=3)
    ax_res.set_ylabel(r"$\chi_{\mathrm{RV}}$"); ax_res.set_ylim(-4,4)
    if npan == 0: ax_res.set_xlabel("Phase")
    else: plt.setp(ax_res.get_xticklabels(), visible=False)

    if binning_config is None: binning_config = {}

    lpi = 2
    for ii, (tel, fd) in enumerate(lightcurves.items()):
        ax_lc = fig.add_subplot(gs[lpi], sharex=ax_rv); lpi += 1
        sm = lc_model_data and lc_model_telescope == tel
        ax_chi = fig.add_subplot(gs[lpi], sharex=ax_rv) if sm else None
        if sm: lpi += 1
        for fn, (tt, ff, ee) in fd.items():
            lts = tt - t_min
            lph = phase_fold(lts, period, ph_off)
            if not eccentric:
                lph = (lph + phase) % 1.0; lph[lph > 0.5] -= 1.0
            fm = np.median(ff); fn_ = ff / fm
            en = ee / fm if ee is not None else None
            col_ = get_filter_color(tel, fn)
            lab = f"{tel}-{fn}" if fn != "default" else tel
            if lab == "BLACKGEM": lab = "CAHA"
            tu = tel.upper()
            if tu in binning_config:
                bs = binning_config[tu]
                nb = None if bs is None else (50 if bs=="auto" and len(tt)>250 else (None if bs=="auto" else bs))
            elif lc_bins is not None: nb = lc_bins
            elif len(tt)>250: nb = 50
            else: nb = None
            if nb:
                pb,fb,eb = bin_data(lph, fn_, en, nbins=nb)
                for sh in (-1,0,1):
                    ax_lc.errorbar(pb+sh,fb,yerr=eb,fmt=".",color=col_,
                                   markersize=6,elinewidth=0.5,
                                   label=f"{lab} (n={nb})" if sh==0 else "",
                                   zorder=5,markeredgecolor="white",markeredgewidth=0.4)
                pc,fc,ec_ = pb,fb,eb
            else:
                for sh in (-1,0,1):
                    ax_lc.errorbar(lph+sh,fn_,yerr=en,fmt=".",color=col_,
                                   markersize=6,elinewidth=0.5,
                                   label=lab if sh==0 else "",zorder=5,
                                   markeredgecolor="white",markeredgewidth=0.4)
                pc,fc,ec_ = lph,fn_,en
            if sm and ax_chi:
                plt.setp(ax_lc.get_xticklabels(), visible=False)
                try:
                    mp_,mf_ = lc_model_data
                    for sh in (-1,0,1):
                        ch = calculate_lightcurve_chis(pc+sh,fc,ec_,mp_+sh,mf_)
                        ax_chi.errorbar(pc+sh,ch,yerr=1,fmt=".",color=col_,
                                        markersize=5,elinewidth=0.5,
                                        label=fn if sh==0 else "",zorder=5,
                                        markeredgecolor="white",markeredgewidth=0.4)
                except Exception as e: print(f"Chi failed {tel}-{fn}: {e}")
        if sm and ax_chi:
            ax_chi.axhline(0,color="grey",ls="--",lw=0.8)
            ax_chi.set_ylabel(r"$\chi_{\mathrm{LC}}$"); ax_chi.set_ylim(-4,4)
            if tel==list(lightcurves)[-1]: ax_chi.set_xlabel("Phase")
            else: plt.setp(ax_chi.get_xticklabels(), visible=False)
        if lc_model_data and lc_model_telescope==tel:
            mp_,mf_ = lc_model_data
            for sh in (-1,0,1):
                ax_lc.plot(mp_+sh,mf_,"k-",lw=1.2,
                           label="Model" if sh==0 else "",zorder=3)
        ax_lc.set_ylabel("Rel. Flux")
        if show_lc_legend:
            ax_lc.legend(loc=legend_loc,ncol=2,framealpha=0.9).set_zorder(99)
        ax_lc.axhline(1,color="gray",ls=":",lw=0.8)
        if ii==npan-1 and not (sm and ax_chi): ax_lc.set_xlabel("Phase")
        else: plt.setp(ax_lc.get_xticklabels(), visible=False)

    ax_rv.set_xlim(-1,1)
    plt.tight_layout(pad=0.0,h_pad=0.0)
    os.makedirs(output_dir, exist_ok=True)
    if output_path is None:
        output_path = os.path.join(output_dir, f"{gaia_id}_rvcurve.pdf")
    plt.savefig(output_path, bbox_inches="tight", dpi=dpi, pad_inches=0)
    print(f"Figure saved to: {output_path}")
    plt.show()
    return fig


# ====================================================================
if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Plot phase-folded RV curve and lightcurves.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("gaia_id", type=int)
    p.add_argument("--use-peak", action="store_true",
                   help="Interactively choose among period aliases")
    p.add_argument("--period", type=float)
    p.add_argument("--amplitude", type=float)
    p.add_argument("--offset", type=float)
    p.add_argument("--phase", type=float, help="Phase (internal); will be shown as T₀")
    p.add_argument("--omega", type=float, help="degrees")
    p.add_argument("--ecc", type=float)
    p.add_argument("--non-eccentric", action="store_true")
    p.add_argument("--lightcurves", type=str, default=None)
    p.add_argument("--no-lightcurves", action="store_true")
    p.add_argument("--no-rv-legend", action="store_true")
    p.add_argument("--no-lc-legend", action="store_true")
    p.add_argument("--legend-loc", type=str, default="upper right")
    p.add_argument("--lc_bins", type=int, default=None)
    p.add_argument("--binning", type=str, default=None)
    p.add_argument("--lc_model_path", type=str)
    p.add_argument("--lc_model_telescope", type=str)
    p.add_argument("--lc_model_t0_offset", type=float, default=0.0)
    p.add_argument("--output", "-o", type=str, default=None)
    p.add_argument("--figsize", type=str, default=None)
    p.add_argument("--fontsize", type=float, default=None)
    p.add_argument("--label-fontsize", type=float, default=None)
    p.add_argument("--legend-fontsize", type=float, default=None)
    p.add_argument("--tick-fontsize", type=float, default=None)
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--base-dir", type=str, default=None)
    p.add_argument("--rv-dir", type=str, default=None)
    p.add_argument("--lc-dir", type=str, default=None)
    p.add_argument("--output-dir", type=str, default="plots")
    args = p.parse_args()

    ecc = not args.non_eccentric
    sel_lc = [s.strip() for s in args.lightcurves.split(",")] if args.lightcurves else None
    bc = parse_binning_config(args.binning)
    fs = None
    if args.figsize:
        try: w,h = args.figsize.split(","); fs = (float(w),float(h))
        except: pass
    fsd = None
    if any([args.fontsize, args.label_fontsize, args.legend_fontsize, args.tick_fontsize]):
        fsd = {}
        if args.fontsize:        fsd["general"] = args.fontsize
        if args.label_fontsize:  fsd["labels"]  = args.label_fontsize
        if args.legend_fontsize: fsd["legend"]  = args.legend_fontsize
        if args.tick_fontsize:   fsd["ticks"]   = args.tick_fontsize

    plot_rv_and_lightcurves(
        args.gaia_id, params=None, eccentric=ecc, use_peak=args.use_peak,
        lc_bins=args.lc_bins,
        lc_model_path=args.lc_model_path,
        lc_model_telescope=args.lc_model_telescope,
        lc_model_t0_offset=args.lc_model_t0_offset,
        show_lightcurves=not args.no_lightcurves,
        selected_lightcurves=sel_lc, binning_config=bc,
        show_rv_legend=not args.no_rv_legend,
        show_lc_legend=not args.no_lc_legend,
        legend_loc=args.legend_loc, figsize=fs, fontsize=fsd,
        output_path=args.output, dpi=args.dpi,
        base_dir=args.base_dir, rv_dir=args.rv_dir,
        lc_dir=args.lc_dir, output_dir=args.output_dir,
        cli_args=args)