import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd
import os
import argparse
from scipy.stats import binned_statistic
from scipy.interpolate import interp1d
import matplotlib as mpl

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial']

## MODIFICATION: Paul Tol's colorblind-friendly palette
# https://personal.sron.nl/~pault/
PAUL_TOL_COLORS = {
    'bright': {
        'blue':    '#4477AA',
        'cyan':    '#66CCEE',
        'green':   '#228833',
        'yellow':  '#CCBB44',
        'red':     '#EE6677',
        'purple':  '#AA3377',
        'grey':    '#BBBBBB',
    },
    'vibrant': {
        'blue':    '#0077BB',
        'cyan':    '#33BBEE',
        'teal':    '#009988',
        'orange':  '#EE7733',
        'red':     '#CC3311',
        'magenta': '#EE3377',
        'grey':    '#BBBBBB',
    },
    'muted': {
        'rose':    '#CC6677',
        'indigo':  '#332288',
        'sand':    '#DDCC77',
        'green':   '#117733',
        'cyan':    '#88CCEE',
        'wine':    '#882255',
        'teal':    '#44AA99',
        'olive':   '#999933',
        'purple':  '#AA4499',
    },
}

def weighted_quantile(values, quantiles, weights):
    """Calculate weighted quantiles."""
    sorter = np.argsort(values)
    v, w = values[sorter], weights[sorter]
    cumw = np.cumsum(w)
    cumw /= cumw[-1]
    return np.interp(quantiles, cumw, v)

def get_best_fit_params(gaia_id, eccentric=True, base_dir=None):
    """Extract best-fit parameters from histogram data."""
    if base_dir is None:
        base = os.path.join(
            "/home/fabian",
            "Projects/subdwarf_rv_simulation",
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
    
    if eccentric:
        names = ['period', 'amplitude', 'offset', 'phase', 'eccentricity', 'omega']
        combinations = [('period','amplitude'), ('period','offset'), ('period','phase'), ('period','eccentricity'), ('period','omega'), ('amplitude','offset'), ('amplitude','phase'), ('amplitude','eccentricity'), ('amplitude','omega'), ('offset','phase'), ('offset','eccentricity'), ('offset','omega'), ('phase','eccentricity'), ('phase','omega'), ('eccentricity','omega')]
    else:
        names = ['period', 'amplitude', 'offset', 'phase']
        combinations = [('period','amplitude'), ('period','offset'), ('period','phase'), ('amplitude','offset'), ('amplitude','phase'), ('offset','phase')]
    
    for a, b in combinations:
        data = load2d(f"{a}_vs_{b}")
        if data is not None: H[(a,b)] = data
    
    if not H: return None
    
    Nx, Ny = next(iter(H.values())).shape
    
    pgram = np.loadtxt(os.path.join(base, "pgram.csv"), delimiter=',')
    pgram_x, _ = pgram.T
    edges = {
        'period':    np.geomspace(pgram_x.min(), pgram_x.max(), Nx+1),
        'amplitude': np.linspace(0, 500.0, Ny+1),
        'offset':    np.linspace(-500.0, 500.0, Ny+1),
        'phase':     np.linspace(-0.5, 0.5, Ny+1),
    }
    
    if eccentric:
        edges['eccentricity'] = np.linspace(0.0, 1.0, Ny+1)
        edges['omega'] = np.linspace(0.0, 2*np.pi, Ny+1)
    
    marg = {}
    for α in names:
        w = np.zeros(len(edges[α]) - 1)
        for β in names:
            if α == β: continue
            key = (α,β) if (α,β) in H else (β,α)
            if key not in H: continue
            arr = H[key]
            w += arr.sum(axis=0) if key[0] == α else arr.sum(axis=1)
        marg[α] = w
    
    params = {}
    for α in names:
        centers = 0.5*(edges[α][1:] + edges[α][:-1])
        wts = marg[α] / marg[α].sum()
        params[α] = weighted_quantile(centers, [0.5], wts)[0]
    
    if 'omega' in params: params['omega'] *= 180 / np.pi
    
    return params

def rv_curve(t, amplitude, gamma, period, t_zero_point, omega, eccentricity):
    """Python implementation of the C++ RV curve function."""
    M = 2 * np.pi * (t - (t_zero_point * period)) / period
    M = np.fmod(M + np.pi, 2 * np.pi) - np.pi
    
    E = M.copy()
    for _ in range(100):
        f = E - eccentricity * np.sin(E) - M
        fprime = 1 - eccentricity * np.cos(E)
        delta = f / fprime
        E -= delta
        if np.all(np.abs(delta) < 1e-6): break
    
    cosE, sinE = np.cos(E), np.sin(E)
    nu = np.arctan2(np.sqrt(1 - eccentricity**2) * sinE, cosE - eccentricity)
    omega_rad = np.deg2rad(omega)
    
    return gamma + amplitude * (np.cos(nu + omega_rad) + eccentricity * np.cos(omega_rad))

def sinusoid(t, amplitude, period, offset, phase):
    """Python implementation of the sinusoid function."""
    return amplitude * np.sin(2 * np.pi * (t / period + phase)) + offset

def phase_fold(times, period, t0):
    """Phase fold times given period and reference time."""
    phases = ((times - t0) / period) % 1
    phases[phases > 0.5] -= 1
    return phases

def bin_data(x, y, yerr=None, nbins=None):
    """Bin data points."""
    if nbins is None:
        nbins = int(np.cbrt(len(x)))
    
    # Sort by x
    sorted_idx = np.argsort(x)
    x_sorted = x[sorted_idx]
    y_sorted = y[sorted_idx]
    
    # Create bins
    bin_edges = np.linspace(x_sorted.min(), x_sorted.max(), nbins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Bin the data
    y_binned, _, _ = binned_statistic(x_sorted, y_sorted, statistic='mean', bins=bin_edges)
    counts, _, _ = binned_statistic(x_sorted, y_sorted, statistic='count', bins=bin_edges)
    
    if yerr is not None:
        yerr_sorted = yerr[sorted_idx]
        # Propagate errors: sigma_mean = sqrt(sum(sigma_i^2)) / n
        yerr_squared, _, _ = binned_statistic(x_sorted, yerr_sorted**2, statistic='sum', bins=bin_edges)
        yerr_binned = np.sqrt(yerr_squared) / counts
    else:
        yerr_binned = None
    
    # Remove empty bins
    mask = ~np.isnan(y_binned)
    
    return bin_centers[mask], y_binned[mask], yerr_binned[mask] if yerr_binned is not None else None

## MODIFICATION: Color function using Paul Tol's palette (Req 3)
def get_filter_color(telescope, filter_name):
    """Get appropriate color for telescope filter from Paul Tol's palette."""
    palette = PAUL_TOL_COLORS['muted']
    color_map = {
        'GAIA': {'BP': palette['indigo'], 'G': palette['sand'], 'RP': palette['rose']},
        'ZTF': {'zg': palette['green'], 'zr': palette['rose'], 'zi': palette['wine']},
        'ATLAS': {'c': palette['cyan'], 'o': PAUL_TOL_COLORS['vibrant']['orange']},
        'TESS': PAUL_TOL_COLORS['bright']['blue'],
        'BLACKGEM': palette['olive'],
    }
    
    if telescope in color_map:
        if isinstance(color_map[telescope], dict):
            return color_map[telescope].get(filter_name, palette['purple'])
        return color_map[telescope]
    return '#808080' # Default gray

def load_lightcurve_multifilter(filepath, time_col, flux_col, err_col, filter_col, time_offset=0):
    """Load lightcurve data with filter information."""
    if not os.path.exists(filepath):
        return {}
    
    try:
        # Try to read with pandas first for string handling
        try:
            data = pd.read_csv(filepath, sep=',', header=None)
            times = data.iloc[:, time_col].values + time_offset
            fluxes = data.iloc[:, flux_col].values
            errors = data.iloc[:, err_col].values if err_col < data.shape[1] else None
            filters = data.iloc[:, filter_col].values if filter_col < data.shape[1] else None
        except:
            # Fallback to numpy
            data = np.loadtxt(filepath, delimiter=",")
            if data.ndim == 1:
                return {}
            times = data[:, time_col] + time_offset
            fluxes = data[:, flux_col]
            errors = data[:, err_col] if err_col < data.shape[1] else None
            filters = None
        
        # Convert to MJD if needed
        if times.mean() > 2400000:
            times = times - 2400000.5
        
        # If no filters, return single dataset
        if filters is None:
            return {'default': (times, fluxes, errors)}
        
        # Split by filter
        result = {}
        unique_filters = np.unique(filters)
        for filt in unique_filters:
            mask = filters == filt
            result[str(filt)] = (times[mask], fluxes[mask], errors[mask] if errors is not None else None)
        
        return result
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return {}

def calculate_lightcurve_chis(phases, fluxes, errors, model_phase, model_flux):
    """Interpolate model onto data phases and return normalized residuals (chi)."""
    if errors is None or np.any(errors == 0):
        raise ValueError("Flux errors must be non-zero for chi calculation.")
    
    interp_model = interp1d(model_phase, model_flux, kind='linear', bounds_error=False, fill_value=np.nan)
    model_interp_flux = interp_model(phases)
    chi = (fluxes - model_interp_flux) / errors
    return chi

def parse_binning_config(binning_str):
    """
    Parse binning configuration string.
    
    Format: "TELESCOPE1:NBINS,TELESCOPE2:NBINS,..." or "TELESCOPE1:none,..."
    Example: "TESS:50,ZTF:30,ATLAS:none,GAIA:auto"
    
    Returns dict: {telescope: nbins} where nbins is int, None (no binning), or 'auto'
    """
    if binning_str is None:
        return {}
    
    config = {}
    for item in binning_str.split(','):
        item = item.strip()
        if ':' in item:
            telescope, value = item.split(':', 1)
            telescope = telescope.strip().upper()
            value = value.strip().lower()
            
            if value == 'none' or value == 'no':
                config[telescope] = None  # No binning
            elif value == 'auto':
                config[telescope] = 'auto'  # Automatic binning
            else:
                try:
                    config[telescope] = int(value)
                except ValueError:
                    print(f"Warning: Invalid binning value '{value}' for {telescope}, using auto")
                    config[telescope] = 'auto'
    
    return config


def plot_rv_and_lightcurves(gaia_id, params=None, eccentric=True, lc_bins=None, 
                            lc_model_path=None, lc_model_telescope=None, lc_model_t0_offset=0.0,
                            show_lightcurves=True, selected_lightcurves=None, binning_config=None,
                            show_rv_legend=True, show_lc_legend=True, legend_loc='upper right',
                            figsize=None, fontsize=None, output_path=None, dpi=300,
                            base_dir=None, rv_dir=None, lc_dir=None, output_dir="plots"):
    """
    Plot phase-folded RV curve and lightcurves.
    
    Parameters
    ----------
    ...
    base_dir : str, optional
        Base directory for histogram data (default: ~/Projects/subdwarf_rv_simulation/out)
    rv_dir : str, optional
        Directory containing RV files (default: ~/Projects/RVVD_refit_2025/output)
    lc_dir : str, optional
        Directory containing lightcurve files (default: ~/workspace/lightcurvequery/lightcurves)
    output_dir : str, optional
        Output directory for plots (default: plots)
    ...
    """
    
    if params is None:
        params = get_best_fit_params(gaia_id, eccentric=eccentric, base_dir=base_dir)
        if params is None:
            print(f"Could not determine parameters for {gaia_id}. Aborting.")
            return
        print(f"Using auto-determined parameters from histograms:")
    else:
        print(f"Using user-provided parameters:")

    if args.period is not None:
        params["period"] = args.period
    if args.offset is not None:
        params["offset"] = args.offset
    if args.amplitude is not None:
        params["amplitude"] = args.amplitude
    if args.phase is not None:
        params["phase"] = args.phase

    if eccentric:
        if args.omega is not None:
            params['omega'] = args.omega
        if args.ecc is not None:
            params['eccentricity'] = args.ecc

    period, amplitude, offset, phase = params['period'], params['amplitude'], params['offset'], params['phase']
    omega = params.get('omega', 0) if eccentric else 0
    eccentricity = params.get('eccentricity', 0) if eccentric else 0

    if rv_dir is None:
        rv_file = os.path.expanduser(f"~/Projects/RVVD_refit_2025/output/{gaia_id}/RV_variation.csv")
    else:
        rv_file = os.path.join(rv_dir, str(gaia_id), "RV_variation.csv")
    
    if not os.path.exists(rv_file):
        print(f"RV file not found: {rv_file}"); return
    
    rv_data = pd.read_csv(rv_file)
    rv_times, rv_values, rv_errors, rv_telescope = rv_data['BJD'].values, rv_data['culum_fit_RV'].values, rv_data['u_culum_fit_RV'].values, rv_data['TEL'].values
    
    t_min = rv_times.min()
    rv_times_shifted = rv_times - t_min
    
    rv_phase_offset = phase * period if eccentric else 0
    rv_phases = phase_fold(rv_times_shifted, period, rv_phase_offset)

    phase_grid = np.linspace(-1, 1, 1000)
    time_grid = phase_grid * period + rv_phase_offset
    if eccentric:
        model_rv = rv_curve(time_grid, amplitude, offset, period, phase, omega, eccentricity)
        observed_model = rv_curve(rv_times_shifted, amplitude, offset, period, phase, omega, eccentricity)
    else:
        model_rv = sinusoid(time_grid, amplitude, period, offset, phase)
        observed_model = sinusoid(rv_times_shifted, amplitude, period, offset, phase)

    residuals = rv_values - observed_model
    chi2 = np.sum((residuals / rv_errors)**2)
    dof = len(rv_values) - (6 if eccentric else 4)
    chi2_reduced = chi2 / dof if dof > 0 else float('inf')
    
    ## Print parameters including chi2/dof to console
    for key, value in params.items():
        unit = 'd' if key=='period' else ('km/s' if key in ['amplitude', 'offset'] else ('deg' if key=='omega' else ''))
        print(f"  {key:<12}: {value:.4f} {unit}")
    print(f"  {'chi2/dof':<12}: {chi2_reduced:.2f}")

    # Load lightcurves
    if show_lightcurves:
        if lc_dir is None:
            lc_base = os.path.expanduser(f"~/workspace/lightcurvequery/lightcurves/{gaia_id}")
        else:
            lc_base = os.path.join(lc_dir, str(gaia_id))
        
        all_lightcurves = {
            'TESS': load_lightcurve_multifilter(os.path.join(lc_base, "tess_lc.txt"), 0, 1, 2, None, 2457000 - 2400000.5),
            'BLACKGEM': load_lightcurve_multifilter(os.path.join(lc_base, "bg_lc.txt"), 0, 1, 2, None, 0),
            'ATLAS': load_lightcurve_multifilter(os.path.join(lc_base, "atlas_lc.txt"), 0, 1, 2, 3, 0),
            'ZTF': load_lightcurve_multifilter(os.path.join(lc_base, "ztf_lc.txt"), 0, 1, 2, 3, 0),
            'GAIA': load_lightcurve_multifilter(os.path.join(lc_base, "gaia_lc.txt"), 0, 1, 2, 3, 2455197.5 - 2400000.5),
        }

        all_lightcurves = {
            tel: lc for tel, lc in all_lightcurves.items() if lc != {}
        }

        # Filter to selected lightcurves if specified
        if selected_lightcurves is not None:
            selected_upper = [s.upper() for s in selected_lightcurves]
            lightcurves = {k: v for k, v in all_lightcurves.items() if k.upper() in selected_upper}
            # Warn about requested but unavailable lightcurves
            available = set(all_lightcurves.keys())
            requested = set(s.upper() for s in selected_lightcurves)
            missing = requested - set(k.upper() for k in available)
            if missing:
                print(f"Warning: Requested lightcurves not available: {missing}")
                print(f"Available lightcurves: {list(available)}")
        else:
            lightcurves = all_lightcurves
    else:
        lightcurves = {}
    
    n_panels = len(lightcurves)

    # Load lightcurve model if provided
    lc_model_data = None
    if lc_model_path and os.path.exists(lc_model_path):
        try:
            model_raw = np.loadtxt(lc_model_path)
            # Phase is col 0, flux is col 2
            model_phase = model_raw[:, 0]
            model_flux = model_raw[:, 2]

            model_flux = model_flux/np.nanmedian(model_flux)
            # Apply user-specified phase offset and wrap to [-0.5, 0.5]
            model_phase_shifted = (model_phase - lc_model_t0_offset) % 1.0
            model_phase_shifted[model_phase_shifted > 0.5] -= 1.0
            # Sort for clean line plotting
            sort_idx = np.argsort(model_phase_shifted)
            lc_model_data = (model_phase_shifted[sort_idx], model_flux[sort_idx])
            print(f"Loaded lightcurve model for {lc_model_telescope} from {lc_model_path}")
        except Exception as e:
            print(f"Could not load or parse lightcurve model: {e}")

    # Set up font sizes
    default_fontsize = {
        'general': 8,
        'labels': 8,
        'legend': 6,
        'ticks': 7
    }
    if fontsize is not None:
        default_fontsize.update(fontsize)
    
    # Set up figure size
    if figsize is None:
        figsize = (7, 3 + 1.5 * n_panels)  # Adjust height based on number of panels
    
    # Set plot style
    plot_params = {
        "figure.figsize": figsize,
        "font.size": default_fontsize['general'],
        "axes.labelsize": default_fontsize['labels'],
        "legend.fontsize": default_fontsize['legend'],
        "xtick.labelsize": default_fontsize['ticks'],
        "ytick.labelsize": default_fontsize['ticks'],
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.0,
        "patch.linewidth": 0.8,
    }
    plt.rcParams.update(plot_params)

    fig = plt.figure(figsize=figsize)
    
    # Compute height ratios and number of rows
    height_ratios = [2, 0.7]
    nrows = 2
    for tel in lightcurves:
        height_ratios.append(2)
        nrows += 1
        if lc_model_data and lc_model_telescope == tel:
            height_ratios.append(0.7)
            nrows += 1

    gs = GridSpec(nrows, 1, height_ratios=height_ratios, hspace=0)

    ax_rv = fig.add_subplot(gs[0])
    ax_res = fig.add_subplot(gs[1], sharex=ax_rv)
    
    # RV Plot
    instrument_colors = {
        0: PAUL_TOL_COLORS['muted']['rose'],
        1: PAUL_TOL_COLORS['muted']['indigo'],
        2: PAUL_TOL_COLORS['muted']['green'],
        3: PAUL_TOL_COLORS['muted']['wine'],
        4: PAUL_TOL_COLORS['muted']['cyan'],
        5: PAUL_TOL_COLORS['muted']['teal'],
        6: PAUL_TOL_COLORS['muted']['olive'],
        7: PAUL_TOL_COLORS['muted']['olive'],
    }
    
    instrument_names = {
        0: 'LAMOST',
        1: 'SDSS',
        2: 'SOAR',
        3: 'LAMOST',
        4: 'EFOSC',
        5: 'ALFOSC',
        6: 'UVES',
        7: 'NOT'
    }

    for idx in np.unique(rv_telescope):
        mask = rv_telescope == idx
        label = instrument_names.get(idx, f'Instrument {idx}')
        color = instrument_colors.get(idx, 'black')

        for shift in [-1, 0, 1]:
            ax_rv.errorbar(rv_phases[mask] + shift, rv_values[mask], yerr=rv_errors[mask],
                           fmt='.', color=color, markersize=10, elinewidth=1.2,
                           label=label if shift == 0 else '', zorder=5, 
                           markeredgecolor='white', markeredgewidth=0.4)

    ax_rv.plot(phase_grid, model_rv, 'k-', linewidth=1.2, label='Model', zorder=3)
    ax_rv.set_ylabel('RV (km/s)')
    
    # Show RV legend if requested
    if show_rv_legend:
        legend = ax_rv.legend(loc=legend_loc, framealpha=0.9)
        legend.set_zorder(99)
    
    plt.setp(ax_rv.get_xticklabels(), visible=False)

    # Residuals Plot
    chi_residuals = residuals / rv_errors
    for idx in np.unique(rv_telescope):
        mask = rv_telescope == idx
        color = instrument_colors.get(idx, 'black')
        for shift in [-1, 0, 1]:
            ax_res.errorbar(rv_phases[mask] + shift, chi_residuals[mask], yerr=1, 
                           fmt='.', color=color, markersize=8, elinewidth=1.2, 
                           zorder=5, markeredgecolor='white', markeredgewidth=0.4)
    ax_res.axhline(0, color='grey', linestyle='--', linewidth=0.8, zorder=3)
    ax_res.set_ylabel(r'$\chi_{\mathrm{RV}}$')
    ax_res.set_ylim(-4, 4)
    if n_panels == 0:
        ax_res.set_xlabel('Phase')
    else:
        plt.setp(ax_res.get_xticklabels(), visible=False)

    # Initialize binning config if not provided
    if binning_config is None:
        binning_config = {}

    # Lightcurve Plots
    lc_panel_idx = 2
    for i, (telescope, filter_dict) in enumerate(lightcurves.items()):
        ax_lc = fig.add_subplot(gs[lc_panel_idx], sharex=ax_rv)
        lc_panel_idx += 1

        show_model = lc_model_data and lc_model_telescope == telescope
        ax_chi = fig.add_subplot(gs[lc_panel_idx], sharex=ax_rv) if show_model else None
        if show_model:
            lc_panel_idx += 1

        for filter_name, (times, fluxes, errors) in filter_dict.items():
            lc_times_shifted = times - t_min
            lc_phases = phase_fold(lc_times_shifted, period, rv_phase_offset)
            if not eccentric:
                lc_phases = (lc_phases + phase) % 1.0
                lc_phases[lc_phases > 0.5] -= 1.0
            
            flux_median = np.median(fluxes)
            flux_norm = fluxes / flux_median
            flux_err_norm = errors / flux_median if errors is not None else None
            
            color = get_filter_color(telescope, filter_name)
            label = f'{telescope}-{filter_name}' if filter_name != 'default' else telescope
            
            print(label)
            if label == "BLACKGEM":
                label = "CAHA"

            # Determine binning for this telescope
            tel_upper = telescope.upper()
            if tel_upper in binning_config:
                bin_setting = binning_config[tel_upper]
                if bin_setting is None:
                    nbins_to_use = None  # No binning
                elif bin_setting == 'auto':
                    nbins_to_use = 50 if len(times) > 250 else None
                else:
                    nbins_to_use = bin_setting
            elif lc_bins is not None:
                nbins_to_use = lc_bins
            elif len(times) > 250:
                nbins_to_use = 50  # Default auto-binning for large datasets
            else:
                nbins_to_use = None
            
            if nbins_to_use:
                p_bin, f_bin, e_bin = bin_data(lc_phases, flux_norm, flux_err_norm, nbins=nbins_to_use)
                for shift in [-1, 0, 1]:
                    ax_lc.errorbar(p_bin + shift, f_bin, yerr=e_bin, fmt='.', color=color, 
                                  markersize=6, elinewidth=0.5, 
                                  label=f'{label} (n={nbins_to_use})' if shift==0 else '', 
                                  zorder=5, markeredgecolor='white', markeredgewidth=0.4)
                # Store for chi calculation
                phases_for_chi, flux_for_chi, err_for_chi = p_bin, f_bin, e_bin
            else:
                for shift in [-1, 0, 1]:
                    ax_lc.errorbar(lc_phases + shift, flux_norm, yerr=flux_err_norm, fmt='.', 
                                  color=color, markersize=6, elinewidth=0.5, 
                                  label=label if shift==0 else '', 
                                  zorder=5, markeredgecolor='white', markeredgewidth=0.4)
                # Store for chi calculation
                phases_for_chi, flux_for_chi, err_for_chi = lc_phases, flux_norm, flux_err_norm

            if show_model and ax_chi:
                plt.setp(ax_lc.get_xticklabels(), visible=False)
                try:
                    m_phase, m_flux = lc_model_data
                    for shift in [-1, 0, 1]:
                        m_p = m_phase + shift
                        chis = calculate_lightcurve_chis(phases_for_chi + shift, flux_for_chi, 
                                                         err_for_chi, m_p, m_flux)
                        ax_chi.errorbar(phases_for_chi + shift, chis, yerr=1, fmt='.', color=color,
                                       markersize=5, elinewidth=0.5, 
                                       label=f'{filter_name}' if shift == 0 else '', 
                                       zorder=5, markeredgecolor='white', markeredgewidth=0.4)
                except Exception as e:
                    print(f"Could not calculate chi residuals for {telescope}-{filter_name}: {e}")
        
        if show_model and ax_chi:    
            ax_chi.axhline(0, color='grey', linestyle='--', linewidth=0.8)
            ax_chi.set_ylabel(r'$\chi_{\mathrm{LC}}$')
            ax_chi.set_ylim(-4, 4)
            if telescope == list(lightcurves)[-1]:
                ax_chi.set_xlabel('Phase')
            else:
                plt.setp(ax_chi.get_xticklabels(), visible=False)

        # Plot lightcurve model on specified panel
        if lc_model_data and lc_model_telescope == telescope:
            m_phase, m_flux = lc_model_data
            for shift in [-1, 0, 1]:
                ax_lc.plot(m_phase + shift, m_flux, 'k-', linewidth=1.2, 
                          label='Model' if shift == 0 else '', zorder=3)

        ax_lc.set_ylabel('Rel. Flux')
        
        # Show LC legend if requested
        if show_lc_legend:
            legend = ax_lc.legend(loc=legend_loc, ncol=2, framealpha=0.9)
            legend.set_zorder(99)

        ax_lc.axhline(1, color='gray', linestyle=':', linewidth=0.8)
        
        if i == n_panels - 1 and not (show_model and ax_chi):
            ax_lc.set_xlabel('Phase')
        else:
            plt.setp(ax_lc.get_xticklabels(), visible=False)
        
    ax_rv.set_xlim(-1, 1)
    plt.tight_layout(pad=0.0, h_pad=0.0)
    
    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    if output_path is None:
        output_path = os.path.join(output_dir, f"{gaia_id}_rvcurve.pdf")
    
    plt.savefig(output_path, bbox_inches='tight', dpi=dpi, pad_inches=0)
    print(f"Figure saved to: {output_path}")
    
    plt.show()
    
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot phase-folded RV curve and lightcurves for a given GAIA source ID.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with auto parameters
  python script.py 123456789
  
  # Select specific lightcurves
  python script.py 123456789 --lightcurves TESS,ZTF
  
  # Custom binning per telescope
  python script.py 123456789 --binning "TESS:50,ZTF:30,ATLAS:none,GAIA:auto"
  
  # Custom figure size and font
  python script.py 123456789 --figsize 10,8 --fontsize 10
  
  # Save with custom name and no legends
  python script.py 123456789 --output my_plot.pdf --no-rv-legend --no-lc-legend
        """
    )
    parser.add_argument("gaia_id", type=int, help="GAIA source ID")
    
    # Orbital parameter arguments
    parser.add_argument("--period", type=float, help="Period in days")
    parser.add_argument("--amplitude", type=float, help="RV semi-amplitude in km/s")
    parser.add_argument("--offset", type=float, help="Systemic velocity/offset in km/s")
    parser.add_argument("--phase", type=float, help="Phase zero point")
    parser.add_argument("--omega", type=float, help="Argument of periastron in degrees")
    parser.add_argument("--ecc", type=float, help="Eccentricity")
    parser.add_argument("--non-eccentric", action="store_true", help="Use non-eccentric sinusoidal model")
    
    # Lightcurve selection (a)
    parser.add_argument("--lightcurves", type=str, default=None,
                        help="Comma-separated list of lightcurves to plot (e.g., 'TESS,ZTF,ATLAS'). "
                             "Available: TESS, BLACKGEM, ATLAS, ZTF, GAIA. Default: all available.")
    parser.add_argument("--no-lightcurves", action="store_true", 
                        help="Do not show lightcurve plots, only RV curve")
    
    # Legend options (b)
    parser.add_argument("--no-rv-legend", action="store_true", 
                        help="Hide legend on RV panel")
    parser.add_argument("--no-lc-legend", action="store_true", 
                        help="Hide legend on lightcurve panels")
    parser.add_argument("--legend-loc", type=str, default="upper right",
                        help="Legend location (e.g., 'upper right', 'lower left', 'best')")
    
    # Binning options (c)
    parser.add_argument("--lc_bins", type=int, default=None, 
                        help="Default number of bins for all lightcurves (can be overridden by --binning)")
    parser.add_argument("--binning", type=str, default=None,
                        help="Per-telescope binning config: 'TELESCOPE:NBINS,...' "
                             "Use 'none' for no binning, 'auto' for automatic. "
                             "Example: 'TESS:50,ZTF:30,ATLAS:none,GAIA:auto'")
    
    # Lightcurve model arguments
    parser.add_argument("--lc_model_path", type=str, help="Path to the lightcurve model file")
    parser.add_argument("--lc_model_telescope", type=str, 
                        help="Telescope the LC model belongs to (e.g., TESS, ZTF)")
    parser.add_argument("--lc_model_t0_offset", type=float, default=0.0, 
                        help="Phase offset (t0) to apply to the lightcurve model")
    
    # Output options (d)
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output file path (e.g., 'plot.pdf', 'plot.png'). "
                             "Default: plots/<gaia_id>_rvcurve.pdf")
    parser.add_argument("--figsize", type=str, default=None,
                        help="Figure size in inches as 'WIDTH,HEIGHT' (e.g., '7,5')")
    parser.add_argument("--fontsize", type=float, default=None,
                        help="Base font size (scales all text elements)")
    parser.add_argument("--label-fontsize", type=float, default=None,
                        help="Axis label font size")
    parser.add_argument("--legend-fontsize", type=float, default=None,
                        help="Legend font size")
    parser.add_argument("--tick-fontsize", type=float, default=None,
                        help="Tick label font size")
    parser.add_argument("--dpi", type=int, default=300,
                        help="DPI for raster output formats (default: 300)")
    
    # Path options
    parser.add_argument("--base-dir", type=str, default=None,
                        help="Base directory for histogram data (default: ~/Projects/subdwarf_rv_simulation/out)")
    parser.add_argument("--rv-dir", type=str, default=None,
                        help="Directory containing RV files (default: ~/Projects/RVVD_refit_2025/output)")
    parser.add_argument("--lc-dir", type=str, default=None,
                        help="Directory containing lightcurve files (default: ~/workspace/lightcurvequery/lightcurves)")
    parser.add_argument("--output-dir", type=str, default="plots",
                        help="Output directory for plots (default: plots)")

    args = parser.parse_args()
    
    eccentric = not args.non_eccentric
    
    # Build params dict if provided
    params = None
    
    # Parse lightcurve selection
    selected_lightcurves = None
    if args.lightcurves:
        selected_lightcurves = [lc.strip() for lc in args.lightcurves.split(',')]
    
    # Parse binning configuration
    binning_config = parse_binning_config(args.binning)
    
    # Parse figure size
    figsize = None
    if args.figsize:
        try:
            w, h = args.figsize.split(',')
            figsize = (float(w), float(h))
        except ValueError:
            print(f"Warning: Invalid figsize '{args.figsize}', using default")
    
    # Build fontsize dict
    fontsize_dict = None
    if any([args.fontsize, args.label_fontsize, args.legend_fontsize, args.tick_fontsize]):
        fontsize_dict = {}
        if args.fontsize:
            fontsize_dict['general'] = args.fontsize
        if args.label_fontsize:
            fontsize_dict['labels'] = args.label_fontsize
        if args.legend_fontsize:
            fontsize_dict['legend'] = args.legend_fontsize
        if args.tick_fontsize:
            fontsize_dict['ticks'] = args.tick_fontsize
    
    plot_rv_and_lightcurves(
        args.gaia_id, 
        params=params, 
        eccentric=eccentric,
        lc_bins=args.lc_bins,
        lc_model_path=args.lc_model_path,
        lc_model_telescope=args.lc_model_telescope,
        lc_model_t0_offset=args.lc_model_t0_offset,
        show_lightcurves=not args.no_lightcurves,
        selected_lightcurves=selected_lightcurves,
        binning_config=binning_config,
        show_rv_legend=not args.no_rv_legend,
        show_lc_legend=not args.no_lc_legend,
        legend_loc=args.legend_loc,
        figsize=figsize,
        fontsize=fontsize_dict,
        output_path=args.output,
        dpi=args.dpi,
        base_dir=args.base_dir,
        rv_dir=args.rv_dir,
        lc_dir=args.lc_dir,
        output_dir=args.output_dir
    )