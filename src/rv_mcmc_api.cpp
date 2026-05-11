#include "rv_mcmc/api.h"

#include "lomb_scargle_periodogram.h"
#include "maths.h"
#include "vector_operations.h"
#include "models.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace rv_mcmc {

// =========================================================================
//  Utilities
// =========================================================================

static std::vector<double> gaussian_filter1d(const std::vector<double>& x,
                                              double sigma)
{
    if (sigma <= 0 || x.size() < 3) return x;
    int radius = std::max(1, (int)std::ceil(4.0 * sigma));
    std::vector<double> k(2*radius + 1);
    double s = 0.0;
    for (int i = -radius; i <= radius; ++i) {
        k[i + radius] = std::exp(-0.5 * (i*i) / (sigma*sigma));
        s += k[i + radius];
    }
    for (auto& v : k) v /= s;

    std::vector<double> y(x.size(), 0.0);
    int n = (int)x.size();
    for (int i = 0; i < n; ++i) {
        double acc = 0.0;
        for (int j = -radius; j <= radius; ++j) {
            int q = i + j;
            if (q < 0)      q = -q;        // reflect
            if (q >= n)     q = 2*n - q - 2;
            if (q < 0 || q >= n) continue;
            acc += x[q] * k[j + radius];
        }
        y[i] = acc;
    }
    return y;
}

struct LocalPeak { int idx; double prominence; };

static std::vector<LocalPeak> find_peaks(const std::vector<double>& y,
                                          double min_height,
                                          double min_prominence,
                                          int    min_distance)
{
    std::vector<LocalPeak> peaks;
    int n = (int)y.size();
    for (int i = 1; i < n - 1; ++i) {
        if (y[i] > y[i-1] && y[i] >= y[i+1] && y[i] >= min_height) {
            // Prominence: highest of the lowest minima walking outward
            double left_min  = y[i];
            for (int j = i - 1; j >= 0; --j) {
                if (y[j] > y[i]) break;
                if (y[j] < left_min) left_min = y[j];
            }
            double right_min = y[i];
            for (int j = i + 1; j < n; ++j) {
                if (y[j] > y[i]) break;
                if (y[j] < right_min) right_min = y[j];
            }
            double prom = y[i] - std::max(left_min, right_min);
            if (prom >= min_prominence)
                peaks.push_back({i, prom});
        }
    }
    // Enforce minimum distance, greedy by prominence
    std::vector<LocalPeak> by_prom = peaks;
    std::sort(by_prom.begin(), by_prom.end(),
              [](const LocalPeak& a, const LocalPeak& b){
                  return a.prominence > b.prominence; });
    std::vector<bool> keep(by_prom.size(), true);
    for (size_t a = 0; a < by_prom.size(); ++a) {
        if (!keep[a]) continue;
        for (size_t b = a + 1; b < by_prom.size(); ++b) {
            if (!keep[b]) continue;
            if (std::abs(by_prom[a].idx - by_prom[b].idx) < min_distance)
                keep[b] = false;
        }
    }
    std::vector<LocalPeak> result;
    for (size_t a = 0; a < by_prom.size(); ++a)
        if (keep[a]) result.push_back(by_prom[a]);
    return result;
}

// =========================================================================
//  Histogram building
// =========================================================================

static void make_uniform_edges(double lo, double hi, int n,
                               std::vector<double>& edges)
{
    if (n < 1) n = 1;
    if (hi <= lo) { hi = lo + 1.0; }
    edges.resize(n + 1);
    double step = (hi - lo) / n;
    for (int i = 0; i <= n; ++i) edges[i] = lo + i*step;
}

static int locate_bin(const std::vector<double>& edges, double x)
{
    if (x < edges.front() || x > edges.back()) return -1;
    auto it = std::upper_bound(edges.begin(), edges.end(), x);
    int idx = (int)(it - edges.begin()) - 1;
    if (idx < 0) idx = 0;
    if (idx >= (int)edges.size() - 1) idx = (int)edges.size() - 2;
    return idx;
}

static Histogram1D build_hist1d(const std::vector<std::vector<double>>& chain,
                                 int col,
                                 const std::string& name,
                                 int nbins,
                                 bool log_scale)
{
    Histogram1D h;
    h.param_name = name;
    h.log_scale  = log_scale;
    if (chain.empty()) return h;

    double lo = +1e300, hi = -1e300;
    for (auto& row : chain) {
        double v = log_scale ? std::log10(std::max(row[col], 1e-300)) : row[col];
        if (v < lo) lo = v;
        if (v > hi) hi = v;
    }
    if (!(hi > lo)) { hi = lo + 1.0; }
    double margin = (hi - lo) * 0.02;
    lo -= margin; hi += margin;
    make_uniform_edges(lo, hi, nbins, h.edges);
    h.counts.assign(nbins, 0.0);
    for (auto& row : chain) {
        double v = log_scale ? std::log10(std::max(row[col], 1e-300)) : row[col];
        int b = locate_bin(h.edges, v);
        if (b >= 0) h.counts[b] += 1.0;
    }
    if (log_scale) for (auto& e : h.edges) e = std::pow(10.0, e);
    return h;
}

static Histogram2D build_hist2d(const std::vector<std::vector<double>>& chain,
                                 int colx, int coly,
                                 const std::string& nx, const std::string& ny,
                                 int nbinsx, int nbinsy,
                                 bool xlog, bool ylog)
{
    Histogram2D h;
    h.x_param = nx; h.y_param = ny; h.x_log = xlog; h.y_log = ylog;
    if (chain.empty()) return h;

    double xlo=1e300, xhi=-1e300, ylo=1e300, yhi=-1e300;
    for (auto& r : chain) {
        double xv = xlog ? std::log10(std::max(r[colx], 1e-300)) : r[colx];
        double yv = ylog ? std::log10(std::max(r[coly], 1e-300)) : r[coly];
        xlo = std::min(xlo, xv); xhi = std::max(xhi, xv);
        ylo = std::min(ylo, yv); yhi = std::max(yhi, yv);
    }
    if (!(xhi > xlo)) xhi = xlo + 1.0;
    if (!(yhi > ylo)) yhi = ylo + 1.0;
    double mx = (xhi-xlo)*0.02, my = (yhi-ylo)*0.02;
    make_uniform_edges(xlo-mx, xhi+mx, nbinsx, h.x_edges);
    make_uniform_edges(ylo-my, yhi+my, nbinsy, h.y_edges);
    h.counts.assign(nbinsx, std::vector<double>(nbinsy, 0.0));

    for (auto& r : chain) {
        double xv = xlog ? std::log10(std::max(r[colx], 1e-300)) : r[colx];
        double yv = ylog ? std::log10(std::max(r[coly], 1e-300)) : r[coly];
        int bx = locate_bin(h.x_edges, xv);
        int by = locate_bin(h.y_edges, yv);
        if (bx >= 0 && by >= 0) h.counts[bx][by] += 1.0;
    }
    if (xlog) for (auto& e : h.x_edges) e = std::pow(10.0, e);
    if (ylog) for (auto& e : h.y_edges) e = std::pow(10.0, e);
    return h;
}

static CornerPlot build_corner(const std::vector<std::vector<double>>& chain,
                                const std::vector<std::string>& names,
                                int nbins1d, int nbins2d)
{
    CornerPlot c;
    c.param_names = names;
    int n = (int)names.size();

    auto is_log = [](const std::string& s){ return s == "period"; };

    c.diagonals.reserve(n);
    for (int i = 0; i < n; ++i)
        c.diagonals.push_back(
            build_hist1d(chain, i, names[i], nbins1d, is_log(names[i])));

    c.off_diagonals.assign(n, std::vector<Histogram2D>(n));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < i; ++j)
            c.off_diagonals[i][j] = build_hist2d(
                chain, j, i, names[j], names[i],
                nbins2d, nbins2d, is_log(names[j]), is_log(names[i]));
    return c;
}

// =========================================================================
//  Period-peak detection (port of plot_corner.find_period_peaks)
// =========================================================================

struct PeriodPeak {
    int    rank;
    double period;
    double prominence;
    std::vector<bool> mask;   // length = chain.size()
};

static double compute_alias_log(const std::vector<double>& obs_t,
                                  double min_p)
{
    if (obs_t.size() < 2) return 0.0;
    std::vector<double> t = obs_t;
    std::sort(t.begin(), t.end());
    double T = t.back() - t.front();
    if (T <= 0) return 0.0;
    double n = std::ceil(T / min_p);
    if (n <= 1) n = 2.0;
    double R_p = T/(n-1.0) - T/n;
    return R_p / (min_p * std::log(10.0));   // log10 spacing of aliases at min_p
}

static std::vector<PeriodPeak> detect_period_peaks(
        const std::vector<std::vector<double>>& chain,
        int period_col,
        const std::vector<double>& obs_times)
{
    std::vector<PeriodPeak> peaks;
    if (chain.size() < 50) return peaks;

    std::vector<double> log_p(chain.size());
    for (size_t i = 0; i < chain.size(); ++i)
        log_p[i] = std::log10(std::max(chain[i][period_col], 1e-300));
    double lo = *std::min_element(log_p.begin(), log_p.end());
    double hi = *std::max_element(log_p.begin(), log_p.end());
    if (!(hi > lo)) return peaks;
    double range = hi - lo;

    // ----- determine resolution -----
    int    nb;
    double bw, sigma_bins;
    int    min_dist;
    double min_p_real = std::pow(10.0, lo);
    double alias_log  = compute_alias_log(obs_times, min_p_real);
    if (alias_log > 0) {
        double target_bw = alias_log / 5.0; // sample_factor = 5
        nb = (int)std::ceil(range / target_bw);
        nb = std::max(5000, std::min(nb, 1000000));
    } else {
        nb = std::max(5000, std::min(80000, (int)(std::sqrt(chain.size())*15)));
    }
    bw = range / nb;
    if (alias_log > 0) {
        sigma_bins = std::min(8.0, std::max(0.5, alias_log / (3.0 * bw)));
        min_dist   = std::max(1, (int)(alias_log / (2.0 * bw)));
    } else {
        sigma_bins = std::min(4.0, std::max(0.5, 0.00025 / bw));
        min_dist   = std::max(1, (int)(0.00015 / bw));
    }

    // ----- histogram -----
    std::vector<double> hist(nb, 0.0);
    for (double v : log_p) {
        int b = std::min(nb-1, std::max(0, (int)((v - lo) / bw)));
        hist[b] += 1.0;
    }
    auto smoothed = gaussian_filter1d(hist, sigma_bins);
    double hmax = *std::max_element(smoothed.begin(), smoothed.end());
    if (hmax <= 0) return peaks;

    auto raw = find_peaks(smoothed,
                          hmax * 0.01,    // height
                          hmax * 0.005,   // prominence
                          min_dist);
    if (raw.empty()) return peaks;

    // Sort detected peaks by position (for trough boundaries)
    std::sort(raw.begin(), raw.end(),
              [](const LocalPeak& a, const LocalPeak& b){ return a.idx < b.idx; });

    // Find troughs between consecutive peaks → non-overlapping bins
    std::vector<int> trough_bins;
    for (size_t k = 0; k + 1 < raw.size(); ++k) {
        int a = raw[k].idx, b = raw[k+1].idx;
        int tmin = a;
        for (int j = a; j <= b; ++j)
            if (smoothed[j] < smoothed[tmin]) tmin = j;
        trough_bins.push_back(tmin);
    }

    // Build masks
    int N = (int)raw.size();
    std::vector<std::vector<bool>> masks(N);
    std::vector<std::pair<double,double>> ranges(N);
    for (int k = 0; k < N; ++k) {
        int lo_bin = (k > 0)     ? trough_bins[k-1] : 0;
        int hi_bin = (k < N - 1) ? trough_bins[k]   : nb;
        double lp_lo = lo + lo_bin * bw;
        double lp_hi = lo + hi_bin * bw;
        ranges[k] = {lp_lo, lp_hi};
        masks[k].assign(chain.size(), false);
        for (size_t i = 0; i < chain.size(); ++i)
            if (log_p[i] >= lp_lo && log_p[i] < lp_hi)
                masks[k][i] = true;
    }

    // Sort by prominence (descending) for ranking
    std::vector<int> order(N);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&](int a, int b){ return raw[a].prominence > raw[b].prominence; });

    int rank = 1;
    for (int oi : order) {
        PeriodPeak p;
        p.rank       = rank++;
        p.period     = std::pow(10.0, lo + (raw[oi].idx + 0.5) * bw);
        p.prominence = raw[oi].prominence;
        p.mask       = std::move(masks[oi]);
        peaks.push_back(std::move(p));
    }
    return peaks;
}

// =========================================================================
//  Public API
// =========================================================================

MCMCConfig default_config(bool eccentric)
{
    MCMCConfig c;
    c.eccentric = eccentric;
    c.n_samples = 5'000'000;
    c.n_burn_in = 1'000'000;
    c.chain_thin = 10;
    c.amp_lim    = 500.0;
    c.offset_lim = 500.0;
    c.min_period = 0.05;
    c.max_period = 50.0;
    return c;
}

FitResult run_fit(const RVData& data, MCMCConfig cfg, const LCPriorData* lc_prior)
{
    FitResult R;

    // ---- sanity ----
    if (data.bjd.size() < 4 ||
        data.bjd.size() != data.rv.size() ||
        data.bjd.size() != data.rv_err.size()) {
        R.error_message = "Need >=4 points and matching vector sizes";
        return R;
    }

    // ---- shift time origin ----
    RVMCMC_Star star;
    double t_ref = *std::min_element(data.bjd.begin(), data.bjd.end());
    R.t_ref = t_ref;
    star.samples          = vadd(data.bjd, -t_ref);
    star.datapoints       = data.rv;
    star.datapoint_errors = data.rv_err;
    star.Npoints          = (int)star.samples.size();

    // ---- periodogram (used by the seeder + returned to caller) ----
    auto opt = genOptimalPeriodogramSamples(star.samples, 20,
                                              cfg.min_period, cfg.max_period);
    if (opt[2] < 1000)
        opt = genOptimalPeriodogramSamples(star.samples, 20,
                                             cfg.min_period, cfg.max_period, 1000);
    star.periodogram_y = gls_fast(star.samples, star.datapoints,
                                    std::vector<double>(star.Npoints, 10),
                                    opt[0], opt[1], (int)std::round(opt[2]));
    auto px = linspace(opt[0], opt[0] + opt[2]*opt[1], (int)std::ceil(opt[2]));
    star.periodogram_x = invert(px);
    R.periodogram_periods = star.periodogram_x;
    R.periodogram_power   = star.periodogram_y;

    // ---- LC prior (if provided) ----
    if (lc_prior) {
        if (lc_prior->periods.size() != lc_prior->powers.size()
            || lc_prior->periods.size() < 2) {
            R.error_message = "Invalid LC prior data";
            return R;
        }
        cfg.lc_prior      = true;
        cfg.lc_pgram_data = { lc_prior->periods, lc_prior->powers };
    } else {
        cfg.lc_prior      = false;
        cfg.lc_pgram_data.clear();
    }

        // ---- in-memory chain output ----
    // Honour a caller-supplied buffer (used by ASTRA for live progress
    // polling). Fall back to a local one if none was provided.
    std::vector<std::vector<double>> local_chain;
    std::vector<std::vector<double>>* chain_ptr =
        cfg.chain_buffer ? cfg.chain_buffer : &local_chain;
    chain_ptr->clear();
    chain_ptr->reserve(cfg.n_samples / std::max(cfg.chain_thin, 1) + 1024);
    cfg.chain_buffer     = chain_ptr;
    cfg.chain_output_dir = "";       // no file output
    cfg.noplot           = true;     // disable gnuplot in library mode

    // ---- run MCMC ----
    try {
        star.run_rv_mcmc(cfg);
    } catch (const std::exception& e) {
        R.error_message = std::string("MCMC failed: ") + e.what();
        return R;
    }
    if (chain_ptr->empty()) {
        R.error_message = "MCMC produced no samples";
        return R;
    }

    // ---- parameter names ----
    R.param_names = cfg.eccentric
        ? std::vector<std::string>{"period","amplitude","offset","phase",
                                   "eccentricity","omega"}
        : std::vector<std::string>{"period","amplitude","offset","phase"};
    R.chain = std::move(*chain_ptr);

    // ---- bin counts (configurable) ----
    int n1d = cfg.n_param_bins  > 0 ? std::min(cfg.n_param_bins,  500) : 100;
    int n2d = cfg.n_period_bins > 0 ? std::min(cfg.n_period_bins, 200) : 100;

    // ---- full corner ----
    R.full_corner = build_corner(R.chain, R.param_names, n1d, n2d);

    // ---- detect period peaks & build per-peak solutions ----
    int period_col = 0; // by construction
    auto peaks = detect_period_peaks(R.chain, period_col, data.bjd);

    for (const auto& pk : peaks) {
        Solution s;
        s.rank       = pk.rank;
        s.period     = pk.period;
        s.prominence = pk.prominence;

        // Sub-chain for this peak
        std::vector<std::vector<double>> sub;
        sub.reserve(R.chain.size() / 4);
        for (size_t i = 0; i < R.chain.size(); ++i)
            if (pk.mask[i]) sub.push_back(R.chain[i]);
        s.n_samples = (int)sub.size();
        if (s.n_samples < 10) continue; // ignore noise

        // Per-parameter median + 16/84 percentiles
        for (size_t p = 0; p < R.param_names.size(); ++p) {
            std::vector<double> v(sub.size());
            for (size_t i = 0; i < sub.size(); ++i) v[i] = sub[i][p];
            std::sort(v.begin(), v.end());
            ParamEstimate e;
            e.median = v[v.size() / 2];
            e.q16    = v[(size_t)(v.size() * 0.16)];
            e.q84    = v[(size_t)(v.size() * 0.84)];
            s.parameters[R.param_names[p]] = e;
        }

        // Per-peak corner (smaller bin counts because samples are fewer)
        int sub_n1d = std::max(20, std::min(n1d, (int)std::sqrt((double)sub.size())*3));
        int sub_n2d = std::max(20, std::min(n2d, sub_n1d));
        s.corner = build_corner(sub, R.param_names, sub_n1d, sub_n2d);

        R.solutions.push_back(std::move(s));
    }

    R.success = true;
    return R;
}

} // namespace rv_mcmc