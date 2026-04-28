#pragma once
//
//  Public C++ API of the RV_MCMC submodule.
//  This is the *only* header ASTRA should need to include.
//

#include "models.h"   // re-exports MCMCConfig

#include <map>
#include <string>
#include <vector>

namespace rv_mcmc {

// ----------------------------- Inputs ---------------------------------------

struct RVData {
    std::vector<double> bjd;     // observation times (BJD)
    std::vector<double> rv;      // radial velocities  (km/s)
    std::vector<double> rv_err;  // 1-sigma errors     (km/s)
};

struct LCPriorData {
    std::vector<double> periods; // period grid (days), monotonic
    std::vector<double> powers;  // periodogram power on that grid (>=0)
};

// ----------------------------- Outputs --------------------------------------

struct Histogram1D {
    std::string         param_name;
    std::vector<double> edges;     // length n_bins+1
    std::vector<double> counts;    // length n_bins
    bool                log_scale = false; // edges are linear in log10 if true
};

struct Histogram2D {
    std::string                       x_param, y_param;
    std::vector<double>               x_edges, y_edges;
    std::vector<std::vector<double>>  counts; // [nx][ny]
    bool                              x_log = false, y_log = false;
};

// Lower-triangular corner: diagonals[k] is the marginal of param k,
// off_diagonals[i][j] (i>j) is the joint marginal of params (j on x, i on y).
struct CornerPlot {
    std::vector<std::string>                 param_names;
    std::vector<Histogram1D>                 diagonals;
    std::vector<std::vector<Histogram2D>>    off_diagonals;
};

struct ParamEstimate {
    double median = 0.0;
    double q16    = 0.0;   // 16th percentile (lower 1-sigma)
    double q84    = 0.0;   // 84th percentile (upper 1-sigma)
};

struct Solution {
    int    rank       = 0;
    double period     = 0.0;
    double prominence = 0.0;
    int    n_samples  = 0;
    std::map<std::string, ParamEstimate> parameters;
    CornerPlot corner;     // built only from samples inside this peak
};

struct FitResult {
    bool        success = false;
    std::string error_message;

    // Reference time: T0 [BJD] = t_ref + phase * period
    double t_ref = 0.0;

    std::vector<std::string>              param_names; // length = chain row dim
    std::vector<std::vector<double>>      chain;       // [n_samples][n_params]

    std::vector<double> periodogram_periods;  // days (descending or ascending)
    std::vector<double> periodogram_power;

    CornerPlot              full_corner;
    std::vector<Solution>   solutions;        // sorted by prominence (descending)
};

// ----------------------------- Configuration helpers ------------------------

// Convenience builder for ASTRA: returns an MCMCConfig with sensible defaults
// for a typical sub-dwarf RV fit.
MCMCConfig default_config(bool eccentric = false);

// ----------------------------- Main entry point -----------------------------

// Runs the full pipeline (periodogram → MCMC → peak detection → histograms).
//   data      : RV observations (≥4 points required).
//   cfg       : MCMC + bounds (see MCMCConfig in models.h).
//   lc_prior  : optional LC periodogram prior; pass nullptr to disable.
//
// Configuration knobs added by this API (set on cfg before calling):
//   cfg.n_param_bins  → bins per 1D marginal (default 100, capped 500)
//   cfg.n_period_bins → bins along period axis in 2D plots (default 100)
FitResult run_fit(const RVData&        data,
                  MCMCConfig           cfg,
                  const LCPriorData*   lc_prior = nullptr);

} // namespace rv_mcmc