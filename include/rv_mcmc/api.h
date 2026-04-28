#pragma once
//
//  Public C++ API of the RV_MCMC submodule.
//  IMPORTANT: this header is the *only* one ASTRA should include.
//             It must NOT pull in models.h (which leaks `Star`).
//

#include <map>
#include <string>
#include <vector>

namespace rv_mcmc {

// ─────────────────────────── MCMC configuration ────────────────────────────
struct MCMCConfig {
    bool eccentric = false;
    int  n_period_bins  = 100;
    int  n_param_bins   = 1000;

    double amp_lim    = 500.0;
    double offset_lim = 500.0;
    double amp_min    = 0.0;
    double amp_max    = 0.0;
    double offset_min = 0.0;
    double offset_max = 0.0;
    double phase_min  = -0.5;
    double phase_max  = 0.5;
    double ecc_min    = 0.0;
    double ecc_max    = 0.9999;
    double omega_min  = 0.0;
    double omega_max  = 360.0;

    double min_period = 0.05;
    double max_period = 50.0;

    int n_samples = 0;
    int n_burn_in = 1000000;

    double period_step = 0.0, amp_step = 0.0, offset_step = 0.0;
    double phase_step  = 0.0, eccentricity_step = 0.0, omega_step = 0.0;

    double period_0 = 0.0, amp_0 = 0.0, offset_0 = 0.0;
    double phase_0  = 0.0, eccentricity_0 = 0.0, omega_0 = 0.0;

    bool noplot   = false;
    bool lc_prior = false;
    std::vector<std::vector<double>> lc_pgram_data;

    int    n_temperatures  = 16;
    double max_temperature = 100.0;
    int    swap_interval   = 20;
    int    adapt_start     = 1000;
    int    adapt_interval  = 100;
    double target_accept   = 0.234;
    double adapt_scale_min = 1e-12;
    double adapt_scale_max = 100.0;

    int         chain_thin = 10;
    std::string chain_output_dir = "";
    std::vector<std::vector<double>>* chain_buffer = nullptr;
};

// ─────────────────────────── Inputs / Outputs ──────────────────────────────
struct RVData       { std::vector<double> bjd, rv, rv_err; };
struct LCPriorData  { std::vector<double> periods, powers; };

struct Histogram1D  {
    std::string param_name;
    std::vector<double> edges, counts;
    bool log_scale = false;
};
struct Histogram2D  {
    std::string x_param, y_param;
    std::vector<double> x_edges, y_edges;
    std::vector<std::vector<double>> counts;
    bool x_log = false, y_log = false;
};
struct CornerPlot {
    std::vector<std::string> param_names;
    std::vector<Histogram1D> diagonals;
    std::vector<std::vector<Histogram2D>> off_diagonals;
};
struct ParamEstimate { double median = 0, q16 = 0, q84 = 0; };
struct Solution {
    int rank = 0; double period = 0, prominence = 0; int n_samples = 0;
    std::map<std::string, ParamEstimate> parameters;
    CornerPlot corner;
};
struct FitResult {
    bool success = false;
    std::string error_message;
    double t_ref = 0.0;
    std::vector<std::string> param_names;
    std::vector<std::vector<double>> chain;
    std::vector<double> periodogram_periods, periodogram_power;
    CornerPlot full_corner;
    std::vector<Solution> solutions;
};

MCMCConfig default_config(bool eccentric = false);

FitResult run_fit(const RVData&     data,
                  MCMCConfig        cfg,
                  const LCPriorData* lc_prior = nullptr);

} // namespace rv_mcmc