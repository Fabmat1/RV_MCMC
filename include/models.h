#pragma once

#include <string>
#include <vector>

// ---------------------------------------------------------------------------
//  MCMC configuration
// ---------------------------------------------------------------------------
struct MCMCConfig {
    bool eccentric = false;

    // Histogram grid
    int n_period_bins  = 100;
    int n_param_bins   = 1000;

    // Hard parameter limits (legacy - used as defaults)
    double amp_lim    = 500.0;
    double offset_lim = 500.0;

    // Parameter bounds (configurable)
    double amp_min    = 0.0;
    double amp_max    = 0.0;      // 0 → use amp_lim
    double offset_min = 0.0;      // 0 → use -offset_lim
    double offset_max = 0.0;      // 0 → use offset_lim
    double phase_min  = -0.5;
    double phase_max  = 0.5;
    double ecc_min    = 0.0;
    double ecc_max    = 0.9999;
    double omega_min  = 0.0;
    double omega_max  = 360.0;

    // Period search range (days)
    double min_period = 0.05;
    double max_period = 50.0;

    // MCMC sizing
    int n_samples  = 0;
    int n_burn_in  = 1000000;

    // Proposal step sizes (0 → use sensible defaults)
    // These now serve as the *initial* diagonal of the proposal covariance.
    // After the adaptation window they are replaced by the learned covariance.
    double period_step        = 0.0;
    double amp_step           = 0.0;
    double offset_step        = 0.0;
    double phase_step         = 0.0;
    double eccentricity_step  = 0.0;
    double omega_step         = 0.0;

    // Starting values (0 → auto-determine)
    double period_0        = 0.0;
    double amp_0           = 0.0;
    double offset_0        = 0.0;
    double phase_0         = 0.0;
    double eccentricity_0  = 0.0;
    double omega_0         = 0.0;

    // Flags
    bool noplot   = false;
    bool lc_prior = false;
    std::vector<std::vector<double>> lc_pgram_data;

    // --- Parallel tempering ---
    int    n_temperatures    = 16;       // number of temperature rungs
    double max_temperature   = 100.0;   // highest temperature
    int    swap_interval     = 20;      // attempt swap every N steps

    // --- Adaptive Metropolis ---
    int    adapt_start       = 1000;    // start adapting after this many steps
    int    adapt_interval    = 100;     // recompute covariance every N steps
    double target_accept     = 0.234;   // optimal for d>=5 (Roberts et al. 1997)
    double adapt_scale_min   = 1e-12;    // floor on global scale factor
    double adapt_scale_max   = 100.0;   // ceiling on global scale factor

    // Add to MCMCConfig struct:
    int chain_thin = 10;                 // Thinning factor for chain output
    std::string chain_output_dir = "";   // Directory for chain.bin / chain_meta.txt

    // If non-null, every (post-burn-in, thinned) sample is also pushed here.
    // One row per sample, length = (eccentric ? 6 : 4):
    //   [period, amplitude, offset, phase, (eccentricity, omega)]
    std::vector<std::vector<double>>* chain_buffer = nullptr;
};


// ---------------------------------------------------------------------------
//  Star
// ---------------------------------------------------------------------------
struct Star {
    double amplitude = 0.0;
    double period    = 0.0;
    double offset    = 0.0;
    double phase     = 0.0;
    int    Npoints   = 0;

    std::vector<double> samples;
    std::vector<double> datapoints;
    std::vector<double> datapoint_errors;

    std::vector<double> periodogram_x;
    std::vector<double> periodogram_y;

    // 6 base histograms
    std::vector<std::vector<double>> period_amp_histogram;
    std::vector<std::vector<double>> period_offset_histogram;
    std::vector<std::vector<double>> period_phase_histogram;
    std::vector<std::vector<double>> amp_offset_histogram;
    std::vector<std::vector<double>> amp_phase_histogram;
    std::vector<std::vector<double>> offset_phase_histogram;

    // 9 eccentric histograms
    std::vector<std::vector<double>> period_ecc_histogram;
    std::vector<std::vector<double>> period_omega_histogram;
    std::vector<std::vector<double>> amp_ecc_histogram;
    std::vector<std::vector<double>> amp_omega_histogram;
    std::vector<std::vector<double>> offset_ecc_histogram;
    std::vector<std::vector<double>> offset_omega_histogram;
    std::vector<std::vector<double>> phase_ecc_histogram;
    std::vector<std::vector<double>> phase_omega_histogram;
    std::vector<std::vector<double>> ecc_omega_histogram;

    std::vector<std::vector<double>> period_amp_histogram_two;

    // Main entry point
    void run_rv_mcmc(const MCMCConfig& cfg);

    // Legacy
    void calculate_orbit_prediction(int Nx, int Ny,
                                    double amp_lim, double offset_lim);
    void run_rv_mcmc_binary(int Nx, int Ny, double amp_lim, double offset_lim,
                            int N_sim, double min_p, double max_p,
                            double min_p_two, double max_p_two,
                            double period_step, double amp_step,
                            double offset_step, double phase_step,
                            double period_0, double amp_0, double offset_0,
                            double phase_0, double period_0_two,
                            double amp_0_two, double phase_0_two,
                            int N_burn_in);
    void process_star(int index,
                      double forced_min_p = 0.0,
                      double forced_max_p = 0.0,
                      int    forced_N     = 0);
    void sort_samples();
};

double rv_curve(double t, double amplitude, double gamma, double period,
                double t_zero_point, double omega, double eccentricity);
void normalize_histogram(std::vector<std::vector<double>>& histogram);