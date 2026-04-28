#pragma once

#include "rv_mcmc/api.h"          // brings in rv_mcmc::MCMCConfig
#include <string>
#include <vector>

using MCMCConfig = rv_mcmc::MCMCConfig;

// ---------------------------------------------------------------------------
//  RVMCMC_Star
// ---------------------------------------------------------------------------
struct RVMCMC_Star {
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