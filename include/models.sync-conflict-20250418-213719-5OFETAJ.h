//
// Created by fabia on 08.09.2024.
//

#ifndef SUBDWARF_RV_SIMULATIONS_MODELS_H
#define SUBDWARF_RV_SIMULATIONS_MODELS_H

#endif //SUBDWARF_RV_SIMULATIONS_MODELS_H


#include "vector"
using namespace std;

struct Star {
    double amplitude;
    double period;
    double offset;
    double phase;
    int Npoints;

    vector<double> samples;
    vector<double> datapoints;
    vector<double> datapoint_errors;
    vector<double> periodogram_x;
    vector<double> periodogram_y;

    vector<vector<double>> period_amp_histogram;
    vector<vector<double>> period_amp_histogram_two;    
    vector<vector<double>> period_phase_histogram;
    vector<vector<double>> period_offset_histogram;
    vector<vector<double>> ecc_omega_histogram;

    void calculate_orbit_prediciton(int Nx, int Ny, double amp_lim, double offset_lim);
    void twod_amp_prediciton(int Nx, int Ny, double amp_lim, double offset_lim, int N_sim, double min_p, double max_p, double period_step, double amp_step, double offset_step, double phase_step, double period_0, double amp_0, double offset_0, double phase_0, int N_burn_in);
    void twod_amp_prediciton_eccentric(int Nx, int Ny, double amp_lim, double offset_lim,
                                   int N_sim, double min_p, double max_p, double period_step,
                                   double amp_step, double offset_step, double phase_step,
                                   double eccentricity_step, double omega_step,
                                   double period_0, double amp_0, double offset_0,
                                   double phase_0, double eccentricity_0, double omega_0, int N_burn_in);
    void twod_amp_prediciton_binary(int Nx, int Ny, double amp_lim, double offset_lim,
                               int N_sim, double min_p, double max_p, double min_p_two, double max_p_two, double period_step,
                               double amp_step, double offset_step, double phase_step,
                               double period_0, double amp_0, double offset_0,
                               double phase_0, double period_0_two, double amp_0_two,
                               double phase_0_two, int N_burn_in);
    void process_star(int index, double forced_min_p=0.0, double forced_max_p=0.0, int forced_N=0);
    void sort_samples();
};
