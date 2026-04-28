#include "models.h"
#include "maths.h"
#include "vector_operations.h"
#include "file_io.h"
#include "lomb_scargle_periodogram.h"
#include "gnuplot-iostream.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <tuple>
#include <vector>

#include <omp.h>

using namespace std;

// ===================================================================
//  Utility
// ===================================================================
void normalize_histogram(vector<vector<double>>& histogram) {
    double total = 0.0;
    for (auto& row : histogram)
        total += accumulate(row.begin(), row.end(), 0.0);
    if (total > 0.0)
        for (auto& row : histogram)
            for (auto& v : row)
                v /= total;
}

// ===================================================================
//  Keplerian RV model
// ===================================================================
double rv_curve(double t, double amplitude, double gamma, double period,
                double t_zero_point, double omega, double eccentricity) {
    double M = 2.0 * M_PI * (t - t_zero_point * period) / period;
    M = fmod(M, 2.0 * M_PI);
    if (M < -M_PI) M += 2.0 * M_PI;
    if (M >  M_PI) M -= 2.0 * M_PI;

    double E = M;
    for (int i = 0; i < 100; ++i) {
        double f      = E - eccentricity * sin(E) - M;
        double fprime = 1.0 - eccentricity * cos(E);
        double delta  = f / fprime;
        E -= delta;
        if (fabs(delta) < 1e-6) break;
    }

    double cosE  = cos(E), sinE = sin(E);
    double denom = 1.0 - eccentricity * cosE;
    double sin_nu = sqrt(1.0 - eccentricity * eccentricity) * sinE / denom;
    double cos_nu = (cosE - eccentricity) / denom;
    double nu     = atan2(sin_nu, cos_nu);

    double omega_rad = omega / 360.0 * 2.0 * M_PI;
    return gamma + amplitude * (cos(nu + omega_rad) + eccentricity * cos(omega_rad));
}


// ===================================================================
//  Parallel-tempered, adaptive-Metropolis MCMC for RV fitting.
//
//  This version writes the T=1 chain to a binary file instead of
//  accumulating into fixed-resolution histograms.  Adaptive binning
//  and histogram construction happen in Python post-processing.
// ===================================================================
void Star::run_rv_mcmc(const MCMCConfig& cfg) {

    const bool ecc = cfg.eccentric;
    const int  dim = ecc ? 6 : 4;

    // Parameter order: [log10(period), amplitude, offset, phase, (ecc, omega)]
    enum { iLPER=0, iAMP=1, iOFF=2, iPH=3, iECC=4, iOMG=5 };

    const int  Nsim  = cfg.n_samples;
    const int  Nburn = cfg.n_burn_in;
    const int  Ntot  = Nsim + Nburn;

    const double log_min_p = log10(cfg.min_period);
    const double log_max_p = log10(cfg.max_period);

    // ---- temperature ladder (geometric spacing) ----
    const int Ntemp = max(cfg.n_temperatures, 1);
    vector<double> temperatures(Ntemp);
    if (Ntemp == 1) {
        temperatures[0] = 1.0;
    } else {
        for (int t = 0; t < Ntemp; ++t)
            temperatures[t] = pow(cfg.max_temperature,
                                   (double)t / (Ntemp - 1));
    }
    cout << "Temperature ladder (" << Ntemp << " rungs):";
    for (auto T : temperatures) cout << " " << fixed << setprecision(1) << T;
    cout << "\n";

    // ---- chain output file ----
    const int chain_thin = max(cfg.chain_thin, 1);
    FILE* chain_file = nullptr;
    int   chain_count = 0;

    if (!cfg.chain_output_dir.empty()) {
        string bin_path  = cfg.chain_output_dir + "chain.bin";
        string meta_path = cfg.chain_output_dir + "chain_meta.txt";

        chain_file = fopen(bin_path.c_str(), "wb");
        if (!chain_file)
            throw runtime_error("Cannot open chain file: " + bin_path);
        setvbuf(chain_file, nullptr, _IOFBF, 1 << 20);   // 1 MB buffer

        // Write metadata (param count + names; sample count inferred from file size)
        ofstream meta(meta_path);
        meta << dim << "\n";
        if (ecc)
            meta << "period amplitude offset phase eccentricity omega\n";
        else
            meta << "period amplitude offset phase\n";
        meta.close();

        int expected = Nsim / chain_thin;
        cout << "Chain output: " << bin_path
             << "  (thin=" << chain_thin
             << ", expected ~" << expected << " samples)\n";
    }

    // ---- LC prior ----
    vector<double> lc_periods, lc_powers;
    if (cfg.lc_prior) {
        if (cfg.lc_pgram_data.size() < 2)
            throw runtime_error("lc_pgram_data needs >=2 rows");
        lc_periods = cfg.lc_pgram_data[0];
        lc_powers  = cfg.lc_pgram_data[1];
        if (lc_periods.size() != lc_powers.size())
            throw runtime_error("lc_periods/lc_powers size mismatch");

        vector<size_t> idx(lc_periods.size());
        iota(idx.begin(), idx.end(), 0);
        sort(idx.begin(), idx.end(),
             [&](size_t a, size_t b){ return lc_periods[a]<lc_periods[b]; });
        vector<double> sp, sw;
        sp.reserve(idx.size()); sw.reserve(idx.size());
        for (size_t i : idx) { sp.push_back(lc_periods[i]);
                                sw.push_back(lc_powers[i]); }
        lc_periods = move(sp);
        lc_powers  = move(sw);
    }

// ---- parameter bounds in INTERNAL space ----
    vector<double> lo(dim), hi(dim);
    lo[iLPER] = log_min_p;
    hi[iLPER] = log_max_p;
    lo[iAMP]  = cfg.amp_min;
    hi[iAMP]  = cfg.amp_max > 0 ? cfg.amp_max : cfg.amp_lim;
    lo[iOFF]  = cfg.offset_min != 0 ? cfg.offset_min : -cfg.offset_lim;
    hi[iOFF]  = cfg.offset_max != 0 ? cfg.offset_max : cfg.offset_lim;
    lo[iPH]   = cfg.phase_min;
    hi[iPH]   = cfg.phase_max;
    if (ecc) {
        lo[iECC] = cfg.ecc_min;
        hi[iECC] = cfg.ecc_max;
        lo[iOMG] = cfg.omega_min;
        hi[iOMG] = cfg.omega_max;
    }

    // Print bounds
    cout << "Parameter bounds:\n"
         << "  Period:     " << cfg.min_period << " - " << cfg.max_period << " d\n"
         << "  Amplitude:  " << lo[iAMP] << " - " << hi[iAMP] << "\n"
         << "  Offset:     " << lo[iOFF] << " - " << hi[iOFF] << "\n"
         << "  Phase:      " << lo[iPH] << " - " << hi[iPH] << "\n";
    if (ecc) {
        cout << "  Ecc:        " << lo[iECC] << " - " << hi[iECC] << "\n"
             << "  Omega:      " << lo[iOMG] << " - " << hi[iOMG] << " deg\n";
    }

    auto to_real_period = [](double log_p) { return pow(10.0, log_p); };

    // ---- chi-squared ----
    auto chi2 = [&](const vector<double>& theta) -> double {
        double period = to_real_period(theta[iLPER]);
        double sum = 0.0;
        for (size_t i = 0; i < datapoints.size(); ++i) {
            double model = ecc
                ? rv_curve(samples[i], theta[iAMP], theta[iOFF],
                           period, theta[iPH], theta[iOMG], theta[iECC])
                : sinusoid(samples[i], theta[iAMP], period,
                           theta[iOFF], theta[iPH]);
            double r = (datapoints[i] - model) / datapoint_errors[i];
            sum += r * r;
        }
        return sum;
    };

    // ---- log-posterior ----
    auto logpost = [&](const vector<double>& theta) -> double {
        for (int i = 0; i < dim; ++i)
            if (theta[i] < lo[i] || theta[i] > hi[i])
                return -1e300;
        double lp = -0.5 * chi2(theta);
        lp += theta[iLPER] * log(10.0);
        if (cfg.lc_prior) {
            double pw = interp_prior(lc_periods, lc_powers,
                                      to_real_period(theta[iLPER]));
            if (pw <= 0.0) return -1e300;
            lp += log(pw);
        }
        return lp;
    };

    // ---- seed chains from periodogram peaks ----
    vector<double> seed_periods;
    if (!periodogram_x.empty() && !periodogram_y.empty()) {
        vector<size_t> pidx(periodogram_x.size());
        iota(pidx.begin(), pidx.end(), 0);
        sort(pidx.begin(), pidx.end(),
             [&](size_t a, size_t b){
                 return periodogram_y[a] > periodogram_y[b]; });
        for (size_t i = 0; i < pidx.size()
                 && (int)seed_periods.size() < Ntemp*2; ++i) {
            double p = periodogram_x[pidx[i]];
            if (p < cfg.min_period || p > cfg.max_period) continue;
            bool close = false;
            for (double sp : seed_periods)
                if (fabs(p - sp) / p < 0.01) { close = true; break; }
            if (!close) seed_periods.push_back(p);
        }
    }

    // ---- initial state ----
    double start_lp  = (cfg.period_0 > 0 && cfg.period_0 >= cfg.min_period
                         && cfg.period_0 <= cfg.max_period)
                        ? log10(cfg.period_0)
                        : (log_min_p + log_max_p) / 2.0;
    double start_amp = cfg.amp_0 > 0 ? cfg.amp_0 : ptp(datapoints)/2;
    if (start_amp > cfg.amp_lim) start_amp = cfg.amp_lim/2;
    double start_off = cfg.offset_0 != 0 ? cfg.offset_0
                       : vsum(datapoints)/(double)datapoints.size();
    if (start_off > cfg.offset_lim || start_off < -cfg.offset_lim) start_off=0;
    double start_ph  = cfg.phase_0;
    double start_ecc = (ecc && cfg.eccentricity_0 > 0 && cfg.eccentricity_0 < 1)
                       ? cfg.eccentricity_0 : 0.001;
    double start_omg = (ecc && cfg.omega_0 > 0 && cfg.omega_0 <= 360)
                       ? cfg.omega_0 : 180.0;

    vector<vector<double>> state(Ntemp, vector<double>(dim));
    vector<double>         state_lp(Ntemp);

    for (int t = 0; t < Ntemp; ++t) {
        state[t][iAMP] = start_amp;
        state[t][iOFF] = start_off;
        state[t][iPH]  = start_ph;
        if (ecc) {
            state[t][iECC] = start_ecc;
            state[t][iOMG] = start_omg;
        }
        if (t < (int)seed_periods.size())
            state[t][iLPER] = log10(seed_periods[t]);
        else
            state[t][iLPER] = start_lp;

        state_lp[t] = logpost(state[t]);
    }

    // ---- initial proposal covariance ----
    double sLP = cfg.period_step > 0
                 ? cfg.period_step * 0.4343 : 0.02;
    double sA  = cfg.amp_step    > 0 ? cfg.amp_step    : 0.5;
    double sO  = cfg.offset_step > 0 ? cfg.offset_step : 0.5;
    double sPh = cfg.phase_step  > 0 ? cfg.phase_step  : 0.05;
    double sE  = cfg.eccentricity_step > 0 ? cfg.eccentricity_step : 0.01;
    double sW  = cfg.omega_step  > 0 ? cfg.omega_step  : 5.0;

    vector<double> init_sigma(dim);
    init_sigma[iLPER] = sLP;
    init_sigma[iAMP]  = sA;
    init_sigma[iOFF]  = sO;
    init_sigma[iPH]   = sPh;
    if (ecc) {
        init_sigma[iECC] = sE;
        init_sigma[iOMG] = sW;
    }

    double sd = 2.38 * 2.38 / (double)dim;

    vector<vector<double>> C(dim, vector<double>(dim, 0.0));
    for (int i = 0; i < dim; ++i)
        C[i][i] = init_sigma[i] * init_sigma[i];

    auto do_cholesky = [&](const vector<vector<double>>& A)
            -> vector<vector<double>> {
        int d = (int)A.size();
        vector<vector<double>> LL(d, vector<double>(d, 0.0));
        for (int i = 0; i < d; ++i) {
            for (int j = 0; j <= i; ++j) {
                double s = 0.0;
                for (int k = 0; k < j; ++k)
                    s += LL[i][k] * LL[j][k];
                if (i == j)
                    LL[i][j] = sqrt(max(A[i][i] - s, 1e-30));
                else
                    LL[i][j] = (A[i][j] - s) / LL[j][j];
            }
        }
        return LL;
    };

    vector<vector<double>> L = do_cholesky(C);
    double global_scale = sd;

    // ---- Welford online mean/covariance ----
    long   welford_n = 0;
    vector<double> welford_mean(dim, 0.0);
    vector<vector<double>> welford_M2(dim, vector<double>(dim, 0.0));

    auto welford_update = [&](const vector<double>& x) {
        welford_n++;
        double n = (double)welford_n;
        vector<double> dx(dim);
        for (int i = 0; i < dim; ++i) {
            dx[i] = x[i] - welford_mean[i];
            welford_mean[i] += dx[i] / n;
        }
        for (int i = 0; i < dim; ++i) {
            double dx2_i = x[i] - welford_mean[i];
            for (int k = 0; k < dim; ++k) {
                welford_M2[i][k] += dx[i] * (x[k] - welford_mean[k]);
            }
        }
    };

    auto welford_covariance = [&]() -> vector<vector<double>> {
        vector<vector<double>> cov(dim, vector<double>(dim, 0.0));
        if (welford_n < 2) return cov;
        double n = (double)welford_n;
        for (int i = 0; i < dim; ++i)
            for (int k = 0; k < dim; ++k)
                cov[i][k] = welford_M2[i][k] / (n - 1.0);
        return cov;
    };

    // ---- per-chain RNG ----
    vector<mt19937> gens(Ntemp);
    {
        random_device rd;
        for (int t = 0; t < Ntemp; ++t)
            gens[t].seed(rd() + t * 12345);
    }
    mt19937 gen_main(random_device{}());
    uniform_real_distribution<> unif01(0.0, 1.0);

    vector<int> chain_accepted(Ntemp, 0);
    vector<int> chain_tried(Ntemp, 0);

    // ---- gnuplot ----
    unique_ptr<Gnuplot> gp;
    if (!cfg.noplot) {
        gp = make_unique<Gnuplot>();
        *gp << "set term qt 0 persist\n"
            << "set title 'Realtime RV Curve'\n";
    }

    // ---- main loop ----
    auto t0 = chrono::high_resolution_clock::now();
    const int logfreq = 100000;
    int n_swaps_accepted = 0, n_swaps_tried = 0;
    int adapt_count = 0;

    for (int j = 0; j < Ntot; ++j) {

        // ---- progress / plotting ----
        if (j > 0 && j % logfreq == 0) {
            auto now = chrono::high_resolution_clock::now();
            double ms = chrono::duration<double,milli>(now-t0).count();
            double t1_rate = chain_tried[0] > 0
                ? 100.0 * chain_accepted[0] / chain_tried[0] : 0;
            double swap_rate = n_swaps_tried > 0
                ? 100.0 * n_swaps_accepted / n_swaps_tried : 0;

            cout << "\r" << fixed << setprecision(1)
                 << "Progress: " << setw(5) << (100.0*j/Ntot) << "%"
                 << "  T1 Accept: " << setw(5) << t1_rate << "%"
                 << "  Swap: " << setw(5) << swap_rate << "%"
                 << "  Scale: " << scientific << setprecision(2) << global_scale
                 << "  Speed: " << fixed << setprecision(3) << ms/j << " ms/it"
                 << "  P(T=1)=" << setprecision(4) << to_real_period(state[0][iLPER]) << "d"
                 << "  Chain=" << chain_count
                 << "    " << flush;

            for (int t = 0; t < Ntemp; ++t) {
                chain_accepted[t] = 0;
                chain_tried[t] = 0;
            }

            if (gp) {
                auto& s = state[0];
                double per = to_real_period(s[iLPER]);
                const int Npl = 1000;
                vector<pair<double,double>> curve(Npl);
                auto xs = linspace(0.0, 1.0, Npl);
                for (int h = 0; h < Npl; ++h) {
                    double tt = (xs[h] + s[iPH]) * per;
                    double rv = ecc
                        ? rv_curve(tt, s[iAMP], s[iOFF], per, s[iPH],
                                   s[iOMG], s[iECC])
                        : sinusoid(tt, s[iAMP], per, s[iOFF], s[iPH]);
                    curve[h] = {xs[h], rv};
                }
                vector<tuple<double,double,double>> pts;
                for (size_t i = 0; i < samples.size(); ++i) {
                    double phi = fmod(samples[i]/per - s[iPH], 1.0);
                    if (phi < 0) phi += 1.0;
                    pts.emplace_back(phi, datapoints[i], datapoint_errors[i]);
                }
                *gp << "plot '-' w lines title 'Model (T=1)', "
                       "'-' w yerrorbars title 'Data'\n";
                gp->send1d(curve);
                gp->send1d(pts);
            }
        }

        // ============================================================
        //  1. Parallel MH step
        // ============================================================
        const auto L_snap = L;
        const double scale_snap = global_scale;

        #pragma omp parallel for schedule(static) num_threads(Ntemp)
        for (int t = 0; t < Ntemp; ++t) {
            normal_distribution<>        norm01(0.0, 1.0);
            uniform_real_distribution<>  u01(0.0, 1.0);

            vector<double> z(dim), prop(dim);
            for (int i = 0; i < dim; ++i) z[i] = norm01(gens[t]);

            vector<double> Lz(dim, 0.0);
            for (int i = 0; i < dim; ++i)
                for (int k = 0; k <= i; ++k)
                    Lz[i] += L_snap[i][k] * z[k];

            for (int i = 0; i < dim; ++i)
                prop[i] = state[t][i] + sqrt(scale_snap) * Lz[i];

            // Wrap periodic parameters
            while (prop[iPH] >  0.5) prop[iPH] -= 1.0;
            while (prop[iPH] < -0.5) prop[iPH] += 1.0;
            if (ecc) {
                while (prop[iOMG] <   0.0) prop[iOMG] += 360.0;
                while (prop[iOMG] > 360.0) prop[iOMG] -= 360.0;
            }

            // Reflect non-wrapped parameters at boundaries
            for (int i = 0; i < dim; ++i) {
                if (i == iPH) continue;
                if (ecc && i == iOMG) continue;
                while (prop[i] < lo[i] || prop[i] > hi[i]) {
                    if (prop[i] < lo[i]) prop[i] = lo[i] + (lo[i] - prop[i]);
                    if (prop[i] > hi[i]) prop[i] = hi[i] - (prop[i] - hi[i]);
                }
            }

            double prop_lp = logpost(prop);
            double log_alpha = (prop_lp - state_lp[t]) / temperatures[t];

            if (log(u01(gens[t])) < log_alpha) {
                state[t]    = prop;
                state_lp[t] = prop_lp;
                #pragma omp atomic
                chain_accepted[t]++;
            }
            #pragma omp atomic
            chain_tried[t]++;
        }

        // ============================================================
        //  2. Parallel tempering swaps
        // ============================================================
        if (Ntemp > 1 && j % cfg.swap_interval == 0) {
            int parity = (j / cfg.swap_interval) % 2;
            for (int t1 = parity; t1 + 1 < Ntemp; t1 += 2) {
                int t2 = t1 + 1;
                ++n_swaps_tried;
                double log_swap = (state_lp[t1] - state_lp[t2])
                    * (1.0/temperatures[t2] - 1.0/temperatures[t1]);
                if (log(unif01(gen_main)) < log_swap) {
                    swap(state[t1],    state[t2]);
                    swap(state_lp[t1], state_lp[t2]);
                    ++n_swaps_accepted;
                }
            }
        }

        // ============================================================
        //  3. Adaptive Metropolis
        // ============================================================
        if (j >= cfg.adapt_start) {
            welford_update(state[0]);

            if (j % cfg.adapt_interval == 0 && welford_n > 2 * dim) {
                ++adapt_count;
                auto emp_cov = welford_covariance();
                double eps = 0.01;
                for (int i = 0; i < dim; ++i)
                    for (int k = 0; k < dim; ++k)
                        C[i][k] = (1.0-eps) * emp_cov[i][k]
                                  + (i==k ? eps * init_sigma[i]*init_sigma[i]
                                          : 0.0);
                L = do_cholesky(C);

                double curr_rate = chain_tried[0] > 0
                    ? (double)chain_accepted[0] / chain_tried[0]
                    : cfg.target_accept;
                double gamma_n = 1.0 / pow((double)adapt_count, 0.6);
                double log_s = log(global_scale);
                log_s += gamma_n * (curr_rate - cfg.target_accept);
                global_scale = exp(log_s);
                global_scale = max(cfg.adapt_scale_min,
                               min(cfg.adapt_scale_max, global_scale));
            }
        }

        // ============================================================
        //  4. Write T=1 chain sample (post burn-in, thinned)
        // ============================================================
        if (j < Nburn) continue;
        if ((j - Nburn) % chain_thin != 0) continue;

        if (chain_file) {
            const auto& s = state[0];
            double row[6];
            row[0] = to_real_period(s[iLPER]);  // period in days
            row[1] = s[iAMP];                   // amplitude
            row[2] = s[iOFF];                   // offset
            row[3] = s[iPH];                    // phase [-0.5, 0.5]
            if (ecc) {
                row[4] = s[iECC];               // eccentricity
                row[5] = s[iOMG];               // omega in degrees
            }
            fwrite(row, sizeof(double), dim, chain_file);
            ++chain_count;
        }
    }

    // ---- close chain file ----
    if (chain_file) {
        fclose(chain_file);
        cout << "\nChain: " << chain_count << " samples written to "
             << cfg.chain_output_dir << "chain.bin"
             << " (" << (chain_count * dim * 8) / (1024*1024) << " MB)\n";
    }

    // ---- summary ----
    cout << "\n\n=== MCMC Summary ===\n"
         << "Final proposal scale: " << scientific << global_scale << "\n"
         << "Swap acceptance: "
         << (n_swaps_tried > 0 ? 100.0*n_swaps_accepted/n_swaps_tried : 0)
         << "%\n"
         << "Chain samples: " << chain_count
         << " (thin=" << chain_thin << ")\n";

    const char* names[] = {"log(P)", "  amp ", "offset", " phase",
                           "  ecc ", " omega"};
    cout << "\nLearned proposal correlations:\n         ";
    for (int i = 0; i < dim; ++i) cout << setw(8) << names[i];
    cout << "\n";
    auto cov = welford_covariance();
    for (int i = 0; i < dim; ++i) {
        cout << names[i] << "  ";
        for (int k = 0; k < dim; ++k) {
            double denom = sqrt(cov[i][i] * cov[k][k]);
            double corr = denom > 0 ? cov[i][k] / denom : 0;
            cout << fixed << setprecision(3) << setw(8) << corr;
        }
        cout << "\n";
    }
    cout << "\n";

    if (gp) {
        *gp << "exit\n";
        gp.reset();
        system("pkill -f gnuplot_qt");
    }
}


// ===================================================================
//  Grid-based orbit prediction (unchanged)
// ===================================================================
void Star::calculate_orbit_prediction(int Nx, int Ny,
                                      double amp_lim, double offset_lim) {
    if (periodogram_x.empty() || periodogram_y.empty()) return;

    period_amp_histogram    = vector<vector<double>>(Nx, vector<double>(Ny, 0.));
    period_offset_histogram = vector<vector<double>>(Nx, vector<double>(Ny, 0.));
    period_phase_histogram  = vector<vector<double>>(Nx, vector<double>(Ny, 0.));

    double minP = *min_element(periodogram_x.begin(), periodogram_x.end());
    double maxP = *max_element(periodogram_x.begin(), periodogram_x.end());

    vector<double> period_edges = linspace(log10(minP), log10(maxP), Nx + 1);
    transform(period_edges.begin(), period_edges.end(), period_edges.begin(),
              [](double x) { return pow(10, x); });

    vector<double> amp_edges    = linspace(0, amp_lim, Ny + 1);
    vector<double> phase_edges  = linspace(0, 1, Ny + 1);
    vector<double> offset_edges = linspace(-offset_lim, offset_lim, Ny + 1);

    transform(periodogram_y.begin(), periodogram_y.end(), periodogram_y.begin(),
              [](double x) { return x < 0.0 ? 0.0 : x; });
    periodogram_y = vdivide(periodogram_y, vsum(periodogram_y));

    vector<double> period_chi_sums(Nx, 0.0);
    int progress = 0;
    auto t0 = chrono::high_resolution_clock::now();

    #pragma omp parallel for schedule(static)
    for (int j = 0; j < (int)periodogram_x.size(); ++j) {
        #pragma omp atomic
        progress++;
        if (progress % 1000 == 0) {
            #pragma omp critical
            {
                auto now = chrono::high_resolution_clock::now();
                double ms = chrono::duration<double, milli>(now - t0).count()
                            / progress;
                cout << "Progress: " << progress << " / "
                     << periodogram_x.size()
                     << "  (" << ms << " ms/loop)\n";
            }
        }

        auto res = sinusoidMonteCarlo(
            samples, datapoints, datapoint_errors,
            periodogram_x[j], 25000,
            ptp(datapoints) / 2.0, 0.5,
            vsum(datapoints) / datapoints.size());

        int pbin = (int)(lower_bound(period_edges.begin(), period_edges.end(),
                                     periodogram_x[j])
                         - period_edges.begin()) - 1;

        const auto& amps    = get<0>(res);
        const auto& phases  = get<1>(res);
        const auto& offsets = get<2>(res);
        const auto& chisqs  = get<3>(res);

        for (size_t i = 0; i < amps.size(); ++i) {
            int ab  = (int)(lower_bound(amp_edges.begin(), amp_edges.end(),
                                        amps[i]) - amp_edges.begin()) - 1;
            int phb = (int)(lower_bound(phase_edges.begin(), phase_edges.end(),
                                        fmod(phases[i], 1.0))
                            - phase_edges.begin()) - 1;
            int ob  = (int)(lower_bound(offset_edges.begin(), offset_edges.end(),
                                        offsets[i]) - offset_edges.begin()) - 1;

            if (pbin >= 0 && pbin < Nx) {
                if (ab  >= 0 && ab  < Ny) {
                    #pragma omp atomic
                    period_amp_histogram[pbin][ab] +=
                        periodogram_y[j] / chisqs[i];
                }
                if (phb >= 0 && phb < Ny) {
                    #pragma omp atomic
                    period_phase_histogram[pbin][phb] +=
                        periodogram_y[j] / chisqs[i];
                }
                if (ob  >= 0 && ob  < Ny) {
                    #pragma omp atomic
                    period_offset_histogram[pbin][ob] +=
                        periodogram_y[j] / chisqs[i];
                }
            }
        }
        if (pbin >= 0 && pbin < Nx) {
            #pragma omp atomic
            period_chi_sums[pbin] += vsum(chisqs);
        }
    }

    for (int i = 0; i < Nx; ++i) {
        if (period_chi_sums[i] != 0.0) {
            period_amp_histogram[i]    = vdivide(period_amp_histogram[i],
                                                  period_chi_sums[i]);
            period_offset_histogram[i] = vdivide(period_offset_histogram[i],
                                                  period_chi_sums[i]);
            period_phase_histogram[i]  = vdivide(period_phase_histogram[i],
                                                  period_chi_sums[i]);
        }
    }
    normalize_histogram(period_amp_histogram);
    normalize_histogram(period_offset_histogram);
    normalize_histogram(period_phase_histogram);
}


// ===================================================================
//  Binary MCMC (unchanged)
// ===================================================================
void Star::run_rv_mcmc_binary(
        int Nx, int Ny, double amp_lim, double offset_lim,
        int N_sim, double min_p, double max_p,
        double min_p_two, double max_p_two,
        double period_step, double amp_step,
        double offset_step, double phase_step,
        double period_0, double amp_0, double offset_0,
        double phase_0, double period_0_two,
        double amp_0_two, double phase_0_two,
        int N_burn_in) {

    period_amp_histogram     = vector<vector<double>>(Nx, vector<double>(Ny, 0.));
    period_amp_histogram_two = vector<vector<double>>(Nx, vector<double>(Ny, 0.));

    vector<double> period_edges = linspace(log10(min_p), log10(max_p), Nx + 1);
    transform(period_edges.begin(), period_edges.end(), period_edges.begin(),
              [](double x) { return pow(10, x); });

    vector<double> period_edges_two = linspace(log10(min_p_two), log10(max_p_two), Nx + 1);
    transform(period_edges_two.begin(), period_edges_two.end(), period_edges_two.begin(),
              [](double x) { return pow(10, x); });

    vector<double> amp_edges    = linspace(0, amp_lim, Ny + 1);
    vector<double> phase_edges  = linspace(0, 1, Ny + 1);
    vector<double> offset_edges = linspace(-offset_lim, offset_lim, Ny + 1);

    if (amp_step    == 0.0) amp_step    = 0.5;
    if (period_step == 0.0) period_step = 0.01;
    if (phase_step  == 0.0) phase_step  = 0.1618;
    if (offset_step == 0.0) offset_step = 0.5;

    if (amp_0    == 0.0)            amp_0    = ptp(datapoints) / 2;
    if (amp_0    >  amp_lim)        amp_0    = amp_lim / 2;
    if (period_0 == 0.0)            period_0 = pow(10, (log10(min_p) + log10(max_p)) / 2);
    if (period_0 > max_p || period_0 < min_p) period_0 = (max_p + min_p) / 2;
    if (phase_0  == 0.0)            phase_0  = 0.5;
    if (offset_0 == 0.0)            offset_0 = vsum(datapoints) / datapoints.size();
    if (offset_0 > offset_lim || offset_0 < -offset_lim) offset_0 = 0;

    if (amp_0_two    == 0.0)                              amp_0_two    = ptp(datapoints) / 20;
    if (period_0_two == 0.0)                              period_0_two = pow(10, (log10(min_p_two) + log10(max_p_two)) / 2);
    if (period_0_two > max_p_two || period_0_two < min_p_two) period_0_two = (max_p_two + min_p_two) / 2;
    if (phase_0_two  == 0.0)                              phase_0_two  = 0.5;

    random_device rd;
    mt19937 gen(rd());
    normal_distribution<>       dist(0.0, 1.0);
    uniform_real_distribution<> dist_uni(0.0, 1.0);

    auto chiSq = [&](double amp, double period, double phase, double offset,
                      double amp2, double period2, double phase2) {
        double c = 0.0;
        for (size_t i = 0; i < datapoints.size(); ++i) {
            double m = rv_curve_binary(samples[i], amp, offset, period, phase,
                                       amp2, period2, phase2);
            double r = (datapoints[i] - m) / datapoint_errors[i];
            c += r * r;
        }
        return c;
    };

    int n_accepted = 0;
    int print_every = 100000;
    auto start = chrono::high_resolution_clock::now();

    for (int j = 0; j < N_sim + N_burn_in; ++j) {
        if (j % print_every == 0) {
            auto now = chrono::high_resolution_clock::now();
            double ms = chrono::duration<double, milli>(now - start).count();
            cout << "\r" << "Progress: " << setw(6) << right
                 << (round(1000.0 * j / (N_sim + N_burn_in)) / 10)
                 << "%, Acceptance: " << setw(6) << right
                 << (round(100000.0 * n_accepted / print_every) / 1000)
                 << "%, Speed: " << setw(10) << right
                 << (j > 0 ? ms / j : 0) << " ms/it" << flush;
            n_accepted = 0;
        }

        double chi2_cur = chiSq(amp_0, period_0, phase_0, offset_0,
                                 amp_0_two, period_0_two, phase_0_two);

        double ap = amp_0      + amp_step    * dist(gen);
        double pp = period_0   * (1 + period_step * dist(gen));
        double hp = fmod(phase_0 + phase_step * dist(gen), 1.0);
        double op = offset_0   + offset_step * dist(gen);
        double a2 = amp_0_two  + amp_step    * dist(gen);
        double p2 = period_0_two * (1 + period_step * dist(gen));
        double h2 = fmod(phase_0_two + phase_step * dist(gen), 1.0);

        while (ap > amp_lim || op > offset_lim || op < -offset_lim ||
               pp < min_p   || pp > max_p      || hp > 1 || hp < 0 ||
               a2 > amp_lim || p2 < min_p_two  || p2 > max_p_two ||
               h2 > 1       || h2 < 0) {
            ap = amp_0      + amp_step    * dist(gen);
            pp = period_0   * (1 + period_step * dist(gen));
            hp = fmod(phase_0 + phase_step * dist(gen), 1.0);
            op = offset_0   + offset_step * dist(gen);
            a2 = amp_0_two  + amp_step    * dist(gen);
            p2 = period_0_two * (1 + period_step * dist(gen));
            h2 = fmod(phase_0_two + phase_step * dist(gen), 1.0);
        }

        double chi2_prop = chiSq(ap, pp, hp, op, a2, p2, h2);

        if (chi2_prop < chi2_cur ||
            exp(-(chi2_prop - chi2_cur) / 2) > dist_uni(gen)) {
            n_accepted++;
            amp_0 = ap; period_0 = pp; phase_0 = hp; offset_0 = op;
            amp_0_two = a2; period_0_two = p2; phase_0_two = h2;
        }

        if (j >= N_burn_in) {
            int ab  = (int)(lower_bound(amp_edges.begin(), amp_edges.end(), amp_0)
                            - amp_edges.begin()) - 1;
            int pb  = (int)(lower_bound(period_edges.begin(), period_edges.end(), period_0)
                            - period_edges.begin()) - 1;
            int a2b = (int)(lower_bound(amp_edges.begin(), amp_edges.end(), amp_0_two)
                            - amp_edges.begin()) - 1;
            int p2b = (int)(lower_bound(period_edges_two.begin(), period_edges_two.end(), period_0_two)
                            - period_edges_two.begin()) - 1;
            if (pb >= 0 && pb < Nx && ab >= 0 && ab < Ny)
                period_amp_histogram[pb][ab] += 1;
            if (p2b >= 0 && p2b < Nx && a2b >= 0 && a2b < Ny)
                period_amp_histogram_two[p2b][a2b] += 1;
        }
    }
    cout << "\n";
}


// ===================================================================
//  process_star (unchanged)
// ===================================================================
void Star::process_star(int index, double forced_min_p,
                        double forced_max_p, int forced_N) {
    vector<double> opt = genOptimalPeriodogramSamples(
        samples, 20, forced_min_p, forced_max_p, forced_N);
    if (opt[2] < 100000) {
        opt[1] *= opt[2] / 100000.0;
        opt[2]  = 100000;
    }
    cout << "MC Simulation from "
         << 1.0 / (opt[0] + opt[1] * opt[2]) << "d to "
         << 1.0 / opt[0] << "d using " << opt[2] << " samples.\n";

    periodogram_y = gls_fast(samples, datapoints,
                             vector<double>(Npoints, 10),
                             opt[0], opt[1], (int)round(opt[2]));

    vector<double> px = linspace(opt[0],
                                 opt[0] + opt[2] * opt[1],
                                 (int)ceil(opt[2]));
    periodogram_x = invert(px);

    saveCSV("out/" + to_string(index) + ".csv",
            {periodogram_x, periodogram_y});
    saveCSV("out/rvs" + to_string(index) + ".csv",
            {samples, datapoints, datapoint_errors});

    cout << "Period:    " << period    << "\n"
         << "Amplitude: " << amplitude << "\n"
         << "Offset:    " << offset    << "\n"
         << "Phase:     " << phase     << "\n";

    calculate_orbit_prediction(10000, 1000, 500., 500.);

    saveCSV("out/pamp"    + to_string(index) + ".csv", period_amp_histogram);
    saveCSV("out/pphase"  + to_string(index) + ".csv", period_phase_histogram);
    saveCSV("out/poffset" + to_string(index) + ".csv", period_offset_histogram);
}


// ===================================================================
//  sort_samples (unchanged)
// ===================================================================
void Star::sort_samples() {
    vector<size_t> idx(samples.size());
    iota(idx.begin(), idx.end(), 0);
    sort(idx.begin(), idx.end(),
         [&](size_t a, size_t b) { return samples[a] < samples[b]; });

    vector<double> ss(samples.size()), sd(samples.size()), se(samples.size());
    for (size_t i = 0; i < idx.size(); ++i) {
        ss[i] = samples[idx[i]];
        sd[i] = datapoints[idx[i]];
        se[i] = datapoint_errors[idx[i]];
    }
    samples          = move(ss);
    datapoints       = move(sd);
    datapoint_errors = move(se);
}