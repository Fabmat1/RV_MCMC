//
//  rv_mcmc.cpp  –  Unified entry point for circular and eccentric
//                  MCMC fitting of radial-velocity curves.
//
//  Replaces the former amp_mcmc.cpp and amp_mcmc_eccentric.cpp.
//
//  Usage:
//      rv_mcmc <gaia-id> [options]               # circular fit
//      rv_mcmc <gaia-id> --eccentric [options]    # eccentric fit
//

#include "models.h"
#include "vector_operations.h"
#include "lomb_scargle_periodogram.h"
#include "file_io.h"
#include "maths.h"

#include <boost/program_options.hpp>

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <unistd.h>   // chdir, getcwd

namespace po = boost::program_options;
using namespace std;

// ---------------------------------------------------------------
//  Small helper – lowercase a string (used for LC-prior sources)
// ---------------------------------------------------------------
static string str_lower(string s) {
    transform(s.begin(), s.end(), s.begin(),
              [](unsigned char c) { return tolower(c); });
    return s;
}

// ---------------------------------------------------------------
//  Fetch / compute the light-curve periodogram prior.
//  Returns a 2-D vector [periods, powers] (empty on failure).
// ---------------------------------------------------------------
static vector<vector<double>> fetch_lc_prior(
        const string& gaia_id,
        const string& lc_prior_source,
        double min_p, double max_p,
        bool   ellipsoidal) {

    const char* lc_dir = getenv("LIGHTCURVEQUERY_DIR");
    if (!lc_dir) {
        cerr << "ERROR: set LIGHTCURVEQUERY_DIR to the directory "
                "containing lightcurvequery.py\n";
        return {};
    }
    string script_dir(lc_dir);

    // Save and change directory
    char cwd[PATH_MAX];
    if (!getcwd(cwd, sizeof(cwd))) {
        cerr << "ERROR: getcwd: " << strerror(errno) << "\n";
        return {};
    }
    if (chdir(script_dir.c_str()) != 0) {
        cerr << "ERROR: chdir('" << script_dir << "'): "
             << strerror(errno) << "\n";
        return {};
    }

    // Parse requested sources
    vector<string> requested;
    {
        istringstream ss(lc_prior_source);
        string tok;
        while (getline(ss, tok, ',')) {
            tok.erase(0, tok.find_first_not_of(" \t\n\r"));
            tok.erase(tok.find_last_not_of(" \t\n\r") + 1);
            transform(tok.begin(), tok.end(), tok.begin(),
                      [](unsigned char c) { return toupper(c); });
            requested.push_back(tok);
        }
    }
    for (auto& r : requested)
        cerr << "Requested LC source: '" << r << "'\n";

    double lc_lo = ellipsoidal ? min_p / 2.0 : min_p;
    double lc_hi = ellipsoidal ? max_p / 2.0 : max_p;
    if (ellipsoidal)
        cout << "Ellipsoidal mode: LC periodogram "
             << lc_lo << "d – " << lc_hi << "d "
             << "(half of " << min_p << "d – " << max_p << "d)\n";

    ostringstream cmd;
    cmd << "python lightcurvequery.py " << gaia_id
        << " --min-p " << lc_lo
        << " --max-p " << lc_hi;

    const vector<string> all_src = {"TESS","ATLAS","ZTF","GAIA","BG"};
    bool use_all = (requested.size() == 1 && requested[0] == "ALL");
    if (!use_all)
        for (auto& s : all_src)
            if (find(requested.begin(), requested.end(), s) == requested.end())
                cmd << " --skip-" << str_lower(s);

    string cmd_str = cmd.str();
    cout << "Running LC prior: " << cmd_str << "\n";
    int ret = system(cmd_str.c_str());

    // Restore working directory before checking result
// Restore working directory before checking result
    if (chdir(cwd) != 0) {
        cerr << "ERROR: failed to restore directory '" << cwd
             << "': " << strerror(errno) << "\n";
        return {};
    }
    if (ret != 0) {
        cerr << "ERROR: lightcurvequery.py exited with code " << ret << "\n";
        return {};
    }

    string pgram_file = script_dir + "/periodograms/" + gaia_id
                        + "/multiplied_pgram.txt";
    auto data = readCSV(pgram_file, true);

    // In ellipsoidal mode, double the periods
    if (ellipsoidal && !data.empty() && !data[0].empty()) {
        cout << "Ellipsoidal mode: scaling LC periods by 2x\n";
        for (auto& v : data[0]) v *= 2.0;
    }
    return data;
}

// ---------------------------------------------------------------
//  Save all corner-plot histograms that are non-empty
// ---------------------------------------------------------------
static void save_histograms(const string& folder, Star& star, bool eccentric) {
    // 6 base-parameter combinations
    saveCSV(folder + "period_vs_amplitude.csv", star.period_amp_histogram);
    saveCSV(folder + "period_vs_offset.csv",    star.period_offset_histogram);
    saveCSV(folder + "period_vs_phase.csv",     star.period_phase_histogram);
    saveCSV(folder + "amplitude_vs_offset.csv", star.amp_offset_histogram);
    saveCSV(folder + "amplitude_vs_phase.csv",  star.amp_phase_histogram);
    saveCSV(folder + "offset_vs_phase.csv",     star.offset_phase_histogram);

    if (eccentric) {
        saveCSV(folder + "period_vs_eccentricity.csv",    star.period_ecc_histogram);
        saveCSV(folder + "period_vs_omega.csv",           star.period_omega_histogram);
        saveCSV(folder + "amplitude_vs_eccentricity.csv", star.amp_ecc_histogram);
        saveCSV(folder + "amplitude_vs_omega.csv",        star.amp_omega_histogram);
        saveCSV(folder + "offset_vs_eccentricity.csv",    star.offset_ecc_histogram);
        saveCSV(folder + "offset_vs_omega.csv",           star.offset_omega_histogram);
        saveCSV(folder + "phase_vs_eccentricity.csv",     star.phase_ecc_histogram);
        saveCSV(folder + "phase_vs_omega.csv",            star.phase_omega_histogram);
        saveCSV(folder + "eccentricity_vs_omega.csv",     star.ecc_omega_histogram);
    }
}

// ---------------------------------------------------------------
//  Write a human-readable metadata / parameter file
// ---------------------------------------------------------------
static void save_metadata(const string& folder, const string& gaia_id,
                          const MCMCConfig& cfg,
                          const string& lc_prior_source, bool ellipsoidal) {
    ofstream f(folder + "mcmc_params.txt");
    f << "MCMC Run Parameters for " << gaia_id << "\n"
      << "==================================\n"
      << "Mode: " << (cfg.eccentric ? "Eccentric" : "Circular") << "\n"
      << "Period range: " << cfg.min_period << " - " << cfg.max_period << " days\n"
      << "Number of samples: " << cfg.n_samples << "\n"
      << "Number of burn-in samples: " << cfg.n_burn_in << "\n"
      << "Period bins x Param bins: " << cfg.n_period_bins << " x " << cfg.n_param_bins << "\n"
      << "Amplitude limit: " << cfg.amp_lim << "\n"
      << "Offset limit: " << cfg.offset_lim << "\n"
      << "\nStep sizes:\n"
      << "  Period:        " << cfg.period_step << "\n"
      << "  Amplitude:     " << cfg.amp_step << "\n"
      << "  Offset:        " << cfg.offset_step << "\n"
      << "  Phase:         " << cfg.phase_step << "\n";
    if (cfg.eccentric) {
        f << "  Eccentricity:  " << cfg.eccentricity_step << "\n"
          << "  Omega:         " << cfg.omega_step << " deg\n";
    }
    f << "\nInitial values:\n"
      << "  Period:        " << cfg.period_0 << "\n"
      << "  Amplitude:     " << cfg.amp_0 << "\n"
      << "  Offset:        " << cfg.offset_0 << "\n"
      << "  Phase:         " << cfg.phase_0 << "\n";
    if (cfg.eccentric) {
        f << "  Eccentricity:  " << cfg.eccentricity_0 << "\n"
          << "  Omega:         " << cfg.omega_0 << " deg\n";
    }
    f << "\nLC Prior: " << (cfg.lc_prior ? "Enabled" : "Disabled") << "\n";
    if (cfg.lc_prior) {
        f << "LC Prior sources: " << lc_prior_source << "\n"
          << "Ellipsoidal mode: " << (ellipsoidal ? "Yes" : "No") << "\n";
    }
    f.close();
}

// ===============================================================
//  main
// ===============================================================
int main(int argc, char* argv[]) {

    // ---------- declare all CLI variables ----------
    string gaia_id;
    string out_path      = "out/";
    string input_path;
    string lc_prior_source = "ALL";

    MCMCConfig cfg;
    bool ellipsoidal = false;

    // -------- define options --------
    try {
        po::options_description desc("Allowed options");
        desc.add_options()
            ("help",
                "produce help message")
            ("gaia-id",
                po::value<string>(&gaia_id)->required(),
                "GAIA ID (positional)")

            // --- mode ---
            ("eccentric",
                po::bool_switch(&cfg.eccentric),
                "Use Keplerian (eccentric) RV model instead of sinusoidal")

            // --- period range / MCMC size ---
            ("l",  po::value<double>(&cfg.min_period),
                "Low period bound (days)")
            ("h",  po::value<double>(&cfg.max_period),
                "High period bound (days)")
            ("n",  po::value<int>(&cfg.n_samples),
                "Number of post-burn-in MCMC samples")
            ("r",  po::value<int>(&cfg.n_period_bins),
                "Number of period bins")
            ("n-burn-in", po::value<int>(&cfg.n_burn_in),
                "Number of burn-in samples (default 1000000)")

            // --- limits ---
            ("amp-lim",    po::value<double>(&cfg.amp_lim),
                "Hard amplitude limit")
            ("offset-lim", po::value<double>(&cfg.offset_lim),
                "Hard offset limit")

            // --- step sizes ---
            ("period-step",        po::value<double>(&cfg.period_step),
                "Period proposal step (fractional)")
            ("amplitude-step",     po::value<double>(&cfg.amp_step),
                "Amplitude proposal step")
            ("offset-step",        po::value<double>(&cfg.offset_step),
                "Offset proposal step")
            ("phase-step",         po::value<double>(&cfg.phase_step),
                "Phase proposal step")
            ("eccentricity-step",  po::value<double>(&cfg.eccentricity_step),
                "Eccentricity proposal step (eccentric only)")
            ("omega-step",         po::value<double>(&cfg.omega_step),
                "Omega proposal step in degrees (eccentric only)")

            // --- initial values ---
            ("period-0",        po::value<double>(&cfg.period_0),
                "Period starting value")
            ("amplitude-0",     po::value<double>(&cfg.amp_0),
                "Amplitude starting value")
            ("offset-0",        po::value<double>(&cfg.offset_0),
                "Offset starting value")
            ("phase-0",         po::value<double>(&cfg.phase_0),
                "Phase starting value")
            ("eccentricity-0",  po::value<double>(&cfg.eccentricity_0),
                "Eccentricity starting value (eccentric only)")
            ("omega-0",         po::value<double>(&cfg.omega_0),
                "Omega starting value in degrees (eccentric only)")

            // --- I/O ---
            ("out-dir", po::value<string>(&out_path),
                "Output directory (default: out/)")
            ("input",   po::value<string>(&input_path),
                "Input CSV file path")

            // --- flags ---
            ("no-plot",   po::bool_switch(&cfg.noplot),
                "Disable real-time gnuplot display")
            ("lc-prior",  po::bool_switch(&cfg.lc_prior),
                "Enable light-curve periodogram prior")
            ("lc-prior-source", po::value<string>(&lc_prior_source),
                "LC prior sources: ALL, TESS, ATLAS, ZTF, GAIA, BG "
                "(comma-separated)")
            ("ellipsoidal", po::bool_switch(&ellipsoidal),
                "With --lc-prior: compute LC periodogram at half the period "
                "range then scale back (for ellipsoidal variations)")
            // Add these to the options description:
            ("n-temperatures",  po::value<int>(&cfg.n_temperatures),
                "Number of parallel tempering chains (default 8)")
            ("max-temperature", po::value<double>(&cfg.max_temperature),
                "Highest temperature in ladder (default 100)")
            ("swap-interval",   po::value<int>(&cfg.swap_interval),
                "Attempt chain swap every N steps (default 20)")
            ("target-accept",   po::value<double>(&cfg.target_accept),
                "Target acceptance rate for adaptation (default 0.234)")
        ;

        po::positional_options_description pos;
        pos.add("gaia-id", 1);

        po::variables_map vm;
        po::store(po::command_line_parser(argc, argv)
                      .options(desc).positional(pos).run(), vm);

        if (vm.count("help")) {
            cout << "Usage: " << argv[0] << " <gaia-id> [options]\n\n"
                 << desc << "\n"
                 << "Examples:\n"
                 << "  " << argv[0] << " 1234567890 -l 0.1 -h 100 -n 5000000 -r 100\n"
                 << "  " << argv[0] << " 1234567890 --eccentric -l 0.1 -h 100 -n 5000000 -r 100 --lc-prior\n";
            return 0;
        }
        po::notify(vm);

        // Default input path
        if (input_path.empty()) {
#ifdef __linux__
            input_path = string(getenv("HOME"))
                         + "/Projects/RVVD_refit_2025/output/"
                         + gaia_id + "/RV_variation.csv";
#else
            input_path = "/mnt/c/Users/fabia/PycharmProjects/RVVD_plus/output/"
                         + gaia_id + "/RV_variation.csv";
#endif
        }

    } catch (const po::error& e) {
        cerr << "Error: " << e.what() << "\n"
             << "Use --help for usage information.\n";
        return 1;
    }

    if (ellipsoidal && !cfg.lc_prior)
        cerr << "Warning: --ellipsoidal has no effect without --lc-prior\n";

    // ========== load data ==========
    Star star;
    auto rvs = readCSV(input_path, true);

    star.samples          = vadd(rvs[3], -get_min(rvs[3]));
    star.datapoints       = rvs[0];
    star.datapoint_errors = rvs[1];
    star.Npoints          = (int)star.samples.size();

    // ========== periodogram ==========
    auto opt = genOptimalPeriodogramSamples(star.samples, 20,
                                            cfg.min_period, cfg.max_period);
    if (opt[2] < 1000)
        opt = genOptimalPeriodogramSamples(star.samples, 20,
                                            cfg.min_period, cfg.max_period, 1000);

    cout << "Pgram params: " << opt[0] << " " << opt[1] << " "
         << (int)round(opt[2]) << "\n";
    cout << "Calculating periodogram from " << cfg.min_period << "d to "
         << cfg.max_period << "d using " << opt[2] << " samples.\n";

    star.periodogram_y = gls_fast(
        star.samples, star.datapoints,
        vector<double>(star.Npoints, 10),
        opt[0], opt[1], (int)round(opt[2]));

    vector<double> pgram_x = linspace(opt[0],
                                       opt[0] + opt[2] * opt[1],
                                       (int)ceil(opt[2]));
    star.periodogram_x = invert(pgram_x);

    // ========== output directory ==========
    string folder = out_path + gaia_id + "/";
    filesystem::create_directories(folder);

    cout << "Saving periodogram...\n";
    saveCSV(folder + "pgram.csv", {star.periodogram_x, star.periodogram_y});

    // ========== LC prior ==========
    if (cfg.lc_prior) {
        cfg.lc_pgram_data = fetch_lc_prior(gaia_id, lc_prior_source,
                                            cfg.min_period, cfg.max_period,
                                            ellipsoidal);
        if (cfg.lc_pgram_data.empty()) {
            cerr << "ERROR: failed to obtain LC prior – aborting.\n";
            return 1;
        }
    }

    // ========== run MCMC ==========
    cout << "\nRunning " << (cfg.eccentric ? "ECCENTRIC" : "CIRCULAR")
         << " MCMC\n"
         << "  Period range : " << cfg.min_period << " – "
         << cfg.max_period << " d\n"
         << "  Samples      : " << cfg.n_samples
         << " (+ " << cfg.n_burn_in << " burn-in)\n"
         << "  Bins         : " << cfg.n_period_bins << " x "
         << cfg.n_param_bins << "\n";
    if (ellipsoidal && cfg.lc_prior)
        cout << "  Ellipsoidal  : ENABLED\n";
    cout << "\n";

    star.run_rv_mcmc(cfg);

    // ========== save results ==========
    // Legacy flat-file (backward compat)
    string suffix = cfg.eccentric ? "_eccentric" : "";
    saveCSV(out_path + "pamp_full" + gaia_id + suffix + ".csv",
            star.period_amp_histogram);

    save_histograms(folder, star, cfg.eccentric);
    save_metadata(folder, gaia_id, cfg, lc_prior_source, ellipsoidal);

    cout << "Saved all corner-plot histograms to " << folder << "\n";
    return 0;
}