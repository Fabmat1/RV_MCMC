//
//  rv_mcmc.cpp  –  Unified entry point for circular and eccentric
//                  MCMC fitting of radial-velocity curves.
//
//  Now writes raw MCMC chain to binary file instead of fixed-resolution
//  histograms.  Post-processing (adaptive histograms, corner plots)
//  is done in Python.
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

#include <unistd.h>

namespace po = boost::program_options;
using namespace std;

// ---------------------------------------------------------------
static string str_lower(string s) {
    transform(s.begin(), s.end(), s.begin(),
              [](unsigned char c) { return tolower(c); });
    return s;
}

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

    if (ellipsoidal && !data.empty() && !data[0].empty()) {
        cout << "Ellipsoidal mode: scaling LC periods by 2x\n";
        for (auto& v : data[0]) v *= 2.0;
    }
    return data;
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
      << "Chain thinning factor: " << cfg.chain_thin << "\n"
      << "\nParameter bounds:\n"
      << "  Amplitude: " << cfg.amp_min << " - " 
      << (cfg.amp_max > 0 ? cfg.amp_max : cfg.amp_lim) << "\n"
      << "  Offset:    " << (cfg.offset_min != 0 ? cfg.offset_min : -cfg.offset_lim) 
      << " - " << (cfg.offset_max != 0 ? cfg.offset_max : cfg.offset_lim) << "\n"
      << "  Phase:     " << cfg.phase_min << " - " << cfg.phase_max << "\n";
    if (cfg.eccentric) {
        f << "  Ecc:       " << cfg.ecc_min << " - " << cfg.ecc_max << "\n"
          << "  Omega:     " << cfg.omega_min << " - " << cfg.omega_max << " deg\n";
    }
    f << "\nStep sizes:\n"
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

    string gaia_id;
    string out_path      = "out/";
    string input_path;
    string lc_prior_source = "ALL";

    MCMCConfig cfg;
    bool ellipsoidal = false;

    try {
        po::options_description desc("Allowed options");
        desc.add_options()
            ("help",
                "produce help message")
            ("gaia-id",
                po::value<string>(&gaia_id)->required(),
                "GAIA ID (positional)")

            ("eccentric",
                po::bool_switch(&cfg.eccentric),
                "Use Keplerian (eccentric) RV model instead of sinusoidal")

            ("l",  po::value<double>(&cfg.min_period),
                "Low period bound (days)")
            ("h",  po::value<double>(&cfg.max_period),
                "High period bound (days)")
            ("n",  po::value<int>(&cfg.n_samples),
                "Number of post-burn-in MCMC samples")
            ("n-burn-in", po::value<int>(&cfg.n_burn_in),
                "Number of burn-in samples (default 1000000)")

            ("amp-lim",    po::value<double>(&cfg.amp_lim),
                "Hard amplitude limit")
            ("offset-lim", po::value<double>(&cfg.offset_lim),
                "Hard offset limit")
            ("amp-min",    po::value<double>(&cfg.amp_min),
                "Minimum amplitude bound (default 0)")
            ("amp-max",    po::value<double>(&cfg.amp_max),
                "Maximum amplitude bound (default amp-lim)")
            ("offset-min", po::value<double>(&cfg.offset_min),
                "Minimum offset bound (default -offset-lim)")
            ("offset-max", po::value<double>(&cfg.offset_max),
                "Maximum offset bound (default offset-lim)")
            ("phase-min",  po::value<double>(&cfg.phase_min),
                "Minimum phase bound (default -0.5)")
            ("phase-max",  po::value<double>(&cfg.phase_max),
                "Maximum phase bound (default 0.5)")
            ("ecc-min",    po::value<double>(&cfg.ecc_min),
                "Minimum eccentricity bound (default 0)")
            ("ecc-max",    po::value<double>(&cfg.ecc_max),
                "Maximum eccentricity bound (default 0.9999)")
            ("omega-min",  po::value<double>(&cfg.omega_min),
                "Minimum omega bound in degrees (default 0)")
            ("omega-max",  po::value<double>(&cfg.omega_max),
                "Maximum omega bound in degrees (default 360)")


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

            ("out-dir", po::value<string>(&out_path),
                "Output directory (default: out/)")
            ("input",   po::value<string>(&input_path),
                "Input CSV file path")

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

            ("n-temperatures",  po::value<int>(&cfg.n_temperatures),
                "Number of parallel tempering chains (default 8)")
            ("max-temperature", po::value<double>(&cfg.max_temperature),
                "Highest temperature in ladder (default 100)")
            ("swap-interval",   po::value<int>(&cfg.swap_interval),
                "Attempt chain swap every N steps (default 20)")
            ("target-accept",   po::value<double>(&cfg.target_accept),
                "Target acceptance rate for adaptation (default 0.234)")

            ("thin", po::value<int>(&cfg.chain_thin),
                "Chain thinning factor – save every Nth post-burn-in sample "
                "(default 100)")
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
                 << "  " << argv[0] << " 1234567890 -l 0.1 -h 100 -n 5000000\n"
                 << "  " << argv[0] << " 1234567890 --eccentric -l 0.1 -h 100 -n 5000000 --lc-prior\n"
                 << "  " << argv[0] << " 1234567890 -n 50000000 --thin 10   # finer chain\n";
            return 0;
        }
        po::notify(vm);

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
    
    // Save reference time for T₀ = t_ref + phase * period
    double t_ref = get_min(rvs[3]);
    {
        ofstream trf(folder + "t_ref.txt");
        trf << fixed << setprecision(10) << t_ref << "\n";
    }
    cout << "Reference time (BJD): " << fixed << setprecision(6) << t_ref << "\n";

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

    // ========== set chain output directory ==========
    cfg.chain_output_dir = folder;

    // ========== run MCMC ==========
    cout << "\nRunning " << (cfg.eccentric ? "ECCENTRIC" : "CIRCULAR")
         << " MCMC\n"
         << "  Period range : " << cfg.min_period << " – "
         << cfg.max_period << " d\n"
         << "  Samples      : " << cfg.n_samples
         << " (+ " << cfg.n_burn_in << " burn-in)\n"
         << "  Chain thin   : " << cfg.chain_thin << "\n";
    if (ellipsoidal && cfg.lc_prior)
        cout << "  Ellipsoidal  : ENABLED\n";
    cout << "\n";

    star.run_rv_mcmc(cfg);

    // ========== save metadata ==========
    save_metadata(folder, gaia_id, cfg, lc_prior_source, ellipsoidal);

    cout << "All output saved to " << folder << "\n";
    return 0;
}