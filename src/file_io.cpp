//
// Created by fabian on 9/4/24.
//

#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <string>
#include <tuple>
#include <vector>
#include <iomanip>


using namespace std;

// helper to lowercase a string
string to_lower(const std::string &s) {
    string out = s;
    transform(out.begin(), out.end(), out.begin(),
                   [](unsigned char c){ return tolower(c); });
    return out;
}

vector<vector<double>> readCSV(const string& filename, bool skipheader, const string& delimiter, const string& comment) {
    vector<vector<double>> columns;
    ifstream file(filename);

    if (!file.is_open()) {
        cerr << "readCSV: Failed to open the file: " << filename << endl;
        return columns; // return empty vector in case of error
    }

    string line;
    int n = 0;
    while (getline(file, line)) {
        // Skip the header line if skipheader is true
        if (skipheader && n == 0) {
            n++;
            continue;
        }

        // Skip lines starting with the comment character if comment is not empty
        if (!comment.empty() && line[0] == comment[0]) {
            continue;
        }

        n++;
        stringstream ss(line);
        string value;
        vector<double> rowValues;

        while (getline(ss, value, ',')) {
            try {
                rowValues.push_back(stod(value));
            } catch (const std::invalid_argument&) {
                // Skip non-numeric columns or print warning
                cerr << "Warning: Skipping non-numeric value: '" << value 
                     << "' at line " << n << endl;
                break;  // Stop processing this row after first non-numeric
            }
        }

        // Resize columns if this is the first row
        if (columns.empty()) {
            columns.resize(rowValues.size());
        }

        // Add values to respective column vectors
        for (size_t i = 0; i < rowValues.size(); ++i) {
            columns[i].push_back(rowValues[i]);
        }
    }

    file.close();
    return columns;
}




void saveCSV(const string& filename, const vector<vector<double>>& data) {
    ofstream file(filename);

    if (!file.is_open()) {
        cerr << "saveCSV: Failed to open the file." << endl;
        return;
    }

    size_t numRows = data.empty() ? 0 : data[0].size();
    size_t numCols = data.size();

    stringstream buffer;

    // Write the data in a buffered manner
    for (size_t i = 0; i < numRows; ++i) {
        for (size_t j = 0; j < numCols; ++j) {
            buffer << setprecision(15) << data[j][i]; // Write the value

            if (j < numCols - 1) {
                buffer << ","; // Add a comma if this is not the last column
            }
        }
        buffer << "\n"; // New line after each row
    }

    file << buffer.rdbuf();  // Write the entire buffer to file in one operation
    file.close();

    cout << "Data saved to " << filename << endl;
}