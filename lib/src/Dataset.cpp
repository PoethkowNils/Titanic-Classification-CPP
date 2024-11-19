#include "../includes/Dataset.h"
#include <fstream>
#include <sstream>
#include <string>
#include <iostream>
#include <random>
#include <algorithm>
#include <cctype> // For isspace

// Seed for random number generator
unsigned int shuffleSeed = 449; // Default seed value for reproducibility

// Helper function to trim spaces from the beginning and end of a string
std::string trim(const std::string& str) {
    size_t first = str.find_first_not_of(' ');
    size_t last = str.find_last_not_of(' ');
    return (first == std::string::npos || last == std::string::npos) ? "" : str.substr(first, last - first + 1);
}

/*----Added Functions----*/
// Convert string type data to double
std::vector<std::vector<double>> Dataset::toNumeric(std::vector<std::vector<std::string>>& data) {
    std::vector<std::vector<double>> numData;
    numData.reserve(data.size());  // Reserve space to avoid unnecessary reallocations

    for (auto& row : data) {
        std::vector<double> numRow;
        numRow.reserve(row.size());  // Reserve space for the row

        for (auto& col : row) {
            numRow.push_back(std::stod(std::move(col)));  // Convert and move the value
        }

        numData.push_back(std::move(numRow));  // Move the row into numData
    }

    // Clear the original data to release memory
    data.clear();
    data.shrink_to_fit();

    return numData;
}
// Encode gender/sex (Male=0, Female=1)
void Dataset::encodeGender(std::vector<std::vector<std::string>>& data) {
    for (auto &row: data) {
        row[1] = (row[1]=="male")? "1.0" : "0.0"; // Sex is the second feature
    }
}
/*---- ----*/

// Helper function to split a string by commas while handling quotes
std::vector<std::string> Dataset::split(const std::string& line, char delimiter) {
    std::vector<std::string> tokens;
    std::string token;
    bool inQuotes = false;
    
    for (char ch : line) {
        if (ch == '"' && !inQuotes) {
            inQuotes = true;
        } else if (ch == '"' && inQuotes) {
            inQuotes = false;
        } else if (ch == delimiter && !inQuotes) {
            tokens.push_back(trim(token)); // Trim spaces before adding
            token.clear();
        } else {
            token += ch;
        }
    }
    // Add the last token
    tokens.push_back(trim(token));

    // Remove unnecessary quotes from tokens
    for (auto& tok : tokens) {
        if (!tok.empty() && tok.front() == '"' && tok.back() == '"') {
            tok = tok.substr(1, tok.size() - 2);
        }
    }
    
    return tokens;
}

// Load the dataset from a CSV file
void Dataset::processDataset(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }

    std::string line;
    bool firstRow = true;
    while (std::getline(file, line)) {
        if (firstRow) {
            firstRow = false; // Skip header row
            continue;
        }
        data.push_back(split(line, ',')); // Parse each line
    }
    file.close();
}

// Select input columns (converting 1-based indices to 0-based internally)
void Dataset::inputColumns(const std::vector<int>& columns) {
    inputCols.clear();
    inputCols.reserve(columns.size());
    std::transform(columns.begin(), columns.end(), std::back_inserter(inputCols), [](int col) { return col - 1; });
}

// Select output column (convert 1-based index to 0-based internally)
void Dataset::outputColumn(int column) {
    outputCol = column - 1;
}

// Filter the data to include only the selected input and output columns
std::vector<std::string> Dataset::filterRow(const std::vector<std::string>& row) {
    std::vector<std::string> filteredRow;
    filteredRow.reserve(inputCols.size() + 1);

    // Add input columns
    for (int colIndex : inputCols) {
        if (colIndex < row.size() && !row[colIndex].empty()) {
            filteredRow.push_back(row[colIndex]);
        } else {
            return {}; // Return empty if any of the input columns have no value
        }
    }

    // Add output column
    if (outputCol < row.size() && !row[outputCol].empty()) {
        filteredRow.push_back(row[outputCol]);
    } else {
        return {}; // Return empty if output column has no value
    }
    return filteredRow;
}

// Divide data into training and testing sets (80/20)
void Dataset::divideTrainAndTestData() {
    size_t trainSize = static_cast<size_t>(data.size() * 0.8);

    trainData.assign(data.begin(), data.begin() + trainSize);
    testData.assign(data.begin() + trainSize, data.end());

    shuffleTrainData(); // Shuffle only the training data
}

// Shuffle the training data using the seed
void Dataset::shuffleTrainData() {
    std::mt19937 g(shuffleSeed); // Use the specified seed for reproducibility
    std::shuffle(trainData.begin(), trainData.end(), g);
}

// Set a custom seed for shuffling
void Dataset::setShuffleSeed(unsigned int seed) {
    shuffleSeed = seed;
}

// Get a sample from the training data with filtered columns
std::vector<std::vector<std::string>> Dataset::getTrainDataSample(size_t sampleSize) {
    return getSample(trainData, sampleSize);
}

// Get a sample from the testing data with filtered columns
std::vector<std::vector<std::string>> Dataset::getTestDataSample(size_t sampleSize) {
    return getSample(testData, sampleSize);
}

// Helper function to get a sample from the data
std::vector<std::vector<std::string>> Dataset::getSample(const std::vector<std::vector<std::string>>& sourceData, size_t sampleSize) {
    std::vector<std::vector<std::string>> sample;
    sample.reserve(sampleSize);
    size_t count = 0;
    
    for (const auto& row : sourceData) {
        auto filteredRow = filterRow(row);
        if (!filteredRow.empty()) {
            sample.push_back(filteredRow);
            count++;
        }
        if (count >= sampleSize) {
            break;
        }
    }
    return sample;
}
