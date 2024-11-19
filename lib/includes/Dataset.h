#ifndef DATASET_H
#define DATASET_H

#include <vector>
#include <string>
#include <utility>

class Dataset {
public:
    // Convert datasat to numeric (double)
    std::vector<std::vector<double>> toNumeric(std::vector<std::vector<std::string>>& data);

    // Encode gender/sex (Male=0, Female=1) 
    void encodeGender(std::vector<std::vector<std::string>>& data);

    // Load the dataset from a CSV file
    void processDataset(const std::string& filename);

    // Select input columns
    void inputColumns(const std::vector<int>& columns);

    // Select output column
    void outputColumn(int column);

    // Divide data into training and testing sets (80/20)
    void divideTrainAndTestData();

    // Get a sample from the training data
    std::vector<std::vector<std::string>> getTrainDataSample(size_t sampleSize);

    // Get a sample from the testing data
    std::vector<std::vector<std::string>> getTestDataSample(size_t sampleSize);

    // Set a custom seed for shuffling
    void setShuffleSeed(unsigned int seed);

    //void encodeGender();

private:
    std::vector<std::vector<std::string>> data;
    std::vector<std::vector<std::string>> trainData;
    std::vector<std::vector<std::string>> testData;
    std::vector<int> inputCols;
    int outputCol;

    // Seed for random shuffling
    unsigned int shuffleSeed = 42;

    // Helper functions
    std::vector<std::string> split(const std::string& line, char delimiter);
    void shuffleTrainData(); // Shuffle only the training data

    // Filter row to include only selected input and output columns
    std::vector<std::string> filterRow(const std::vector<std::string>& row);

    // Helper function to get a sample from the data
    std::vector<std::vector<std::string>> getSample(const std::vector<std::vector<std::string>>& sourceData, size_t sampleSize);
};

#endif // DATASET_H
