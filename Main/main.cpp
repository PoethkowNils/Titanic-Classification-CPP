#include "../lib/includes/Dataset.h"
#include <iostream>
#include <fstream>  // for saveErros
#include <string>
#include <limits>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include "../lib/includes/NeuralNetwork.h"

using std::cout, std::endl, std::string, std::vector;

// Function to generate a random value in the range [-limit, limit]
double randomValue(double limit) {
    return (static_cast<double>(rand()) / RAND_MAX) * 2 * limit - limit;
}

// He initialization for weight matrices
void heInitializeWeights(std::vector<std::vector<double>>& weights, int inputSize) {
    // Calculate the standard deviation for He initialization
    double limit = std::sqrt(2.0 / inputSize);

    // Initialize each weight with a random value in the range [-limit, limit]
    for (auto& row : weights) {
        for (auto& weight : row) {
            weight = randomValue(limit);
        }
    }
}
/* ----- ----- */

void splitFeaturesAndLabels(std::vector<std::vector<double>>& data, 
                                   std::vector<double>& labels) 
{
    for (auto& row : data) {
        labels.push_back(row.back());
        row.pop_back();
    }
}

void saveErrors(const std::vector<double>& trainError, const std::vector<double>& testError, std::string fileName) {
    std::ofstream file(fileName);

    file<<"train"<< ", " << "test" << std::endl;

    for (int i=0; i<trainError.size(); i++) {
        file << trainError[i] << ", " << testError[i] << std::endl;
    }
}

double check_balance(const vector<double>& labels) {
    double c = 0;
    for (const double& label: labels) {
        if (label==0) {
            c += 1;
        }
    }
    return c/labels.size();
}

/*----- Main -----*/

int main() {
    Dataset dataset;

    // Load the dataset
    dataset.processDataset("titanic_dataset.csv");

    // Set input columns (Pclass, Sex, Age, Fare; 1-based indexing)
    dataset.inputColumns({3, 5, 6, 10});
    // all the othe features are not likely to have a high correlation with whether the passenger survived

    // Set output column (Survived; 1-based indexing)
    dataset.outputColumn(2);

    // Shuffle the dataset
    // ATTENTION, doesn't shuffle dataset
    dataset.setShuffleSeed(42);

    // Divide into train and test sets (80/20 split)
    dataset.divideTrainAndTestData();

    /*----- Training and Testing -----*/
    vector<vector<string>> temp1 = dataset.getTrainDataSample(500);
    vector<vector<string>> temp2 = dataset.getTestDataSample(300);
    // Encode the Gender column to be 0 and 1
    dataset.encodeGender(temp1);
    dataset.encodeGender(temp2);
    // Convert datasat of strings to double
    vector<vector<double>> trainData = dataset.toNumeric(temp1);
    vector<vector<double>> testData = dataset.toNumeric(temp2);
    // Create seperate Label vectors
    vector<double> trainLabels;
    vector<double> testLabels;
    splitFeaturesAndLabels(trainData, trainLabels);
    splitFeaturesAndLabels(testData, testLabels);

    NeuralNetwork nn;
    // Initialize weights and biases for 4 Input, 7 hidden, 5 hidden and 1 output Neuron Neural Network
        // Initialize the biases to be zero
    std::vector<std::vector<double>> inputToHidden1Weights(4, std::vector<double>(7, 0.0));
    std::vector<std::vector<double>> hidden1ToHidden2Weights(7, std::vector<double>(5, 0.0));
    std::vector<double> hidden2ToOutputWeights(5, 0.0);
    nn.initializeWeights(inputToHidden1Weights, hidden1ToHidden2Weights, hidden2ToOutputWeights, trainLabels.size());

    // Initialize biases
    std::vector<double> hidden1Biases(7, 0.0);
    std::vector<double> hidden2Biases(5, 0.0);
    srand((unsigned int)time(0));  
    for (int i = 0; i<7; i++) {
        hidden1Biases[i] = (rand() / (double)RAND_MAX) * 0.001;
    }
    for (int i = 0; i<5; i++) {
        hidden2Biases[i] = (rand() / (double)RAND_MAX) * 0.01;
    }
    double outputBias = (rand() / (double)RAND_MAX) * 0.01;

    // Check for balance in dataset
    // double balance = check_balance(trainLabels);
    // cout << balance*100 << "% of the training samples have the label: died(0)" << endl;

    vector<double> trainError;
    vector<double> testError;
    double learningRate = 0.001;
    int batchSize = 20;
    int epochs = 900;

    // Check for Training and Test Dataset size
    // cout << trainLabels.size() << endl;
    // cout << testLabels.size() << endl;

    // Train model with batches
    for (int e = 0; e < epochs; e++) {

        // Compare with and without
        learningRate = learningRate;  // LEARNING RATE DECAY * std::exp(-0.002 * e)

        // Train the model and get training loss for this epoch
        double trainLoss = nn.modelFit(trainData, trainLabels, batchSize, 
                                    inputToHidden1Weights, hidden1Biases, 
                                    hidden1ToHidden2Weights, hidden2Biases, 
                                    hidden2ToOutputWeights, outputBias, learningRate);
        trainError.push_back(trainLoss);

        // Evaluate the model on test data and get test loss
        double testLoss = nn.modelPredict(testData, testLabels, 
                                        inputToHidden1Weights, hidden1Biases, 
                                        hidden1ToHidden2Weights, hidden2Biases, 
                                        hidden2ToOutputWeights, outputBias);
        testError.push_back(testLoss);

        // std::cout << "Epoch " << e + 1 << " - Train Loss: " << trainLoss 
        //         << ", Test Loss: " << testLoss << std::endl;
    }
    cout << "Training completed, Best Test Error: " << *std::min_element(testError.begin(), testError.end()) << endl;

    // Save Train and Test Erros of Training
    saveErrors(trainError, testError, "trainTestError.txt");

    // Save the network after training
    nn.saveNetwork("testNetwork.txt", 4, 7, 5, 1, inputToHidden1Weights, hidden1Biases, hidden1ToHidden2Weights, hidden2Biases, hidden2ToOutputWeights, outputBias);

    return 0;
}