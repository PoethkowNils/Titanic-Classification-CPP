#include <iostream>
#include <cmath>
#include <vector>
#include <map>
#include <fstream>
#include <random>
#include <limits>
#include <iomanip>
#include "../includes/NeuralNetwork.h"

using std::cout, std::endl;

double NeuralNetwork::singleNeuron(double input, double weight) 
{
    return input*weight;
}

double NeuralNetwork::multipleInputSingleOutput(const std::vector<double>& inputs, const std::vector<double>& weights, double bias)
{
    int size = inputs.size();
    double sum; 

    for (int i=0; i<size; i++)
        sum += inputs[i]*weights[i];

    return sum;
}

void NeuralNetwork::singleInputMultipleOutput(double input, std::vector<double> weights, double bias, std::vector<double>& outputs)
{ //function is void because outputs should be changed in place (by reference)
    int size = weights.size();

    for (int i=0; i<size; i++)
        outputs[i] = input*weights[i]+bias; //add bias for every calculation
}

//where do I need these functions?
void NeuralNetwork::calculateError(std::vector<double>& predictedOutput, const std::vector<double>& groundTruth, std::vector<double>& error) 
{       
    int size = predictedOutput.size();

    for (int i = 0; i<size; i++) {
        error[i] = pow(predictedOutput[i] - groundTruth[i], 2);
        cout << "For ground truth "<< groundTruth[i] << ", predicted output is " << predictedOutput[i] << ", error is " << error[i] << endl;
    }
}

double NeuralNetwork::calculateMSE(std::vector<double>& error) 
{
    double sum;
    for (double err: error) {
        sum += pow(err, 2);
    }
    
    return sum/(2*error.size());
}

double NeuralNetwork::calculateRMSE(double mse) {
    return sqrt(mse);
}

// Note: Derivative of relu is 1 (x'=1)
double NeuralNetwork::relu(double x) {
    return (x > 0) ? x : 0.0;
}
double NeuralNetwork::reluDerivative(double x) {
    return (x >= 0) ? 1.0 : 0.0;
}

double NeuralNetwork::sigmoid(double x) 
{
    return (1 / (1 + exp(-x)));
}

double NeuralNetwork::sigmoidDerivative(double x) 
{
    return ((1 - sigmoid(x)) * sigmoid(x));
}

// New function to replace multipleInputMultipleOutuput (handles 2d vectors)
void NeuralNetwork::multipleInputMultipleOutput(const std::vector<double>& input, 
                                 const std::vector<std::vector<double>>& weights,
                                 const std::vector<double>& biases, 
                                 std::vector<double>& output, 
                                 int inputSize, int outputSize) 
{
    for (int i = 0; i < outputSize; i++) {
        output[i] = biases[i];
        for (int j = 0; j < inputSize; j++) {
            output[i] += input[j] * weights[j][i];
        }
        output[i] = relu(output[i]);
    }
}

// Initialize the Weights for ReLu function (Variance 2/n) with Xavier initialization
void NeuralNetwork::initializeWeights(std::vector<std::vector<double>>& inputToHidden1Weights, 
                                      std::vector<std::vector<double>>& hidden1ToHidden2Weights, 
                                      std::vector<double>& hidden2ToOutputWeights, 
                                      int n) {
    // Random number generator
    std::random_device rd;
    std::mt19937 gen(rd());

    // Normal distribution with mean 0 and stddev sqrt(2/n)
    std::normal_distribution<> dist(0.0, std::sqrt(2.0 / n));

    // Initialize weights by drawing from distribution
    for (auto& row : inputToHidden1Weights) {
        for (auto& weight : row) {
            weight = dist(gen);
        }
    }

    for (auto& row : hidden1ToHidden2Weights) {
        for (auto& weight : row) {
            weight = dist(gen);
        }
    }

    for (auto& weight : hidden2ToOutputWeights) {
        weight = dist(gen);
    }
}

double computeBinaryCrossEntropyLoss(const std::vector<double>& predictedOutputs, 
                                     const std::vector<double>& actualLabels) {
    double loss = 0.0;
    int n = predictedOutputs.size();

    for (int i = 0; i < n; i++) {
        double y = actualLabels[i];
        double y_hat = predictedOutputs[i];

        // To avoid log(0) errors, add a small epsilon
        double epsilon = 1e-15;
        y_hat = std::max(epsilon, std::min(1.0 - epsilon, y_hat));

        loss += y * std::log(y_hat) + (1 - y) * std::log(1 - y_hat);
    }

    return -loss / n;  // Return average loss
}

void NeuralNetwork::forwardPropagation(const std::vector<std::vector<double>>& data, 
                                       std::vector<double>& predictedOutputs, 
                                       const std::vector<std::vector<double>>& inputToHidden1Weights, 
                                       const std::vector<double>& hidden1Biases,
                                       const std::vector<std::vector<double>>& hidden1ToHidden2Weights, 
                                       const std::vector<double>& hidden2Biases,
                                       const std::vector<double>& hidden2ToOutputWeights, 
                                       double& outputBias) {
    
    // Resize predicted outputs to match the input data size
    predictedOutputs.resize(data.size());

    // Iterate through each input sample in the batch
    for (int i = 0; i < data.size(); i++) {
        const auto& input = data[i];

        // Input to Hidden1
        std::vector<double> hidden1Activation(hidden1Biases.size(), 0.0);
        for (int h1 = 0; h1 < hidden1Biases.size(); h1++) {
            for (int in = 0; in < input.size(); in++) {
                hidden1Activation[h1] += input[in] * inputToHidden1Weights[in][h1];
            }
            hidden1Activation[h1] += hidden1Biases[h1];
            hidden1Activation[h1] = relu(hidden1Activation[h1]);
        }

        // 2. Hidden1 to Hidden2
        std::vector<double> hidden2Activation(hidden2Biases.size(), 0.0);
        for (int h2 = 0; h2 < hidden2Biases.size(); h2++) {
            for (int h1 = 0; h1 < hidden1Activation.size(); h1++) {
                hidden2Activation[h2] += hidden1Activation[h1] * hidden1ToHidden2Weights[h1][h2];
            }
            hidden2Activation[h2] += hidden2Biases[h2];
            hidden2Activation[h2] = relu(hidden2Activation[h2]);
        }

        // Hidden2 to Output
        double outputSum = outputBias;
        for (int h2 = 0; h2 < hidden2ToOutputWeights.size(); h2++) {
            outputSum += hidden2ToOutputWeights[h2] * hidden2Activation[h2];
        }

        predictedOutputs[i] = sigmoid(outputSum);
    }
}

double calculateAccuracy(const std::vector<double>& predictedOutputs, 
                         const std::vector<double>& expectedOutputs) {
    if (predictedOutputs.size() != expectedOutputs.size()) {
        std::cerr << "Error: Size of predicted and expected outputs must match!" << std::endl;
        return 0.0;
    }

    int correctCount = 0;
    int total = predictedOutputs.size();

    // Iterate over predictions and expected outputs
    for (int i = 0; i < total; i++) {
        // Apply threshold to classify as 0 or 1
        int predictedClass = predictedOutputs[i] >= 0.5 ? 1 : 0;
        int expectedClass = expectedOutputs[i];

        // Check if prediction matches expected output
        if (predictedClass == expectedClass) {
            ++correctCount;
        }
    }

    return (static_cast<double>(correctCount) / total) * 100.0;
}

double calculateF1score(const std::vector<double>& predictedOutputs, 
                        const std::vector<double>& expectedOutputs) {
    if (predictedOutputs.size() != expectedOutputs.size()) {
        std::cerr << "Error: Size of predicted and expected outputs must match!" << std::endl;
        return 0.0;
    }

    // Vector to store tp, tn, fp, fn
    std::map<std::string, double> measures{{"tp", 0.0}, {"tn", 0.0}, {"fp", 0.0}, {"fn", 0.0}};

    for (int i = 0; i<predictedOutputs.size(); i++) {
        if (predictedOutputs[i] == expectedOutputs[i] && predictedOutputs[i]==1)
            measures["tp"] += 1;
        else if (predictedOutputs[i] == expectedOutputs[i] && predictedOutputs[i]==0)
            measures["tn"] += 1;
        else if (predictedOutputs[i] != expectedOutputs[i] && predictedOutputs[i]==1)
            measures["fp"] += 1;
        else if (predictedOutputs[i] != expectedOutputs[i] && predictedOutputs[i]==0)
            measures["fn"] += 1;
    }

    double precision = measures["tp"] / (measures["tp"] + measures["fp"]);
    double recall = measures["tp"] / (measures["tp"] + measures["fn"]);

    if (std::isnan(precision) || std::isnan(recall) || (precision + recall) == 0) {
        return 0.0;
    }

    double f1score = 2* (precision*recall) / (precision+recall);

    return f1score;
}

// Predict the labels of training data and return the test-loss
double NeuralNetwork::modelPredict(const std::vector<std::vector<double>>& data, 
                                    std::vector<double>& labels, //temp non const
                                    const std::vector<std::vector<double>>& inputToHidden1Weights, 
                                    const std::vector<double>& hidden1Biases,
                                    const std::vector<std::vector<double>>& hidden1ToHidden2Weights, 
                                    const std::vector<double>& hidden2Biases,
                                    const std::vector<double>& hidden2ToOutputWeights, 
                                    double& outputBias) {

    std::vector<std::vector<double>> normalizedData(data.size(), std::vector<double>(data[0].size(), 0.0));
    if (normalizeData2D(data, normalizedData)) {
        std::cerr << "Unable to fit the model, due to dataset" << std::endl;
        return 0;
    }   

    std::vector<double> predictedOutputs(labels.size());
    forwardPropagation(normalizedData, predictedOutputs, 
                       inputToHidden1Weights, hidden1Biases, 
                       hidden1ToHidden2Weights, hidden2Biases, 
                       hidden2ToOutputWeights, outputBias);

    double cost = computeCost(predictedOutputs, labels);

    // Set predictions to labels based on threshold (for metric calculationss)
    for (int i = 0; i<predictedOutputs.size(); i++) {
        predictedOutputs[i] = (predictedOutputs[i] >= 0.5) ? 1.0 : 0.0;
    }

    double accuracy = calculateAccuracy(predictedOutputs, labels);
    double f1score = calculateF1score(predictedOutputs, labels);

    cout << accuracy << endl;

    return cost;
}

double NeuralNetwork::modelFit(const std::vector<std::vector<double>>& data, 
                               const std::vector<double>& labels, int batchSize,
                               std::vector<std::vector<double>>& inputToHidden1Weights, 
                               std::vector<double>& hidden1Biases,
                               std::vector<std::vector<double>>& hidden1ToHidden2Weights, 
                               std::vector<double>& hidden2Biases,
                               std::vector<double>& hidden2ToOutputWeights, 
                               double& outputBias, double learningRate) {
    // Normalize data
    std::vector<std::vector<double>> normalizedData(data.size(), std::vector<double>(data[0].size(), 0.0));
    if (normalizeData2D(data, normalizedData)) {
        std::cerr << "Unable to fit the model, due to dataset" << std::endl;
        return 0;
    }

    double totalLoss = 0.0;  // Track loss across batches

    // Iterate over batches
    for (int b = 0; b < (data.size() + batchSize - 1) / batchSize; b++) {
        int startIndex = b * batchSize;
        int endIndex = std::min(startIndex + batchSize, static_cast<int>(data.size()));

        // Prepare batch data
        std::vector<std::vector<double>> batchInputs;
        std::vector<double> batchExpectedOutputs;
        for (int i = startIndex; i < endIndex; ++i) {
            batchInputs.push_back(normalizedData[i]);
            batchExpectedOutputs.push_back(labels[i]);
        }

        // Backpropagation to update weights and biases
        backpropagation2layer(batchInputs, batchExpectedOutputs, 
                              inputToHidden1Weights, hidden1Biases, 
                              hidden1ToHidden2Weights, hidden2Biases,
                              hidden2ToOutputWeights, outputBias, learningRate);

        // Forward pass to get predicted outputs for the batch
        std::vector<double> batchPredictedOutputs(batchExpectedOutputs.size());
        forwardPropagation(batchInputs, batchPredictedOutputs, 
                           inputToHidden1Weights, hidden1Biases, 
                           hidden1ToHidden2Weights, hidden2Biases, 
                           hidden2ToOutputWeights, outputBias);
        
        totalLoss += computeCost(batchPredictedOutputs, batchExpectedOutputs);
    }

    // Return the loss for this epoch (devided by the number of batches)
    return totalLoss/(data.size()/std::ceil(batchSize));
}

// Binary cross entropy for loss
double NeuralNetwork::computeCost(const std::vector<double>& yhat, const std::vector<double>& y) 
{
    double result = 0.0;
    double offset = 1e-9;  // Small value to prevent log(0)

    for (int i = 0; i < y.size(); i++) 
    {
        // Ensure predicted probabilities are in the range [offset, 1 - offset] to avoid log(0)
        double clippedYhat = std::max(offset, std::min(1.0 - offset, yhat[i]));
        
        // Binary Cross-Entropy Loss for each sample
        result += -(y[i] * log(clippedYhat) + (1 - y[i]) * log(1 - clippedYhat));
    }

    // Return the average loss across all samples
    return result / y.size();
}

void NeuralNetwork::backpropagation2layer(const std::vector<std::vector<double>>& batchInputs, 
                           const std::vector<double>& batchExpectedOutputs,
                           std::vector<std::vector<double>>& inputToHidden1Weights, std::vector<double>& hidden1Biases,
                           std::vector<std::vector<double>>& hidden1ToHidden2Weights, std::vector<double>& hidden2Biases,
                           std::vector<double>& hidden2ToOutputWeights, double& outputBias,
                           double learningRate)
{
    int batchSize = batchInputs.size();
    int inputSize = batchInputs[0].size();
    int hidden1Size = hidden1Biases.size();
    int hidden2Size = hidden2Biases.size();

    // Initialize gradient accumulators
    std::vector<std::vector<double>> inputToHidden1WeightGradients(inputSize, std::vector<double>(hidden1Size, 0.0));
    std::vector<double> hidden1BiasGradients(hidden1Size, 0.0);
    std::vector<std::vector<double>> hidden1ToHidden2WeightGradients(hidden1Size, std::vector<double>(hidden2Size, 0.0));
    std::vector<double> hidden2BiasGradients(hidden2Size, 0.0);
    std::vector<double> hidden2ToOutputWeightGradients(hidden2Size, 0.0);
    double outputBiasGradient = 0.0;

    for (int b = 0; b < batchSize; b++) {
        const auto input = batchInputs[b];
        double expectedOutput = batchExpectedOutputs[b];

        // Forward pass
        std::vector<double> hidden1Activation(hidden1Size, 0.0);
        std::vector<double> hidden2Activation(hidden2Size, 0.0);
        double output;

        // Input to Hidden1
        multipleInputMultipleOutput(input, inputToHidden1Weights, hidden1Biases, hidden1Activation, inputSize, hidden1Size);

        // Hidden1 to Hidden2
        multipleInputMultipleOutput(hidden1Activation, hidden1ToHidden2Weights, hidden2Biases, hidden2Activation, hidden1Size, hidden2Size);

        // Hidden2 to Output
        output = outputBias;
        for (int i = 0; i < hidden2Size; ++i) {
            output += hidden2Activation[i] * hidden2ToOutputWeights[i];
        }
        output = sigmoid(output);

        // Output Gradient
        double outputError = output - expectedOutput;
        double outputGradient = outputError * reluDerivative(output);

        // Hidden2 to Output gradients
        for (int i = 0; i < hidden2Size; ++i) {
            hidden2ToOutputWeightGradients[i] += outputGradient * hidden2Activation[i];
        }
        outputBiasGradient += outputGradient;

        // Hidden2 gradients
        std::vector<double> hidden2Gradients(hidden2Size, 0.0);
        for (int i = 0; i < hidden2Size; ++i) {
            hidden2Gradients[i] = outputGradient * hidden2ToOutputWeights[i] * reluDerivative(hidden2Activation[i]);
            hidden2BiasGradients[i] += hidden2Gradients[i];
        }

        // Hidden1 to Hidden2 gradients
        for (int i = 0; i < hidden1Size; ++i) {
            for (int j = 0; j < hidden2Size; ++j) {
                hidden1ToHidden2WeightGradients[i][j] += hidden2Gradients[j] * hidden1Activation[i]; // ARE THESE INDECIES WRONG?????
            }
        }

        // Hidden1 gradients
        std::vector<double> hidden1Gradients(hidden1Size, 0.0);
        for (int i = 0; i < hidden1Size; ++i) {
            hidden1Gradients[i] = 0;
            for (int j = 0; j < hidden2Size; ++j) {
                hidden1Gradients[i] += hidden2Gradients[j] * hidden1ToHidden2Weights[i][j]; // INDECIES ARE WRONG!!!! swap [i][j] to [j][i]
            }
            hidden1Gradients[i] *= reluDerivative(hidden1Activation[i]);
            hidden1BiasGradients[i] += hidden1Gradients[i];
        }

        // Input to Hidden1 gradients
        for (int i = 0; i < inputSize; ++i) {
            for (int j = 0; j < hidden1Size; ++j) {
                inputToHidden1WeightGradients[i][j] += hidden1Gradients[j] * input[i];
            }
        }
    }

    // Update weights and biases
    double scaleFactor = learningRate / std::ceil(batchInputs.size()/batchSize);
    
    for (int i = 0; i < inputSize; ++i) {
        for (int j = 0; j < hidden1Size; ++j) {
            inputToHidden1Weights[i][j] -= learningRate * inputToHidden1WeightGradients[i][j];
        }
    }

    for (int i = 0; i < hidden1Size; ++i) {
        hidden1Biases[i] -= learningRate * hidden1BiasGradients[i];
        for (int j = 0; j < hidden2Size; ++j) {
            hidden1ToHidden2Weights[i][j] -= learningRate * hidden1ToHidden2WeightGradients[i][j];
        }
    }

    for (int i = 0; i < hidden2Size; ++i) {
        hidden2Biases[i] -= learningRate * hidden2BiasGradients[i];
        hidden2ToOutputWeights[i] -= learningRate * hidden2ToOutputWeightGradients[i];
    }

    outputBias -= scaleFactor * outputBiasGradient;
}

// Changed from original template function (just one input)
void NeuralNetwork::vectorReLU(std::vector<double>& input) 
{
    for (int i = 0; i<input.size(); i++) {
        input[i] = relu(input[i]);
    }
}

void NeuralNetwork::vectorSigmoid(std::vector<double>& input) 
{
    for (int i = 0; i<input.size(); i++) {
        input[i] = sigmoid(input[i]);
    }
}

void NeuralNetwork::printMatrix(int rows, int cols, const std::vector<std::vector<double>>& matrix) 
{
    // Loop through each row
    for (int i = 0; i < rows; ++i) {
        // Loop through each column in the current row
        for (int j = 0; j < cols; ++j) {
            // Print the element with some spacing for readability
            std::cout << std::setw(10) << std::fixed << std::setprecision(4) << matrix[i][j] << " ";
        }
        std::cout << std::endl;  // Move to the next line after each row
    }
}

int NeuralNetwork::normalizeData2D(const std::vector<std::vector<double>>& inputMatrix, std::vector<std::vector<double>>& outputMatrix) 
{
    int rows = inputMatrix.size();
    int cols = inputMatrix[0].size();

    if (rows <= 1) {
        std::cerr << "ERROR: At least 2 examples are required. Current dataset length is " << rows << std::endl;
        return 1;
    } else {
        for (int j = 0; j < cols; j++) {
            double max = -9999999;
            double min = 9999999;

            // Find MIN and MAX values in the given column
            for (int i = 0; i < rows; i++) {
                if (inputMatrix[i][j] > max) {
                    max = inputMatrix[i][j];
                }
                if (inputMatrix[i][j] < min) {
                    min = inputMatrix[i][j];
                }
            }

            // Normalization
            for (int i = 0; i < rows; i++) {
                outputMatrix[i][j] = (inputMatrix[i][j] - min) / (max - min);
            }
        }
    }
    return 0;
}

// Now work for 2 hidden layer neural networks
void NeuralNetwork::saveNetwork(const std::string& filename, int numOfFeatures, int numOfHiddenNodes1, 
                                int numOfHiddenNodes2, int numOfOutputNodes,
                                std::vector<std::vector<double>>& inputToHidden1Weights, std::vector<double>& hidden1Biases,
                                std::vector<std::vector<double>>& hidden1ToHidden2Weights, std::vector<double>& hidden2Biases,
                                std::vector<double>& hidden2ToOutputWeights, double outputLayerBias) { // const correctness!!!!

    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << " for writing.\n";
        return;
    }

    file << "Input to Hidden Layer 1 Weights:\n";
    for (int i = 0; i < numOfHiddenNodes1; i++) {
        for (int j = 0; j < numOfFeatures; j++) {
            file << inputToHidden1Weights[j][i] << " ";
        }
        file << "\n";
    }

    file << "Hidden Layer 1 Biases:\n";
    for (double bias : hidden1Biases) {
        file << bias << " ";
    }
    file << "\n";

    file << "Hidden Layer 1 to Hidden Layer 2 Weights:\n";
    for (int i = 0; i < numOfHiddenNodes2; i++) {
        for (int j = 0; j < numOfHiddenNodes1; j++) {
            file << hidden1ToHidden2Weights[j][i] << " ";
        }
        file << "\n";
    }

    file << "Hidden Layer 2 Biases:\n";
    for (double bias : hidden2Biases) {
        file << bias << " ";
    }
    file << "\n";

    file << "Hidden Layer 2 to Output Layer Weights:\n";
    for (int j = 0; j < hidden2ToOutputWeights.size(); j++) {
        file << hidden2ToOutputWeights[j] << " ";
    }
    file << "\n";

    file << "Output Layer Bias:\n";
    file << outputLayerBias << " ";
    file << "\n";

    file.close();
    std::cout << "Network saved to file: " << filename << "\n";
}

void NeuralNetwork::loadNetwork(const std::string& filename, int numOfFeatures, int numOfHiddenNodes1, 
                                int numOfHiddenNodes2, int numOfOutputNodes,
                                std::vector<std::vector<double>>& inputToHidden1Weights, std::vector<double>& hidden1Biases,
                                std::vector<std::vector<double>>& hidden1ToHidden2Weights, std::vector<double>& hidden2Biases,
                                std::vector<std::vector<double>>& hidden2ToOutputWeights, std::vector<double>& outputLayerBias) {

    // Clear and resize vectors
    inputToHidden1Weights.clear();
    hidden1Biases.clear();
    hidden1ToHidden2Weights.clear();
    hidden2Biases.clear();
    hidden2ToOutputWeights.clear();
    outputLayerBias.clear();

    inputToHidden1Weights.resize(numOfHiddenNodes1, std::vector<double>(numOfFeatures));
    hidden1Biases.resize(numOfHiddenNodes1);
    hidden1ToHidden2Weights.resize(numOfHiddenNodes2, std::vector<double>(numOfHiddenNodes1));
    hidden2Biases.resize(numOfHiddenNodes2);
    hidden2ToOutputWeights.resize(numOfOutputNodes, std::vector<double>(numOfHiddenNodes2));
    outputLayerBias.resize(numOfOutputNodes);

    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << " for reading.\n";
        return;
    }

    std::string temp;

    // Read Input to Hidden Layer 1 Weights
    std::getline(file, temp);
    for (int i = 0; i < numOfHiddenNodes1; i++) {
        for (int j = 0; j < numOfFeatures; j++) {
            if (!(file >> inputToHidden1Weights[i][j])) {
                std::cerr << "Error reading input-to-hidden1 weight at (" << i << ", " << j << ")\n";
                return;
            }
        }
    }
    file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

    // Read Hidden Layer 1 Biases
    std::getline(file, temp);
    for (int i = 0; i < numOfHiddenNodes1; i++) {
        if (!(file >> hidden1Biases[i])) {
            std::cerr << "Error reading hidden1 bias at index " << i << "\n";
            return;
        }
    }
    file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

    // Read Hidden Layer 1 to Hidden Layer 2 Weights
    std::getline(file, temp);
    for (int i = 0; i < numOfHiddenNodes2; i++) {
        for (int j = 0; j < numOfHiddenNodes1; j++) {
            if (!(file >> hidden1ToHidden2Weights[i][j])) {
                std::cerr << "Error reading hidden1-to-hidden2 weight at (" << i << ", " << j << ")\n";
                return;
            }
        }
    }
    file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

    // Read Hidden Layer 2 Biases
    std::getline(file, temp);
    for (int i = 0; i < numOfHiddenNodes2; i++) {
        if (!(file >> hidden2Biases[i])) {
            std::cerr << "Error reading hidden2 bias at index " << i << "\n";
            return;
        }
    }
    file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

    // Read Hidden Layer 2 to Output Layer Weights
    std::getline(file, temp);
    for (int i = 0; i < numOfOutputNodes; i++) {
        for (int j = 0; j < numOfHiddenNodes2; j++) {
            if (!(file >> hidden2ToOutputWeights[i][j])) {
                std::cerr << "Error reading hidden2-to-output weight at (" << i << ", " << j << ")\n";
                return;
            }
        }
    }
    file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

    // Read Output Layer Biases
    std::getline(file, temp);
    for (int i = 0; i < numOfOutputNodes; i++) {
        if (!(file >> outputLayerBias[i])) {
            std::cerr << "Error reading output layer bias at index " << i << "\n";
            return;
        }
    }

    if (file.fail()) {
        std::cerr << "File stream encountered an error.\n";
        return;
    }

    file.close();
    std::cout << "Network loaded from file: " << filename << "\n";
}
