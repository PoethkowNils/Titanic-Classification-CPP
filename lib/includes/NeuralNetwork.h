#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include <vector>
#include <string>

class NeuralNetwork {
public:
    // Single neuron calculation
    double singleNeuron(double input, double weight);

    // Multiple inputs, single output
    double multipleInputSingleOutput(const std::vector<double>& inputs, const std::vector<double>& weights, double bias);

    // Single input, multiple outputs
    void singleInputMultipleOutput(double input, std::vector<double> weights, double bias, std::vector<double>& outputs);

    // Multiple inputs, multiple outputs
    void multipleInputMultipleOutput(const std::vector<double>& input, 
                                 const std::vector<std::vector<double>>& weights,
                                 const std::vector<double>& biases, 
                                 std::vector<double>& output, 
                                 int inputSize, int outputSize);

    // Hidden layer function
    void hiddenLayer(const std::vector<double>& inputs, std::vector<double>& hiddenWeights, std::vector<double>& hiddenBiases, std::vector<double>& hiddenOutputs, int inputSize, int hiddenSize);

    // Error calculation
    void calculateError(std::vector<double>& predictedOutput, const std::vector<double>& groundTruth, std::vector<double>& error);

    // Mean Squared Error (MSE)
    double calculateMSE(std::vector<double>& error);

    // Root Mean Squared Error (RMSE)
    double calculateRMSE(double mse);

    // Brute-force learning to find the best weight
    void bruteForceLearning(double input, double& weight, double expectedValue, double learningRate, int maxEpochs);

    // Backpropagation learning function
    void backpropagation(const std::vector<double>& input, const std::vector<double>& expectedOutput, 
                     std::vector<std::vector<double>>& inputToHiddenWeights, std::vector<double>& hiddenBiases,
                     std::vector<std::vector<double>>& hiddenToOutputWeights, std::vector<double>& outputBiases,
                     double learningRate, int epochs);

    void backpropagation2layer(const std::vector<std::vector<double>>& batchInputs, 
                           const std::vector<double>& batchExpectedOutputs, // Changed from 2d vector
                           std::vector<std::vector<double>>& inputToHidden1Weights, std::vector<double>& hidden1Biases,
                           std::vector<std::vector<double>>& hidden1ToHidden2Weights, std::vector<double>& hidden2Biases,
                           std::vector<double>& hidden2ToOutputWeights, double& outputBias,
                           double learningRate);


    // Activation functions (ReLU and Sigmoid)
    double relu(double x);
    double sigmoid(double x);

    double reluDerivative(double x);
    double sigmoidDerivative(double x);

    // Vectorized activation functions
    void vectorReLU(std::vector<double>& input);
    void vectorSigmoid(std::vector<double>& input);

    // Print a 2D matrix
    void printMatrix(int rows, int cols, const std::vector<std::vector<double>>& matrix);

    // Train Model
    double modelFit(const std::vector<std::vector<double>>& data, const std::vector<double>& labels, int batchSize,
                                std::vector<std::vector<double>>& inputToHidden1Weights, std::vector<double>& hidden1Biases,
                                std::vector<std::vector<double>>& hidden1ToHidden2Weights, std::vector<double>& hidden2Biases,
                                std::vector<double>& hidden2ToOutputWeights, double& outputBias,
                                double learningRate);

    // Make predictions
    double modelPredict(const std::vector<std::vector<double>>& data, std::vector<double>& labels,
                                const std::vector<std::vector<double>>& inputToHidden1Weights, const std::vector<double>& hidden1Biases,
                                const std::vector<std::vector<double>>& hidden1ToHidden2Weights, const std::vector<double>& hidden2Biases,
                                const std::vector<double>& hidden2ToOutputWeights, double& outputBias);

    void forwardPropagation(const std::vector<std::vector<double>>& data, std::vector<double>& predictedOutputs, 
                                const std::vector<std::vector<double>>& inputToHidden1Weights, const std::vector<double>& hidden1Biases,
                                const std::vector<std::vector<double>>& hidden1ToHidden2Weights, const std::vector<double>& hidden2Biases,
                                const std::vector<double>& hidden2ToOutputWeights, double& outputBias);

    // Compute cost for logistic regression
        // Changed to 1D vector input parameters
    double computeCost(const std::vector<double>& yhat, const std::vector<double>& y);

    // Normalize a 2D matrix
    int normalizeData2D(const std::vector<std::vector<double>>& inputMatrix, std::vector<std::vector<double>>& outputMatrix);

    // Save network
    void saveNetwork(const std::string& filename, int numOfFeatures, int numOfHiddenNodes1, 
                                int numOfHiddenNodes2, int numOfOutputNodes,
                                std::vector<std::vector<double>>& inputToHidden1Weights, std::vector<double>& hidden1Biases,
                                std::vector<std::vector<double>>& hidden1ToHidden2Weights, std::vector<double>& hidden2Biases,
                                std::vector<double>& hidden2ToOutputWeights, double outputLayerBias);

    // Load network
    void loadNetwork(const std::string& filename, int numOfFeatures, int numOfHiddenNodes1, 
                                int numOfHiddenNodes2, int numOfOutputNodes,
                                std::vector<std::vector<double>>& inputToHidden1Weights, std::vector<double>& hidden1Biases,
                                std::vector<std::vector<double>>& hidden1ToHidden2Weights, std::vector<double>& hidden2Biases,
                                std::vector<std::vector<double>>& hidden2ToOutputWeights, std::vector<double>& outputLayerBias);

    void initializeWeights(std::vector<std::vector<double>>& inputToHidden1Weights, std::vector<std::vector<double>>& hidden1ToHidden2Weights, 
                                std::vector<double>& hidden2ToOutputWeights, int n);
};

#endif // NEURALNETWORK_H
