#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <vector>
#include <iomanip>

/// on-going tasks:
/// find out why the network does not work on more than 3 hidden layers


/// things to remember:
/// normalize training data, so that the activation function can easily distinct different inputs !!!(the network works with 0-1 or +/-)
/// after having the network trained once, if you want to do it the second time, randomize it (because if not done, it will still think the same way as before)

using namespace std;

/// width of a layer = number of neurons in a layer

const float e = 2.71828;
const int layers = 5;               /// number of layers except for the input one
const int width[layers] = {4, 4, 4, 4, 2};   /// width of each layer
const int inputWidth = 2;           /// width of the input layer
const float learningRate = 0.2;     /// of the weights
const float biasLearningRate = 0.2; /// because the biases sometimes may seem to be escalating too much at first (depending on data), just for the flexibility

class Neuron {
    private:
        ///activation functions and their derivative calculations
        double Sigmoid() {              /// sigmoid(x) = 1 / (1 + e^(-x))
            return 1 / (1 + exp(-input));
        }
        double SigmoidDerivative() {
            return output * (1 - output); /// assuming the output is already sigmoid
        }
        double Swish() {                /// swish(x) = x * sigmoid(x)
            return input * Sigmoid();
        }
        double SwishDerivative() {
            return Sigmoid() * (input * (1 - Sigmoid()) + 1); /// assuming the output is already swish
        }

    public:
        double input;                /// the sum of all the neurons multiplied by the biases from the previous layer
        double output;               /// the input converted by an activation function
        double costDerivative;       /// the value of the derivative of the cost function with respect to this particular neuron
                                                                                            ///(how much it influences the error)
        double activationDerivative; /// the value of the activation function's derivative
        int activationFunction;      /// type of an activation function (1 - sigmoid, 2 - swish)

        void Activate() {
            if(activationFunction == 1) {
                output = Sigmoid();
            }else if(activationFunction == 2) {
                output = Swish();
            }
        }
        void CalculateActivationDerivative() {
            if(activationFunction == 1) {
                activationDerivative = SigmoidDerivative();
            }else if(activationFunction == 2) {
                activationDerivative = SwishDerivative();
            }
        }
        void Clear() {
            input = 0;
            output = 0;
            costDerivative = 0;
            activationDerivative = 0;
        }

        Neuron(int activation = 1) { /// type of an activation function (1 - sigmoid, 2 - swish), sigmoid is default
            activationDerivative = 0;
            activationFunction = activation;
            input = 0;
            output = 0;
            costDerivative = 0;
        }
};

class Neurons {
    public:
        vector<float> input;            /// the input ones do not need to be the neurons, just values
        vector<vector<Neuron> > main;   /// all of the hidden neurons + output neurons

        Neurons(int activationFunction = 1, int outputActivationFunction = 1) { /// constructor (just creates empty arrays with activation function)
            vector<float> a(inputWidth);
            input = a;
            Neuron n(activationFunction);
            Neuron s(outputActivationFunction);                            /// the output is said to be sigmoid
            vector<Neuron> b;
            vector<vector<Neuron> > c;
            for(int l = 0; l < layers - 1; l++) {   /// assign activation functions to individual neurons
                c.push_back(b);
                for(int i = 0; i < width[l]; i++) {
                    c[l].push_back(n);
                }
            }
            c.push_back(b);
            for(int i = 0; i < width[layers - 1]; i++) {
                c[layers - 1].push_back(s);
            }
            main = c;
        }
};

class Weights {
    public:
        vector<vector<vector<float> > > main;     /// each neuron in layer n is connected to all the neurons in the layer n + 1
                                                                                                ///(except for the output ones)

        /// main[l][j][k] means that the j'th neuron from the l'th hidden layer is connected to the k'th neuron in the l-1'th layer (so it works backwards)
        vector<vector<vector<float> > > gradient; /// each weight has its gradient calculated (can be added when back-propagating)

        void Randomize() {  /// randomization of the weights' values and plugging them into the "weights.txt" file
            ofstream file("./weights.txt");
            if(!file) {
                cout<<"Error";
            }else {
                for(int j = 0; j < width[0]; j++) {
                    int s = inputWidth;
                    for(int k = 0; k < s; k++) {
                        main[0][j][k] = sqrt(1 / (double)(s)) * (double)(rand()%s - s/2) / (double)(s/2);
                        file<<main[0][j][k]<<" ";
                    }
                    file<<endl;
                }
                file<<endl;
                for(int l = 1; l < layers; l++) {
                    for(int j = 0; j < width[l]; j++) {
                        int s = width[l - 1];
                        for(int k = 0; k < s; k++) {
                            main[l][j][k] = sqrt(1 / (double)(s)) * (double)(rand()%s - s/2) / (double)(s/2);
                            file<<main[l][j][k]<<" ";
                        }
                        file<<endl;
                    }
                    file<<endl;
                }
                file.close();
            }
            return;
        }
        void Fill() {       /// fills all of the weights' values from the "weights.txt" file
            ifstream file("./weights.txt");
            if(!file) {
                cout<<"Error";
            }else {
                for(int j = 0; j < width[0]; j++) {
                    for(int k = 0; k < inputWidth; k++) {
                        file>>main[0][j][k];
                    }
                }
                for(int l = 1; l < layers; l++) {
                    for(int j = 0; j < width[l]; j++) {
                        for(int k = 0; k < width[l - 1]; k++) {
                            file>>main[l][j][k];
                        }
                    }
                }
                file.close();
            }
            return;
        }
        void UpdateFile() { /// updates all of the weights' values in the "weights.txt" file from the current values
            ofstream file("./weights.txt");
            if(!file) {
                cout<<"Error";
            }else {
                for(int j = 0; j < width[0]; j++) {
                    for(int k = 0; k < inputWidth; k++) {
                        file<<main[0][j][k]<<" ";
                    }
                    file<<endl;
                }
                file<<endl;
                for(int l = 1; l < layers; l++) {
                    for(int j = 0; j < width[l]; j++) {
                        for(int k = 0; k < width[l - 1]; k++) {
                            file<<main[l][j][k]<<" ";
                        }
                        file<<endl;
                    }
                    file<<endl;
                }
                file.close();
            }
            return;
        }

        Weights() { ///constructor (just creates empty arrays with the given widths)
            vector<vector<vector<float> > > a;
            a.resize(layers);
            a[0].resize(width[0]);
            for(int j = 0; j < width[0]; j++) {
                a[0][j].resize(inputWidth);
            }
            for(int l = 1; l < layers; l++) {
                a[l].resize(width[l]);
                for(int j = 0; j < width[l]; j++) {
                    a[l][j].resize(width[l - 1]);
                }
            }
            gradient = a;
            main = a;
        }
};

class Biases {
    public:
        vector<vector<float> > main;     /// each neuron has its bias (except for the input ones)
        vector<vector<float> > gradient; /// each bias has its gradient calculated (can be added when back-propagating)

        void Randomize() {  /// randomization of the biases' values and plugging them into the "biases.txt" file
            ofstream file("./biases.txt");
            if(!file) {
                cout<<"Error";
            }else {
                for(int l = 0; l < layers; l++) {
                    for(int j = 0; j < width[l]; j++) {
                        //float r = rand() - rand();
                        //main[l][j] = r/100000;
                        main[l][j] = 0; /// start with no biases
                        file<<main[l][j]<<" ";
                    }
                    file<<endl;
                }
                file.close();
            }
            return;
        }
        void Fill() {       /// fills all of the biases' values from the "biases.txt" file
            ifstream file("./biases.txt");
            if(!file) {
                cout<<"Error";
            }else {
                for(int l = 0; l < layers; l++) {
                    for(int j = 0; j < width[l]; j++) {
                        file>>main[l][j];
                    }
                }
                file.close();
            }
            return;
        }
        void UpdateFile() { /// updates all of the biases' values in the "biases.txt" file from the current values
            ofstream file("./biases.txt");
            if(!file) {
                cout<<"Error";
            }else {
                for(int l = 0; l < layers; l++) {
                    for(int j = 0; j < width[l]; j++) {
                        file<<main[l][j]<<" ";
                    }
                    file<<endl;
                }
                file.close();
            }
            return;
        }

        Biases() { /// constructor (just creates empty arrays with given widths)
            vector<vector<float> > a;
            a.resize(layers);
            for(int l = 0; l < layers; l++) {
                a[l].resize(width[l]);
            }
            gradient = a;
            main = a;
        }
};

class BatchData {
    private: /// these values will help us access the Network's values more easily
        int inputSize;
        int outputSize;
        int batchSize;
        string directory;
    public:
        vector<vector<float> > input;         /// all the inputs from the batch
        vector<vector<float> > desiredOutput; /// all the desired outputs to each input from the batch

        void CollectBatchFromExample(int start) { /// collect the batch starting at the 'int start' line
            ifstream trainingData(directory.c_str());
            if(!trainingData) {
                cout<<"Error. Could not find the training data"<<endl;
                return;
            }
            while(start > 0) { /// get to the start line
                string useless;
                getline(trainingData, useless);
                start--;
            }
            for(int i = 0; i < batchSize; i++) { /// collect the batch data
                for(int j = 0; j < inputSize; j++) {
                    trainingData>>input[i][j]; /// collect inputs
                }
                for(int j = 0; j < outputSize; j++) {
                    trainingData>>desiredOutput[i][j]; /// collect desired outputs
                }
            }
            trainingData.close();
        }
        BatchData(int batchS = 0, int inSize = 0, int outputWidth = 0, string path = "") {
            directory = path;
            batchSize = batchS;
            inputSize = inSize;
            outputSize = outputWidth;
            desiredOutput.resize(batchSize);
            input.resize(batchSize);
            for(int i = 0; i < batchSize; i++) {
                input[i].resize(inputSize);
                desiredOutput[i].resize(outputSize);
            }
        }
};

class Network {
    private:
        int batchSize;
        int batchIterations;
        vector<float> desiredOutput; /// the output that we truly desire ;)
        vector<float> cost;          /// the activation cost (imagine it is squared but actually it's not, because we just need its derivative which is linear)
        Neurons neurons;
        Weights weights;
        Biases biases;
        BatchData data;

        void Activate() {                       /// activate the network
            for(int j = 0; j < width[0]; j++) { /// take values from the input layer
                for(int k = 0; k < inputWidth; k++) {
                    neurons.main[0][j].input += neurons.input[k] * weights.main[0][j][k];
                }
                neurons.main[0][j].input += biases.main[0][j];
                neurons.main[0][j].Activate(); /// calculates the neuron's output
            }
            for(int l = 1; l < layers; l++) { /// fill the rest of the network
                for(int j = 0; j < width[l]; j++) {
                    for(int k = 0; k < width[l - 1]; k++) {
                        neurons.main[l][j].input += neurons.main[l - 1][k].output * weights.main[l][j][k];
                    }
                    neurons.main[l][j].input += biases.main[l][j];
                    neurons.main[l][j].Activate(); /// calculates the neuron's output
                }
            }
        }
        void ClearNeurons() {                   /// clear all the neurons (without gradients)
            for(int l = 0; l < layers; l++) {
                for(int j = 0; j < width[l]; j++) {
                    neurons.main[l][j].Clear();
                }
            }
        }
        void ClearAll() {                       /// clear everything
            for(int j = 0; j < width[layers - 1]; j++) {
                desiredOutput[j] = 0;
            }
            for(int j = 0; j < width[layers - 1]; j++) {
                cost[j] = 0;
            }
            for(int j = 0; j < width[0]; j++) {
                neurons.main[0][j].Clear(); /// including the gradient
                biases.gradient[0][j] = 0;
                for(int k = 0; k < inputWidth; k++) {
                    weights.gradient[0][j][k] = 0;
                }
            }
            for(int l = 1; l < layers; l++) {
                for(int j = 0; j < width[l]; j++) {
                    neurons.main[l][j].Clear();
                    biases.gradient[l][j] = 0;
                    for(int k = 0; k < width[l - 1]; k++) {
                        weights.gradient[l][j][k] = 0;
                    }
                }
            }
        }
        void CalculateCost() {                  /// the cost is negative when the output is too large and positive when too small
            for(int i = 0; i < width[layers - 1]; i++) {
                cost[i] += desiredOutput[i] - neurons.main[layers - 1][i].output;
            }
        }
        void CalculateActivationDerivatives() { /// calculate the activation derivatives of all the neurons
            for(int l = 0; l < layers; l++) {
                for(int j = 0; j < width[l]; j++) {
                    neurons.main[l][j].CalculateActivationDerivative();
                }
            }
        }
        void CalculateGradient() {              /// calculate gradient on one activation for each weight and bias
            CalculateActivationDerivatives();
            for(int j = 0; j < width[layers - 1]; j++) { /// calculate cost derivative with respect to the last layer (output)
                neurons.main[layers - 1][j].costDerivative = 2 * cost[j];
            }

            for(int l = layers - 1; l > 0; l--) { /// calculate gradient for the middle layers

                for(int j = 0; j < width[l]; j++) {
                        ///the derivative of the cost with respect to the neuron[j]
                    float a = neurons.main[l][j].costDerivative * neurons.main[l][j].activationDerivative;
                    biases.gradient[l][j] += a; /// cost derivative * the previous neuron (in case of biases it's always one)

                    for(int k = 0; k < width[l - 1]; k++) {
                            /// the derivative of the cost[j] with respect to the weight linking the k'th neuron
                        weights.gradient[l][j][k] += a * neurons.main[l - 1][k].output;
                            /// the derivative of the cost[j] with respect to the k'th neuron from the previous layer
                        neurons.main[l - 1][k].costDerivative += a * weights.main[l][j][k]; /// the last's neuron derivative * the linking weight
                    }
                }
            }

            for(int j = 0; j < width[0]; j++) { /// calculate gradient for the last layer (because it needs the input values it's not among the middle ones)
                    ///the derivative of the cost with respect to the neuron[j]
                float a = neurons.main[0][j].costDerivative * neurons.main[0][j].activationDerivative;
                biases.gradient[0][j] += a; /// cost derivative * the previous neuron (in case of biases it's always one)

                for(int k = 0; k < inputWidth; k++) {
                        /// the derivative of the cost[j] with respect to the weight linking the k'th neuron
                    weights.gradient[0][j][k] += a * neurons.input[k]; /// the last's neuron derivative * the previous neuron (in this case the input one)
                }
            }
        }
        void BackPropagate() {                  /// update weights and biases based on their gradients and learningRate
            for(int j = 0; j < width[0]; j++) { /// update weights and biases from the first non-input layer
                biases.main[0][j] += biases.gradient[0][j] * biasLearningRate;
                for(int k = 0; k < inputWidth; k++) {
                    weights.main[0][j][k] += weights.gradient[0][j][k] * learningRate;
                }
            }

            for(int l = 1; l < layers; l++) { /// update the remaining weights and biases
                for(int j = 0; j < width[l]; j++) {
                    biases.main[l][j] += biases.gradient[l][j] * biasLearningRate;
                    for(int k = 0; k < width[l - 1]; k++) {
                        weights.main[l][j][k] += weights.gradient[l][j][k] * learningRate;
                    }
                }
            }
        }
        void TrainOnExample() {                 /// train the network on one example
            Activate();
            CalculateCost();
            CalculateGradient();
            ClearNeurons();
        }
        void TrainOnBatch() {                   /// calculate gradients for all the examples from the batch combined
            for(int i = 0; i < batchSize; i++) {
                desiredOutput = data.desiredOutput[i];
                neurons.input = data.input[i];
                TrainOnExample();
            }
            BackPropagate();
        }

    public:
        void Randomize() {  /// randomize all the weights and biases and plug them into the files
            weights.Randomize();
            biases.Randomize();
        }
        void UpdateFile() { /// update the files containing current weights and biases
            weights.UpdateFile();
            biases.UpdateFile();
        }
        void RepeatTraining() {
            data.CollectBatchFromExample(0);
            for(int i = 0; i < batchIterations; i++) {          /// if we do not want to debug (and not to check if we want every time in a for)
                TrainOnBatch();
                ClearAll();
            }

        }
        void ShowResults() {
            TrainOnBatch();
            DebugCost();
            ClearAll();
        }
        void DebugCost(int iteration = 0) {
            cout<<"-------------- ITERATION "<<iteration<<" --------------"<<endl;
            cout<<"------------------------------------------"<<endl;
            ClearAll();
            for(int i = 0; i < batchSize; i++) {
                desiredOutput = data.desiredOutput[i];
                neurons.input = data.input[i];
                Activate();
                CalculateCost();
                cout<<"On the "<<i<<" training example: "<<endl;
                for(int j = 0; j < width[layers - 1]; j++) {
                    cout<<"["<<j<<"]"<<" cost: "<<cost[j]<<setw(10)<<desiredOutput[j]<<" - "<<neurons.main[layers - 1][j].output<<endl;
                }
                ClearAll();
            }
        }

        Network(int outputActivationFunction = 1, int activationFunction = 1, string directory = "", int _batchIterations = 50, int _batchSize = 0) {
            batchSize = _batchSize;
            batchIterations = _batchIterations;
            BatchData d(batchSize, inputWidth, width[layers - 1], directory);
            data = d;
            Neurons a(activationFunction, outputActivationFunction);
            cost.resize(width[layers - 1]);
            neurons = a;
            biases.Fill();
            weights.Fill();
        }
};


int main()
{
    srand(time(NULL));

    Network n(1, 1, "./trainingData.txt", 10000, 4);
    n.Randomize();
    n.RepeatTraining();
    n.ShowResults();
    n.UpdateFile();
    return 0;
}
