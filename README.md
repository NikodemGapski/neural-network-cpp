# neural-network-cpp
C++ implementations of different neural network approaches (currently just feed-forward, backpropagated network based on layers).

In StaticTemplate folder you can find implementation for feedforward, backpropagated network.
Its neurons are grouped in layers. Size of each layer, as well as activation function for both hidden and output neurons can be easily set when initializing an instance of a Network class.
  Available activation funcitons:
  - Sigmoid,
  - Tanh,
  - ReLU,
  - ELU,
  - Swish.

Network is trained on examples in trainingData.txt file by backpropagation with added momentum.

Weights and biases are stored and can be accessed from files weights.txt and biases.txt accordingly.

## Disclaimer

This implementation was created for educational purposes only. It is not optimised, so training the network on large data may not be feasible.

## License

This project is under MIT License.