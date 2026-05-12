# Abstract

This project investigates whether the performance of small neural networks can be improved by optimizing the distribution of activation functions across hidden neurons.

In standard neural network training, gradient descent updates the weights and biases while activation functions are usually selected manually before training and then kept fixed. In this project, these two aspects are separated: gradient descent trains the weights, while Simulated Annealing searches over discrete activation-function layouts.

A layout defines which activation function, such as ReLU, GELU, sigmoid, tanh, Swish, or identity, is assigned to each hidden neuron. During the search, Simulated Annealing proposes local layout changes, for example replacing the activation function of one neuron or changing a complete layer. To evaluate such a change, the network loss is measured before and after the activation-function modification using the same weights and the same mini-batch, without retraining the candidate layout first. If the candidate is accepted, the network is then trained on that mini-batch and the search continues from the updated network state.

This online-delta approach follows a continuously trained search process: the network is not restarted for every candidate, but evolves through the search while its weights are gradually improved. This reduces the noise introduced by short retraining runs and focuses the acceptance decision on the immediate effect of an activation-function change.

After the search, selected layouts are compared under identical conditions, including the same benchmark, topology, training budget, data split, and random seeds. These layouts include the start layout, the best layout found by Simulated Annealing, the final layout after the search, random layouts, and homogeneous single-activation baselines. A key comparison is whether the inherited network produced during the search behaves differently from the same activation layout trained again from scratch.

The goal is to study whether searching over activation-function distributions can support neural network design and whether Simulated Annealing can find layouts that are competitive with or better than manually chosen activation-function configurations.
