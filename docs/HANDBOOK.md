# Activation Playground Handbook

This handbook explains how to use the project as a learning and experimentation tool.

## 1. What This Tool Is

The project is a compact playground for small multilayer perceptrons with editable activation layouts.

It lets you:

- inspect how inputs move through a network
- change activation functions per layer or per neuron
- train a model and inspect loss and accuracy
- compare layouts manually
- optimize layouts with simulated annealing
- run reproducible multi-seed experiments

The same core model and benchmark pipeline is reused across all modes.

## 2. The Three GUI Modes

### Demo Mode

Use Demo Mode when you want to understand one concrete network.

Typical use:

- choose a benchmark
- inspect one visible sample
- change activations
- train step by step
- click neurons and inspect local computations

Best for:

- first contact with the project
- understanding forward and backward behavior
- understanding what activations do

### Playground Mode

Use Playground Mode when you want to see one simulated annealing run in detail.

Typical use:

- define a start layout
- choose an objective
- choose neighborhood moves
- configure temperature and cooling
- run simulated annealing step by step

Best for:

- understanding state, neighbor, temperature, and acceptance
- comparing start layout, candidate, current state, and best state

### Experiment Builder

Use Experiment Builder when you want reproducible experiments.

Typical use:

- define benchmark, hidden sizes, and start layout
- choose run mode: `manual_training` or `simulated_annealing`
- choose seeds
- optionally define a grid or random search space
- run and store results
- inspect aggregated multi-seed results

Best for:

- fair comparisons
- repeated runs with several seeds
- ranking configurations by validation metrics
- storing and reloading results later

## 3. Core Concepts

### Benchmark

A benchmark is the dataset and split configuration used for training and evaluation.

Available benchmarks:

- `breast_cancer`
- `wine`
- `digits`
- `test_activation`

### Hidden Layer

A hidden layer is a layer between input and output.

More hidden neurons or more hidden layers increase model capacity, but they also increase complexity and can make interpretation harder.

### Layout

A layout describes which activation function each hidden neuron uses.

Examples:

- `relu|relu`
- `relu*16|tanh*8`
- `relu*8,tanh*8|sigmoid*4,leaky_relu*4`

In the GUI, the layout can be edited:

- per layer
- per neuron
- or directly as a layout string

### Sample

A sample is one concrete input example.

In Demo and Playground, one visible sample is used for inspection:

- prediction
- target
- class probabilities
- hidden activations
- local neuron calculations

Important:

The visible sample is not the whole training process. It is only an analysis window.

### Seed

A seed controls randomness.

It affects:

- data splitting
- weight initialization
- candidate training reproducibility
- random search sampling

In the Experiment Builder, multiple seeds are important because one single run may be misleading.

### Run

A run is one concrete execution with:

- one benchmark
- one parameter configuration
- one seed

### Configuration

A configuration is one fixed set of parameter values.

For example:

- one learning rate
- one batch size
- one epoch count
- one candidate epoch count
- one start temperature

### Experiment

An experiment is a collection of runs produced from:

- one benchmark
- one architecture
- one start layout
- one run mode
- one seed strategy
- optionally one search space

## 4. Training in This Project

Training uses mini-batch gradient descent.

At a high level:

1. the benchmark is loaded and split into train / validation / test
2. the model performs a forward pass
3. the loss is computed
4. gradients are computed by backpropagation
5. weights and biases are updated
6. train and validation metrics are stored
7. final test metrics are measured at the end

Important:

- training changes weights and biases
- training does not change the activation layout
- in simulated annealing, each candidate layout is trained from scratch for a short budget

## 5. Simulated Annealing in This Project

In this repository, simulated annealing optimizes the activation layout.

### State

One state is one concrete activation layout.

### Neighbor

A neighbor is a small layout modification.

Possible neighborhood operations:

- change one neuron
- fill one full layer
- swap two neuron activations

### Objective

The objective is currently based on validation metrics:

- `validation_loss`
- `validation_accuracy`

Internally, the search compares candidates in a consistent way so that it can decide whether one state is better or worse.

### Temperature

Temperature controls how willing the search is to accept worse states.

- high temperature: more exploration
- low temperature: more conservative search

### Cooling

Cooling controls how temperature decreases over time.

Available schedules:

- geometric
- linear
- logarithmic

### Start Layout, Best Layout, End Layout

- Start layout: the layout before the SA run begins
- Best layout: the best layout found so far
- End layout: the layout where the run stops

## 6. Why Validation Matters

Validation is used for:

- ranking configurations
- choosing best configurations
- comparing runs fairly

Test should remain the final evaluation only.

If you choose by test metrics, you leak final evaluation information into the model-selection process.

## 7. Parameter Guide

### Learning Rate

Controls the step size of weight updates.

- too small: learning is slow
- too large: training can become unstable

Typical first values:

- `0.001`
- `0.01`
- `0.03`
- `0.1`

### Batch Size

Controls how many samples are processed together per update.

- small batches: noisier but often more sensitive
- large batches: smoother but sometimes less responsive

Typical first values:

- `8`
- `16`
- `32`
- `64`

### Weight Scale

Controls the magnitude of the initial random weights.

- too small: weak early signals
- too large: overly strong early activations

Typical first values:

- `0.01`
- `0.03`
- `0.05`
- `0.1`

### Epochs

Number of full passes over the training split.

- too few: undertrained
- too many: more runtime, possible overfitting

### Candidate Epochs

In simulated annealing, this is the training budget per candidate layout.

- too few: candidate score can be noisy
- more: fairer comparison, but slower search

### Start Temperature

Controls how exploratory SA is at the beginning.

- low: conservative
- high: more willingness to accept worse states

### Cooling Parameter

Controls how strongly temperature decreases.

Its meaning depends on the chosen schedule.

### Iterations per Temperature

How many SA steps are executed before the next cooling step is applied.

### Max Steps

Hard upper bound on run length.

### Min Temperature

Stop threshold once the search has cooled enough.

## 8. Suggested First Workflows

### Demo Mode

1. start with `wine`
2. keep hidden sizes small
3. inspect one sample
4. compare `relu` and `tanh`
5. train step by step

### Playground Mode

1. start with `wine`
2. use a simple start layout like `relu|relu`
3. choose `validation_loss`
4. use `candidate_epochs = 10`
5. use geometric cooling
6. inspect start, candidate, current, and best states

### Experiment Builder

1. start with `manual_training`
2. use 5 seeds
3. compare two small configurations
4. rank by validation metrics
5. only then move to simulated annealing or search spaces

## 9. Result Files

Stored experiments create:

- one `manifest.json`
- one `summary.json`
- one run JSON per run

Typical location:

`outputs/experiments/<experiment_id>/`

This allows:

- later reloading
- later plotting
- later multi-seed analysis

## 10. Practical Advice

- do not overinterpret a single run
- compare configurations with the same seeds
- use validation for model selection
- use test only for final reporting
- keep the first experiments small and readable
- if simulated annealing feels random, increase `candidate_epochs` before changing everything else

## 11. Where To Continue

For code structure and data flow, see:

- `docs/ARCHITECTURE.md`

For guided experiments, see:

- `docs/EXPERIMENT_RECIPES.md`
