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

## 2. The GUI Modes

### Presentation Mode

Use Presentation Mode when you want to explain the project live without exposing all expert controls.

Typical use:

- choose `Neural Network Preview` for a 5-7 minute explanation of one concrete model
- choose `Simulated Annealing Preview` for a 5-7 minute explanation of the search process
- click through the slides from problem statement to takeaway

Best for:

- classroom demos
- oral presentations
- first contact before opening Demo or Playground Mode

Presentation Mode uses fixed `concentric_circles` defaults so that the visual story remains stable.

For the prepared talk flow, use [PRESENTATION_SCRIPT.md](PRESENTATION_SCRIPT.md). It contains slide-by-slide speaker text, click instructions, and a fallback route for the live demo.

### Activation Workflow

Use Activation Workflow when you want the complete project flow in one window.

Typical use:

- load one official CSV benchmark
- choose or edit an activation layout
- train that layout directly
- optionally run simulated annealing over layouts
- finally retrain and compare start, best, end, random, and homogeneous baseline layouts under identical conditions

Best for:

- explaining the actual project pipeline
- comparing manual layout choices against SA-found layouts
- producing the clearest result tables for discussion

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

Official benchmarks:

- `concentric_circles`: 2 inputs, 2 classes, topology `2-8-1`, 100 epochs
- `iris`: 4 inputs, 3 classes, topology `4-8-3`, 150 epochs
- `crossing_spirals`: 6 inputs, 2 classes, topology `6-16-16-1`, 250 epochs

`test_activation` remains an internal computation lab, not an official benchmark for reports.

### Hidden Layer

A hidden layer is a layer between input and output.

More hidden neurons or more hidden layers increase model capacity, but they also increase complexity and can make interpretation harder.

### Layout

A layout describes which activation function each hidden neuron uses.

Examples:

- `relu|relu`
- `relu*16|tanh*8`
- `relu*8,tanh*8|sigmoid*4,swish*4`

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

1. start with `concentric_circles`
2. keep hidden sizes small
3. inspect one sample
4. compare `relu` and `tanh`
5. train step by step

### Playground Mode

1. start with `concentric_circles`
2. use a simple start layout like `relu`
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

## 11. Guided Experiment Recipes

### Recipe 1: ReLU vs Tanh on `iris`

Goal:

- compare two simple activation layouts on the official medium multiclass benchmark

Suggested setup:

- benchmark: `iris`
- hidden sizes: `8`
- seeds: `5`
- mode: `manual_training`
- layout A: `relu*8`
- layout B: `tanh*8`
- epochs: `50` to `120`
- learning rate: `0.03`

What to watch:

- mean validation accuracy
- standard deviation across seeds
- probability profiles per class
- convergence speed and stability

### Recipe 2: Mixed vs Homogeneous Layout on `crossing_spirals`

Goal:

- see whether mixed hidden activations help on the official hard benchmark

Suggested setup:

- benchmark: `crossing_spirals`
- hidden sizes: `16 / 16`
- seeds: `5`
- mode: `manual_training`
- layout A: `relu*16|relu*16`
- layout B: `relu*8,tanh*8|gelu*8,swish*8`

What to watch:

- mean validation accuracy
- test accuracy
- final validation and test accuracy
- whether SA-found layouts beat homogeneous baselines

### Recipe 3: Simulated Annealing from a Homogeneous ReLU Start

Goal:

- use SA to improve a simple baseline layout

Suggested setup:

- benchmark: `concentric_circles`
- hidden sizes: `8`
- mode: `simulated_annealing`
- start layout: `relu*8`
- objective: `validation_loss`
- candidate epochs: `10`
- start temperature: `1.0`
- cooling: `geometric`
- cooling parameter: `0.85`
- iterations per temperature: `3`
- max steps: `20`
- seeds: `3` to `5`

What to watch:

- start layout vs best layout
- acceptance rate
- temperature drop
- whether best really improves validation metrics

### Recipe 4: Multi-Seed Stability Check

Goal:

- test whether one promising configuration is actually stable

Suggested setup:

- choose one benchmark and one layout
- run mode: `manual_training`
- seeds: at least `5`
- no search space

What to watch:

- mean validation accuracy
- standard deviation
- best seed vs worst seed

### Recipe 5: `test_activation` as Computation Lab

Goal:

- understand raw activation behavior in a tiny network

Suggested setup:

- benchmark: `test_activation`
- hidden sizes: `4 / 3`
- use custom sample
- compare layouts such as:
  - `relu,tanh,sigmoid,gelu|relu,tanh,sigmoid`
  - `relu*4|relu*3`
  - `sigmoid*4|sigmoid*3`

What to watch:

- local neuron equations
- weighted sum `z`
- activation output `a`
- how small input changes affect visible computation

## 12. Where To Continue

For code structure and data flow, see:

- `docs/ARCHITECTURE.md`
