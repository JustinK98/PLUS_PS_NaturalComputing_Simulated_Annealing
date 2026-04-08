# Experiment Recipes

These recipes are meant as concrete starting points.

They are not the only possible experiments, but they are good first uses of the tool.

## Recipe 1: ReLU vs Tanh on `wine`

### Goal

Compare two simple activation layouts on a compact multiclass dataset.

### Suggested setup

- benchmark: `wine`
- hidden sizes: `16 / 8`
- seeds: `5`
- mode: `manual_training`
- layout A: `relu*16|relu*8`
- layout B: `tanh*16|tanh*8`
- epochs: `50` to `120`
- learning rate: `0.03`

### What to observe

- mean validation accuracy
- standard deviation across seeds
- class probability profiles
- whether one layout converges faster or more stably

### Interpretation

If one layout wins consistently across several seeds, that is much stronger evidence than one good single run.

## Recipe 2: Mixed Layout vs Homogeneous Layout on `digits`

### Goal

See whether a mixed hidden activation layout changes prediction behavior on image-like data.

### Suggested setup

- benchmark: `digits`
- hidden sizes: `32 / 16`
- seeds: `5`
- mode: `manual_training`
- layout A: `relu*32|relu*16`
- layout B: `relu*16,tanh*16|relu*8,sigmoid*8`
- epochs: `50` to `80`

### What to observe

- mean validation accuracy
- mean test accuracy
- uncertainty patterns on difficult digits
- class probability changes

### Interpretation

The interesting part is not only final accuracy, but also whether uncertainty is distributed differently.

## Recipe 3: Simulated Annealing from a Homogeneous ReLU Start

### Goal

Use SA to improve a simple start layout.

### Suggested setup

- benchmark: `wine`
- hidden sizes: `16 / 8`
- mode: `simulated_annealing`
- start layout: `relu*16|relu*8`
- objective: `validation_loss`
- candidate epochs: `10`
- start temperature: `1.0`
- cooling: `geometric`
- cooling parameter: `0.85`
- iterations per temperature: `3`
- max steps: `20`
- seeds: `3` to `5`

### What to observe

- start layout vs best layout
- best objective
- acceptance rate
- how quickly temperature reduces exploration

### Interpretation

Look at whether best layouts improve validation consistently, not just whether one run found a visually different layout.

## Recipe 4: Multi-Seed Stability Check

### Goal

Measure whether one promising configuration is actually stable.

### Suggested setup

- choose one benchmark and one layout
- run mode: `manual_training`
- seeds: at least `5`
- no search space

### What to observe

- mean validation accuracy
- standard deviation
- best seed vs worst seed

### Interpretation

A configuration with slightly lower mean but much lower variance can be more useful than a configuration with one lucky high result.

## Recipe 5: `test_activation` as a Computation Lab

### Goal

Understand the raw effects of different activation functions in a tiny network.

### Suggested setup

- benchmark: `test_activation`
- hidden sizes: `4 / 3`
- use custom sample
- inspect several layouts:
  - `relu,tanh,sigmoid,leaky_relu|relu,tanh,sigmoid`
  - `relu*4|relu*3`
  - `sigmoid*4|sigmoid*3`

### What to observe

- local neuron equations
- weighted sum `z`
- activation output `a`
- how small input changes affect the visible computation

### Interpretation

This recipe is about understanding computation, not about reaching a high final accuracy.
