# Activation Playground

Base project for the seminar **Natural Computing** at the **Paris Lodron University Salzburg**.

This repository is a compact playground for small neural networks with editable activation layouts. It combines a didactic demo interface with a simulated annealing playground on top of the same benchmark, model, and visualization pipeline.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
mkdir -p .mplconfig
export MPLCONFIGDIR=.mplconfig
python main.py
```

This starts the interactive terminal assistant.

Start the GUI directly:

```bash
python main.py --gui
```

Start the GUI in simulated annealing mode:

```bash
python main.py --gui --gui-app-mode playground
```

Show CLI help:

```bash
python main.py --help
```

Show layout and neighbor syntax help:

```bash
python main.py --show-layout-help
```

## Current Capabilities

- small MLPs with **1 to 4 hidden layers**
- activation functions assigned per layer or per individual neuron
- supported activations: `relu`, `tanh`, `sigmoid`, `leaky_relu`
- benchmarks: `breast_cancer`, `wine`, `digits`, `test_activation`
- training with **train / validation / test**, **loss**, and **accuracy**
- layout editing, local neighbor operations, and layout diffs
- terminal ASCII views and Matplotlib plots
- bilingual GUI in **German** and **English**
- `Demo Mode` for understanding data flow, activations, training, and neuron behavior
- `Playground Mode` for **Simulated Annealing** over activation layouts

## GUI Workspaces

### Demo Mode

```bash
python main.py --gui --gui-app-mode demo
```

Demo Mode focuses on understanding one concrete network:

- dataset and sample inspection
- network visualization with neurons and connections
- activation colors per hidden neuron
- clickable neuron inspection with local computations
- training plots
- forward/backward stepper
- baseline vs. current comparison

### Playground Mode

```bash
python main.py --gui --gui-app-mode playground
```

Playground Mode adds a real simulated annealing workflow:

- activation layout as optimization state
- configurable neighborhood operations
- objective selection: `validation_loss` or `validation_accuracy`
- cooling schedules: `geometric`, `linear`, `logarithmic`
- visible current state, candidate, best state, and layout diffs
- score, temperature, acceptance probability, and acceptance rate plots
- didactic explanations for acceptance and rejection decisions

## CLI Examples

Standard training run:

```bash
python main.py --benchmark wine --hidden-sizes 16 8 --layout "relu|tanh" --epochs 120
```

Expert GUI:

```bash
python main.py --gui --gui-mode expert
```

English Playground Mode:

```bash
python main.py --gui --gui-app-mode playground --gui-language en --gui-mode expert
```

Save plots from a CLI run:

```bash
python main.py --benchmark digits --layout "relu|relu" --epochs 40 --save-prefix demo/digits_run
```

## Benchmarks

### `breast_cancer`

- 30 numerical input features
- 2 classes
- compact binary classification benchmark

### `wine`

- 13 numerical input features
- 3 classes
- useful for comparing activation layouts in a small multiclass setting

### `digits`

- 64 inputs from 8x8 grayscale digit images
- 10 classes
- especially useful for visible input, target, and prediction flow

### `test_activation`

- artificial mini benchmark with 3 inputs
- intended for raw forward computations and activation comparisons

## Layout Syntax

A layout describes the activation functions of all hidden layers.

```text
relu|relu
relu|tanh|sigmoid
relu*16|tanh*8
relu*5,tanh*5|sigmoid*4
```

Rules:

- `|` separates hidden layers
- `,` separates activation groups inside one layer
- `relu*16` means 16 neurons with `relu`

## Neighbor Syntax

Neighbor operations modify a layout locally.

```text
set:L1:0:sigmoid
fill:L2:tanh
cycle:L1:3
swap:L2:1:4
```

These operations are used both for manual experiments and for the simulated annealing search space.

## Project Structure

- `main.py`: CLI, interactive assistant, program entry point
- `benchmarks.py`: dataset loading and splitting
- `activations.py`: activation functions, layouts, neighbor logic
- `model.py`: MLP, forward pass, backpropagation, neuron inspection
- `trainer.py`: training loop and metrics
- `plotting.py`: Matplotlib plots
- `terminal_viz.py`: ASCII terminal views
- `gui.py`: interactive GUI
- `configs.py`: shared defaults and configuration
- `annealing.py`: simulated annealing state and acceptance logic
- `annealing_schedules.py`: cooling schedules
- `annealing_objectives.py`: layout evaluation objectives
- `annealing_runner.py`: simulated annealing execution and history

## Current Scope

This is the first base project of the repository. It already covers model building, layout editing, visualization, training, and simulated annealing over activation layouts, while keeping the codebase modular enough for further optimization methods later on.
