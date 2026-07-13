


Abstract
(all) 

(all) 
Introduction 
(half page- 1 page) 

( Ziri + Negar connect) --> With questions to Justin if neccessary)
Background
(Ziri) 
--> Introcuction into SA 
--> Introduction into NNs
--> Introduction into AFs 
(forumals, simple explainable, and Ground Sources Basic Machine Learning Literture) 
(2 Pages) 

Related Work + Methodology (Distinguish from other approaches) 
(Negar)
--> Simulated Annealing how it is being used and when?
--> Is there a paper where they distiguish layout designs?
--> Is there a Paper where something similar to online Delta is being introduced? 
--> The role of activation functions for certain benchmarks or problems
...
How may we arrange to use that conclusions out of that papers into our project? 
(1-2 Pages) 

(Justin + Patrick Connect with Questions to Ziri + Negar if neccesary)
Online Delta 
(Patrick)
--> Find common grpund from background and related work, summarize that and descirbe the approach 
--> Layouts (what is a layout?) 
--> "We need a benchmark X" and we inprove the difficulty" 
--> Start Layout, best layout found, end Layout
--> Comparison
--> "Pseudo Code Algorithm" 
--> Optimization Algorithm

(2 pages) 

Experimental Design 
(Justin) 


Results
(Justin + X)
--> Description
--> Metrics
--> Interpretation of the plots and the results 

(all) 
Conclusion
(all + Justin) 

































# Activation Playground

Qt-only desktop playground for small neural networks with editable activation layouts, stepwise training inspection, and simulated annealing over activation distributions.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
mkdir -p .mplconfig
export MPLCONFIGDIR=.mplconfig
python main.py
```

`python main.py` starts the interactive assistant.

Start the GUI directly:

```bash
python main.py gui --mode demo
python main.py gui --mode playground
python main.py gui --mode experiment_builder
```

Useful GUI options:

```bash
python main.py gui --mode playground --detail-level expert --language en
```

## Core CLI Commands

Single training run:

```bash
python main.py run --benchmark wine --hidden-sizes 16 8 --layout "relu|tanh"
```

Export an experiment definition:

```bash
python main.py experiment template --output outputs/examples/example_experiment.json
```

Run a stored experiment headless:

```bash
python main.py experiment run --config outputs/examples/example_experiment.json
```

Analyze stored experiment results:

```bash
python main.py experiment analyze --path outputs/backend_regression/backend_regression_grid
```

Show layout and neighbor syntax help:

```bash
python main.py --show-layout-help
```

## What the Application Covers

- small MLPs with 1 to 4 hidden layers
- activation functions per layer or per neuron
- supported activations: `relu`, `tanh`, `sigmoid`, `leaky_relu`
- benchmarks: `breast_cancer`, `wine`, `digits`, `test_activation`
- train / validation / test evaluation
- sample-level inspection, neuron tracker, activation curve, and stepper
- simulated annealing over activation layouts
- experiment builder with multi-seed runs, grid search, random search, and JSON result storage
- bilingual Qt GUI in German and English

## Main Workspaces

### Demo

Use `Demo` to understand one concrete network:

- inspect one sample
- train step by step
- read the training plot
- click neurons and inspect local computations
- compare current state against a stored baseline

### Playground

Use `Playground` to inspect one simulated annealing run:

- define a start layout
- choose an objective
- select neighborhood operations
- step SA manually or run to completion
- compare start, candidate, current, best, and end states

### Experiment Builder

Use `Experiment Builder` for reproducible experiments:

- define benchmark, hidden sizes, layout, and run mode
- run multiple seeds
- perform grid or random search
- store manifest / summary / per-run JSON files
- reload and analyze experiment results later

## Benchmarks

- `breast_cancer`: 30 numeric features, 2 classes
- `wine`: 13 numeric features, 3 classes
- `digits`: 64 inputs from 8x8 grayscale digits, 10 classes
- `test_activation`: tiny artificial benchmark for raw activation-path analysis

## Project Structure

- `ui_qt/`: Qt shell, workspaces, widgets, and table models
- `services/`: GUI-neutral orchestration and didactic payload builders
- `cli/`: parser, interactive assistant, and command handlers
- `model.py`, `trainer.py`, `activations.py`, `benchmarks.py`: model and training core
- `annealing*.py`: simulated annealing core
- `experiment_*.py`, `results_*.py`, `search_spaces.py`: experiment execution and storage
- `tests/`: CLI, service, Qt widget, and smoke coverage

## Documentation

- [Handbook](docs/HANDBOOK.md): user-facing learning guide, plot interpretation, recipes
- [Architecture](docs/ARCHITECTURE.md): current Qt/service/CLI architecture
