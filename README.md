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
python main.py gui --mode activation_workflow
python main.py gui --mode experiment_builder
```

Useful GUI options:

```bash
python main.py gui --mode activation_workflow --detail-level expert --language en
```

## Core CLI Commands

Single training run:

```bash
python main.py run --benchmark two_moons --hidden-sizes 8 --layout "relu"
```

Official suite runs:

```bash
python main.py experiment suite --exp online-delta --benchmark two_moons
python main.py experiment suite --exp random-baseline --benchmark two_moons
python main.py experiment suite --exp all-baseline --benchmark two_moons
python main.py experiment suite --exp swap-ablation --benchmark two_moons
python main.py experiment suite --exp online-delta --benchmark two_moons --learning-rate 2
```

Aggregate Online-Delta progress and activation statistics:

```bash
python main.py experiment report-online-delta --path outputs/experiment_suites/<run-dir>
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

Build report-ready tables and plots from existing result artifacts:

```bash
python main.py experiment report --output-dir outputs/report_assets
```

Show layout and neighbor syntax help:

```bash
python main.py --show-layout-help
```

## What the Application Covers

- small MLPs with 1 to 4 hidden layers
- activation functions per layer or per neuron
- supported activations: `relu`, `gelu`, `sigmoid`, `tanh`, `swish`, `identity`
- official binary CSV benchmarks: `two_moons`, `concentric_circles`, `crossing_spirals`
- train / validation / test evaluation
- sample-level inspection, neuron tracker, activation curve, and stepper
- simulated annealing over activation layouts
- online-delta SA evaluation using same weights and same mini-batch
- experiment builder with multi-seed runs, grid search, random search, and JSON result storage
- bilingual Qt GUI in German and English

## Main Workspaces

### Activation Workflow

Use `Activation Workflow` for the full project flow in one window:

- load one official CSV benchmark
- choose or edit an activation layout
- train that layout directly
- optionally run simulated annealing over layouts with online-delta scoring or the legacy short-retrain mode
- finally compare retrained layouts against inherited SA model states under identical evaluation conditions

### Experiment Builder

Use `Experiment Builder` for reproducible experiments:

- define benchmark, hidden sizes, layout, and run mode
- run multiple seeds
- perform grid or random search
- store manifest / summary / per-run JSON files
- reload and analyze experiment results later

## Official Benchmarks

- `two_moons`: 2 numeric inputs, 2 classes, topology `2-8-1`, 100 epochs
- `concentric_circles`: 2 numeric inputs, 2 classes, topology `2-8-8-1`, 150 epochs
- `crossing_spirals`: 6 numeric inputs, 2 classes, topology `6-16-16-1`, 250 epochs

The official CSV files are vendored under `data/benchmarks/basics_group/`; `SOURCE.md` records the upstream repository and commit SHA.

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
- [Experiments](docs/EXPERIMENTS.md): terminal benchmark suites, presets, artifacts, and smoke commands
- [Experiment Workflow Report](docs/EXPERIMENT_WORKFLOW_REPORT.md): metric definitions, fixed workflow, Online-Delta-SA method, and current result interpretation
- [Architecture](docs/ARCHITECTURE.md): current Qt/service/CLI architecture
- [Results Report](docs/RESULTS_REPORT.md): archived legacy report path; regenerate after the new suite baseline
