# Architecture Overview

This document explains how the repository is structured and how data moves through it.

## 1. High-Level Structure

The repository separates:

- benchmark loading
- activation layout logic
- model mathematics
- training loop
- plotting
- GUI orchestration
- simulated annealing
- experiment execution and result storage

## 2. Main Files

### Core Model and Training

- `activations.py`
  - activation functions
  - activation derivatives
  - layout parsing
  - layout modification logic
  - neighborhood generation

- `model.py`
  - `ModularMLP`
  - forward pass
  - loss and gradients
  - parameter updates
  - neuron inspection
  - model state serialization

- `trainer.py`
  - training loop
  - mini-batch iteration
  - history collection

- `benchmarks.py`
  - benchmark loading
  - scaling
  - train / validation / test split
  - target names

### Visualization

- `plotting.py`
  - standard plots for training histories

- `terminal_viz.py`
  - terminal ASCII views

- `gui.py`
  - interactive GUI
  - all three work modes
  - network visualization
  - learning help
  - builder analysis views

### Simulated Annealing

- `annealing.py`
  - annealing dataclasses
  - acceptance logic

- `annealing_schedules.py`
  - temperature schedules

- `annealing_objectives.py`
  - candidate evaluation by short training

- `annealing_runner.py`
  - one complete SA run
  - step history
  - current / candidate / best state handling

### Experiment Builder

- `experiment_builder.py`
  - builder dataclasses
  - experiment and run definitions

- `search_spaces.py`
  - discrete search spaces
  - fixed / list / range
  - grid search expansion
  - random search sampling

- `experiment_runner.py`
  - executes builder experiments
  - manual training runs
  - SA runs
  - multi-seed orchestration

- `results_analysis.py`
  - aggregation across runs and seeds
  - ranking
  - summary generation

- `results_store.py`
  - JSON storage
  - manifest / summary / per-run loading

- `experiment_plots.py`
  - plots reconstructed from builder results

### Program Entry

- `main.py`
  - CLI
  - terminal assistant
  - GUI start logic

- `configs.py`
  - defaults
  - shared configuration dataclasses

## 3. Data Flow

The standard data flow is:

1. `main.py` chooses CLI or GUI
2. a benchmark is loaded through `benchmarks.py`
3. a layout is parsed through `activations.py`
4. the model is constructed in `model.py`
5. training is executed through `trainer.py`
6. plots and visualizations are created

In GUI mode, `gui.py` orchestrates these steps live.

## 4. Training Flow

Normal training works as follows:

1. benchmark is loaded and split
2. a `ModularMLP` is initialized
3. mini-batches are created
4. forward pass is run
5. loss and gradients are computed
6. gradients are applied
7. train and validation metrics are stored each epoch
8. final test metrics are measured

The important conceptual split is:

- `model.py` knows the math
- `trainer.py` knows the training loop

## 5. Layout Flow

An activation layout is a structured object that describes all hidden activations.

Typical flow:

1. user edits a layout string or GUI controls
2. `parse_layout_spec(...)` creates an `ActivationLayout`
3. the model uses that layout
4. the GUI colors neurons by activation type
5. SA treats that layout as a search state

## 6. Simulated Annealing Flow

In this project, simulated annealing searches over activation layouts.

Flow:

1. choose a start layout
2. build an evaluator for candidate layouts
3. generate neighbors from the current layout
4. train each candidate briefly from scratch
5. compute validation-based objective
6. accept or reject candidate depending on score and temperature
7. update current, best, and history

Important detail:

SA does not directly optimize weights.

It compares layouts by training each layout for a short budget and then evaluating it.

## 7. Experiment Builder Flow

The Experiment Builder adds a second layer of orchestration.

Flow:

1. define one experiment
2. expand search space into configurations
3. combine each configuration with all selected seeds
4. execute one run per configuration-seed pair
5. aggregate metrics
6. store manifest, summary, and runs
7. reload later for analysis

## 8. JSON Result Structure

Stored experiments typically contain:

- `manifest.json`
- `summary.json`
- `runs/<run_id>.json`

### Manifest

Contains:

- experiment identity
- benchmark
- run mode
- search type
- summary path
- run file list

### Summary

Contains:

- number of runs
- number of seeds
- ranking
- aggregated metrics per configuration

### Per-Run JSON

Contains:

- run definition
- metrics
- history
- layout
- extra metadata

For SA runs, extra metadata includes:

- objective name
- start layout
- best layout
- end layout
- annealing config
- stop reasons
- step history

## 9. GUI Architecture Notes

`gui.py` is the largest file because it coordinates:

- controls
- state variables
- plots
- tabs
- network drawing
- live inspection
- builder analysis

The most important design rule is:

The GUI should orchestrate and explain, while the actual training, layout logic, and SA backend stay in separate modules.

## 10. Where To Read First

For someone new to the codebase, a good reading order is:

1. `main.py`
2. `benchmarks.py`
3. `activations.py`
4. `model.py`
5. `trainer.py`
6. `annealing.py`
7. `annealing_runner.py`
8. `experiment_runner.py`
9. `gui.py`

This order helps because it follows the pipeline before the large orchestration layer.
