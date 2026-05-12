# Architecture Overview

This repository is now a Qt-only desktop application with a thin CLI layer and a shared service boundary.

## 1. High-Level Shape

The codebase is split into four practical layers:

- `cli/`
  - command parsing
  - interactive assistant
  - headless command handlers
- `ui_qt/`
  - Qt application shell
  - workspaces
  - reusable widgets
  - Qt-specific background task handling
- `services/`
  - GUI-neutral orchestration
  - didactic payload builders
  - session abstractions for training and simulated annealing
- core modules
  - benchmarks
  - activation layouts
  - MLP model
  - trainer
  - simulated annealing
  - experiment execution and storage

## 2. Current Entry Paths

### Main Entry

- `main.py`
  - builds the parser
  - resolves a command
  - dispatches to CLI or Qt GUI

### GUI Entry

- `python main.py gui --mode activation_workflow`
- `python main.py gui --mode experiment_builder`

Qt startup lives in:

- `ui_qt/app.py`
- `ui_qt/shell/main_window.py`

### CLI Entry

- `python main.py run ...`
- `python main.py experiment run ...`
- `python main.py experiment analyze ...`
- `python main.py experiment template ...`

## 3. Core Model and Training

- `benchmarks.py`
  - dataset loading
  - scaling
  - train / validation / test split
  - target names

- `activations.py`
  - activation functions and derivatives
  - layout parsing
  - layout editing helpers
  - neighborhood operations

- `model.py`
  - `ModularMLP`
  - forward pass
  - loss and gradients
  - tracing and neuron inspection
  - model state serialization

- `trainer.py`
  - mini-batch training loop
  - history collection
  - epoch streaming hooks used by the Qt training session path

## 4. Simulated Annealing

- `annealing.py`
  - SA dataclasses and acceptance logic

- `annealing_schedules.py`
  - cooling schedules

- `annealing_objectives.py`
  - legacy candidate evaluation after short training

- `annealing_runner.py`
  - legacy short-retrain SA runs

- `services/annealing_service.py`
  - legacy short-retrain annealing execution

- `services/annealing_session_service.py`
  - legacy short-retrain stepwise SA session snapshots

- `services/online_annealing_training_service.py`
  - default online-delta SA mode
  - measures candidate AF changes on the same weights and same mini-batch
  - trains only after the SA decision

## 5. Experiment System

- `experiment_builder.py`
  - experiment and run definitions

- `search_spaces.py`
  - fixed / list / range search spaces
  - grid and random search expansion

- `experiment_runner.py`
  - experiment execution over configurations and seeds

- `results_store.py`
  - manifest / summary / per-run JSON storage

- `results_analysis.py`
  - aggregation and ranking

- `services/experiment_service.py`
  - GUI- and CLI-friendly experiment load/save/run helpers

- `services/preview_service.py`
  - summary formatting helpers

## 6. Qt UI Layer

### Shell

- `ui_qt/shell/main_window.py`
  - top-level workspace router
  - language / detail controls
  - global handbook window

### Workspaces

- `ui_qt/workspaces/activation_workflow_workspace.py`
  - primary GUI workflow
  - benchmark and layout editing
  - sample inspection, network view, neuron tracker, and stepper
  - manual training
  - online-delta SA search and legacy short-retrain comparison mode
  - final layout comparison

- `ui_qt/workspaces/experiment_builder_workspace.py`
  - experiment definition editing
  - results loading
  - run table
  - per-run preview

### Shared Widgets

- `ui_qt/widgets/network_view.py`
- `ui_qt/widgets/sample_panel.py`
- `ui_qt/widgets/neuron_detail_panel.py`
- `ui_qt/widgets/activation_curve_widget.py`
- `ui_qt/widgets/stepper_panel.py`
- `ui_qt/widgets/annealing_*_panel.py`
- `ui_qt/widgets/help_dialog.py`
- `ui_qt/widgets/info_button.py`

### Qt Tasking

- `ui_qt/tasking.py`
  - Qt thread-pool based background execution for UI actions

## 7. Service Layer Responsibilities

The service layer is the application boundary. Qt widgets should render service output, not reimplement domain logic.

Important services:

- `services/analysis_sample_service.py`
  - benchmark-specific sample payloads

- `services/network_projection_service.py`
  - reduced network projection for readable visualization

- `services/neuron_analysis_service.py`
  - neuron tracker payloads

- `services/stepper_service.py`
  - forward/backward teaching entries

- `services/help_service.py`
  - handbook, workspace help, and field-level didactic content

- `services/training_service.py`
  - single-run training from CLI or services

## 8. Data Flow

### Activation Workflow

1. load benchmark
2. build activation layout
3. build preview model
4. inspect one sample through service payloads
5. train in chunks
6. optionally run online-delta SA over AF layouts
7. compare start, best, end, random, and homogeneous baselines under identical final training conditions

### Experiment Builder

1. build experiment definition
2. run experiment headless or load stored results
3. aggregate summary
4. render run table, detail view, and network preview

## 9. Test Strategy

The current test suite is intentionally conservative:

- parser and CLI smoke tests
- service tests
- Qt widget tests for key didactic panels
- Qt smoke tests for active workspaces

The repo should be kept green with:

- `python -m pytest tests -q`
- offscreen Qt launch smokes for both active workspaces
