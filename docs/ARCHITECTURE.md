# Architecture Overview

## Product Boundary

The repository exposes two intentional entry paths:

```text
CLI = reproducible scientific experiments
GUI = focused Online-Delta demonstration
```

## Public Commands

```bash
python main.py run ...
python main.py gui ...
python main.py experiment suite ...
python main.py experiment tune ...
python main.py experiment report-online-delta ...
```

The official suite types are:

```text
online-delta
random-baseline
all-baseline
```

`swap-ablation` is an explicitly optional comparison. It is not part of the main
claim.

Normal suites are validation-only. Passing an explicit versioned
`--evaluation-profile` marks a final reporting run and unlocks held-out test
metrics.

## Active Core

| Area | Files |
| --- | --- |
| Benchmark registry and loading | `benchmark_registry.py`, `benchmarks.py` |
| Activation layouts and sampling | `activations.py` |
| MLP and SGD training | `model.py`, `trainer.py` |
| Online-Delta session | `services/online_annealing_training_service.py` |
| Official experiment suites | `services/experiment_suite_service.py` |
| Staged tuning | `services/hyperparameter_tuning_service.py` |
| Fixed evaluation profiles | `services/evaluation_profile_service.py`, `configs/evaluation_profiles.json` |
| Online-Delta reports | `services/online_delta_reporting_service.py` |
| Layout visualization | `services/layout_visualization_service.py` |

## Online-Delta Data Flow

1. Load one official binary benchmark.
2. Use its fixed Basics topology.
3. Generate a random mixed hidden-neuron activation layout.
4. Initialize Xavier weights and zero biases.
5. Select one hidden neuron uniformly.
6. Replace its activation with a different Basics activation.
7. Evaluate candidate and current state with the same weights and mini-batch.
8. Accept improvements and apply temperature-based SA acceptance to worse moves.
9. Continue mini-batch training after accepted candidate steps.
10. Retrain start, best, and end layouts from scratch for final comparison.

## GUI

The public GUI contains one `activation_workflow` workspace. It is intended for
interactive explanation of Online-Delta behavior. Multi-run evidence belongs to
the CLI suites. The selected benchmark determines the fixed Basics topology;
the GUI no longer exposes workspace routing or topology overrides.

The workspace exposes `demo` and `tuned_20260530` parameter profiles and an
in-memory SA timeline. `fill_layer`, short-retrain controls, and Builder routing
are not part of the visible GUI.

Qt startup:

- `ui_qt/app.py`
- `ui_qt/shell/main_window.py`
- `ui_qt/workspaces/activation_workflow_workspace.py`


## Generated Artifacts

Normal Online-Delta suites retain compact histories and plots. Per-step PNG frame
series are opt-in:

```bash
python main.py experiment suite --exp online-delta --benchmark two_moons --export-layout-frames
```


## Verification

```bash
python -m pytest tests -q
git diff --check
```
