# Activation Playground Handbook

## Purpose

Activation Playground demonstrates one method clearly:

```text
random mixed hidden activations
  -> Online-Delta Simulated Annealing
  -> retrain selected layouts
  -> compare against matched random and homogeneous baselines
```

The GUI is a didactic view of one run. Reportable multi-run experiments belong
to the CLI.

## Official Benchmarks

| Benchmark | Inputs | Topology | Basics epochs |
| --- | ---: | --- | ---: |
| `two_moons` | 2 | `2-8-1` | 100 |
| `concentric_circles` | 2 | `2-8-8-1` | 150 |
| `crossing_spirals` | 6 | `6-16-16-1` | 250 |

All three tasks are binary classifications with one sigmoid output neuron.
Hidden neurons use `relu`, `gelu`, `sigmoid`, `tanh`, `swish`, or `identity`.

## GUI Demo

Start the focused workspace:

```bash
python main.py gui
```

Use the validation-selected evaluation parameters when you want to inspect the
fixed final setup:

```bash
python main.py gui --profile tuned_20260530
```

The GUI offers two parameter profiles:

| Profile | Purpose |
| --- | --- |
| `demo` | Fast interactive explanation with compact Basics defaults |
| `tuned_20260530` | Fixed benchmark-specific parameters from the reviewed overnight tuning artifact |

The left side follows the intended demonstration order:

1. Select benchmark and profile.
2. Generate a random mixed layout.
3. Optionally train the current layout as a baseline.
4. Run Online-Delta one step, ten steps, or to completion.
5. Inspect the SA Timeline.
6. Retrain and compare selected layouts.

The selected benchmark fixes the Basics topology. The GUI intentionally does
not expose hidden-size or workspace overrides.

## Online-Delta

Every Online-Delta step is deliberately simple:

1. Select one hidden neuron uniformly.
2. Replace its activation with a different Basics activation.
3. Evaluate old and candidate layout using the same weights and mini-batch.
4. Compute `delta = candidate_loss - current_loss`.
5. Accept improvements directly.
6. Accept some worse candidates according to the current SA temperature.
7. Train the accepted state further with one mini-batch update.

Input neurons never change. The sigmoid output neuron never changes. The
official main path uses only `set_neuron`.

The Timeline tab shows the start layout and every later decision. A green border
marks an accepted changed neuron; a red border marks the proposed change of a
rejected neighbor. The detail box shows delta, temperature, acceptance
probability, validation loss, and current layout.

## Metrics

| Metric | Meaning | Desired direction |
| --- | --- | --- |
| `train_loss` | Binary cross entropy on training data | lower |
| `val_loss` | Binary cross entropy on held-out validation data | lower |
| `train_accuracy` | Correct predictions on training data | higher |
| `val_accuracy` | Correct predictions on validation data | higher |
| `test_loss` | Final loss on unseen test data | lower |
| `test_accuracy` | Final accuracy on unseen test data | higher |

Use validation metrics for selection. Use test metrics only after the
configuration has been fixed.

## CLI Evidence

Run the three official comparisons:

```bash
python main.py experiment suite --evaluation-profile tuned_20260530 --exp online-delta --benchmark two_moons --runs 30
python main.py experiment suite --evaluation-profile tuned_20260530 --exp random-baseline --benchmark two_moons --runs 30
python main.py experiment suite --evaluation-profile tuned_20260530 --exp all-baseline --benchmark two_moons --runs 30
```

Repeat these commands for the other benchmarks. Use
`experiment report-online-delta` to aggregate histories and activation
statistics.

## Scope Boundary

`swap-ablation` remains an optional CLI comparison. It is not part of the main
claim. Historical design notes are archived under `docs/archive/`.
