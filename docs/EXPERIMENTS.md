# Terminal Experiments

The active benchmark suite follows the Basics specification:

| Benchmark | Topology | Epochs | Task |
| --- | --- | ---: | --- |
| `two_moons` | `2-8-1` | 100 | binary |
| `concentric_circles` | `2-8-8-1` | 150 | binary |
| `crossing_spirals` | `6-16-16-1` | 250 | binary |

All official benchmarks use binary cross entropy, one sigmoid output neuron, Xavier/Glorot uniform initialization, zero biases, learning rate `0.01`, and batch size `32`.

## Mayer Design

The main Online-Delta-SA experiment is intentionally simple:

- random mixed start layouts
- all six Basics activation functions: `relu`, `gelu`, `sigmoid`, `tanh`, `swish`, `identity`
- one neighborhood function in the main path: `set_neuron`
- one SA step changes exactly one hidden neuron to a different activation function
- many independent runs instead of many loosely motivated experiment variants
- progress curves and activation statistics over the best layouts

`swap_neurons` is available only through `swap-ablation`. `fill_layer` was
removed from the runtime because it is not part of the official Online-Delta
design.

The fallback suite defaults are calibrated for quick experiments on the small Basics nets:

- `start_temperature`: `0.03`
- `cooling_parameter`: `0.95`
- `iterations_per_temperature`: `5`
- `max_steps`: `120`
- `min_temperature`: `0.001`

These values are intentionally much colder than the didactic GUI defaults used earlier. The previous `T=1.5` accepted almost every candidate on `two_moons`, so it behaved too much like a random walk.

For reportable follow-up evaluations, use the benchmark-specific fixed profile
`tuned_20260530` instead of the fallback values. It was selected by the staged
validation-only tuning pipeline and is stored in `configs/evaluation_profiles.json`.

## Suite Commands

```bash
python main.py experiment suite --exp online-delta --benchmark two_moons
python main.py experiment suite --exp random-baseline --benchmark two_moons
python main.py experiment suite --exp all-baseline --benchmark two_moons
python main.py experiment suite --exp swap-ablation --benchmark two_moons
python main.py experiment suite --exp online-delta --benchmark two_moons --learning-rate 2
```

Fixed-profile evaluation:

```bash
python main.py experiment suite --evaluation-profile tuned_20260530 --exp online-delta --benchmark two_moons --runs 30
python main.py experiment suite --evaluation-profile tuned_20260530 --exp random-baseline --benchmark two_moons --runs 30
python main.py experiment suite --evaluation-profile tuned_20260530 --exp all-baseline --benchmark two_moons --runs 30
```

Repeat these commands for `concentric_circles` and `crossing_spirals`.

`--exp` accepts:

- `online-delta`: main Online-Delta-SA run from random mixed start layouts with `set_neuron` only
- `random-baseline`: the same random start-layout mechanism without SA
- `all-baseline`: homogeneous single-activation references (`all_relu`, `all_gelu`, `all_sigmoid`, `all_tanh`, `all_swish`, `all_identity`)
- `swap-ablation`: optional Online-Delta-SA run with `set_neuron` and `swap_neurons`

`--runs` controls the number of independent repetitions. Internally each run stores `layout_seed` and `training_seed`, but the reporting should primarily talk about independent runs.

`--learning-rate` is a preset-group index from `configs/experiment_suites.json`, not a numeric learning rate. Preset `2` expands to multiple concrete learning rates.

`--evaluation-profile tuned_20260530` selects one fixed, benchmark-specific
learning rate and the corresponding training and SA parameters. It also unlocks
final test metrics. Suites without an evaluation profile remain validation-only:
their JSON, CSV, and plots do not contain test metrics. Explicit CLI overrides
such as `--epochs` or `--max-steps` still take precedence and should only be
used for smokes or new calibration work.

For calibration runs, the suite exposes direct annealing overrides:

```bash
python main.py experiment suite --exp online-delta --benchmark two_moons --runs 10 --start-temperature 0.02 --cooling-parameter 0.95
python main.py experiment suite --exp online-delta --benchmark two_moons --runs 10 --start-temperature 0.05 --cooling-parameter 0.95
```

## Online-Delta Reports

After a suite run, aggregate Online-Delta histories with:

```bash
python main.py experiment report-online-delta --path outputs/experiment_suites/<run-dir>
```

The reporter writes:

- `online_delta_steps.csv`
- `online_delta_runs.csv`
- `online_delta_progress_mean.png`
- `online_delta_acceptance_rate.png`
- `online_delta_delta_distribution.png`
- `online_delta_temperature.png`
- `activation_counts_best.csv`
- `activation_counts_best.png`
- `layout_similarity_best.csv`

During normal `online-delta` suite runs without `--no-plots`, every run also
writes compact layout-level visual artifacts:

- `plots_single/<run_id>_layout_summary.png`: start, best, and end layout as color-coded AF heatmaps
- `plots_single/<run_id>_accepted_timeline.png`: accepted AF changes over SA steps

Per-step frame series are intentionally opt-in because they can create thousands
of regenerable PNG files. Enable them only when needed:

```bash
python main.py experiment suite --exp online-delta --benchmark two_moons --export-layout-frames
```

The explicit frame export additionally writes:

- `layout_frames/<run_id>/frame_000_start.png`: random mixed start layout
- `layout_frames/<run_id>/frame_<n>_step_<step>.png`: one frame per accepted layout change
- `layout_frames/<run_id>/accepted_changes.csv`: accepted `set_neuron` changes with delta, temperature, validation loss, and neighbor label

The GUI follows the same main design for Online-Delta-SA: when a new Online-Delta
workflow is opened, the layout editor is already filled with a random mixed
start layout generated from the current hidden sizes and seed. Every click on
`Neues Random-Mixed-Layout` draws a new layout from the reproducible GUI RNG
sequence. Manual baseline training uses the visible layout. Starting Online-Delta
then reuses that same visible random layout, so the baseline plot remains visible
and the GUI comparison has one shared starting point. The `SA-Timeline` tab uses
the same network visualization as the main comparison: the changed neuron of an
accepted candidate is outlined in green and the changed neuron of a rejected
candidate in red. The detail box keeps Online-Delta, temperature, acceptance
probability, validation loss, and the actually retained current layout together.
The `Netzvergleich` tab shows only the start layout before the first SA step and
then adds a selectable best, current, or latest candidate layout on the right.
This makes every in-memory candidate step inspectable without exporting
per-step PNG frames. `python main.py gui --profile tuned_20260530` loads the
promoted benchmark-specific training and SA values.

## Outputs

Suite outputs are written to:

```text
outputs/experiment_suites/<timestamp>_<benchmark>_<exp>_<preset>/
```

Each concrete learning rate gets its own folder:

```text
learning_rate_0.010/
  config.json
  manifest.json
  runs/*.json
  summary.csv
  plots_single/
```

Aggregated suite outputs are written to:

```text
aggregate/
  summary.csv
  mean_curves.png
  val_loss_by_layout.png
  hyperparameter_comparison.png
```

Fixed-profile evaluation additionally writes `test_loss_by_layout.png`. Ranking
is always by mean validation loss. Test loss and test accuracy are reported only
after selection through an explicit evaluation profile.

## Fast Smokes

Use these when checking plumbing, not for scientific claims:

```bash
python main.py experiment suite --exp online-delta --benchmark two_moons --runs 2 --max-steps 5 --epochs 1 --no-plots
python main.py experiment suite --exp random-baseline --benchmark two_moons --runs 2 --epochs 1 --no-plots
python main.py experiment report-online-delta --path outputs/experiment_suites/<smoke-run-dir>
```

Full claims require the default 10 runs or more, depending on runtime budget.

## Fixed Evaluation Profile

The reviewed overnight run is stored at:

```text
outputs/hyperparameter_tuning/20260530_143043_overnight
```

Its promoted profile is:

```text
configs/evaluation_profiles.json -> tuned_20260530
```

The final 30-run results are summarized in `docs/RESULTS_REPORT.md`. The confirmed
finding is deliberately narrow: Online-Delta is slightly positive on `two_moons`,
effectively neutral on `concentric_circles`, and negative on `crossing_spirals`
under the evaluated method.

## Staged Hyperparameter Tuning

The official tuning path separates ordinary SGD calibration from Online-Delta-SA
calibration. Screening and refinement are validation-only: they do not evaluate
or persist test metrics. Test loss and test accuracy are produced only by the
final confirmation phase.

Run the complete overnight profile:

```bash
python main.py experiment tune --profile overnight --benchmark all --phase full --workers 4
```

Run or resume individual phases:

```bash
python main.py experiment tune --benchmark two_moons --phase training-screen --workers 4
python main.py experiment tune --resume outputs/hyperparameter_tuning/<run-dir> --benchmark two_moons --phase training-refine --workers 4
python main.py experiment tune --resume outputs/hyperparameter_tuning/<run-dir> --benchmark two_moons --phase delta-probe --workers 4
python main.py experiment tune --resume outputs/hyperparameter_tuning/<run-dir> --benchmark two_moons --phase sa-screen --workers 4
python main.py experiment tune --resume outputs/hyperparameter_tuning/<run-dir> --benchmark two_moons --phase sa-refine --workers 4
python main.py experiment tune --resume outputs/hyperparameter_tuning/<run-dir> --benchmark two_moons --phase confirm --workers 4
```

The machine-readable search spaces live in `configs/hyperparameter_tuning.json`.
The pipeline writes checkpoints under `outputs/hyperparameter_tuning/`. Completed
trials are skipped during resume; failed trials are retried.

The stages are:

1. `training-screen`: learning rate and epoch search on random mixed layouts.
2. `training-refine`: batch size and Xavier scale search plus homogeneous references.
3. `delta-probe`: measure positive Online-Delta loss changes and derive benchmark-specific temperatures.
4. `sa-screen`: reproducible random search over Online-Delta numeric parameters without retraining every layout.
5. `sa-refine`: retrain matched random starts and SA-found layouts for the best SA candidates.
6. `confirm`: 30-run comparison with test metrics and aggregate reports. Regenerable
   per-step layout frames remain opt-in through `--export-layout-frames`.

For `crossing_spirals`, the fixed-topology SGD calibration must reach mean
validation loss below `0.65` and mean validation accuracy above `0.60`. If that
gate fails, the pipeline stores `diagnostic_only.json`, screens one SA
configuration, and marks the confirmation accordingly. This is a diagnostic
result, not a positive SA claim.

Use the smoke profile only to verify plumbing:

```bash
python main.py experiment tune --benchmark two_moons --phase full --workers 2 --smoke
```
