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

`swap_neurons` is available only through `swap-ablation`. `fill_layer` is not part of the official Online-Delta main experiment.

The suite defaults are calibrated for the observed Online-Delta loss scale on the small Basics nets:

- `start_temperature`: `0.03`
- `cooling_parameter`: `0.95`
- `iterations_per_temperature`: `5`
- `max_steps`: `120`
- `min_temperature`: `0.001`

These values are intentionally much colder than the didactic GUI defaults used earlier. The previous `T=1.5` accepted almost every candidate on `two_moons`, so it behaved too much like a random walk.

## Suite Commands

```bash
python main.py experiment suite --exp online-delta --benchmark two_moons
python main.py experiment suite --exp random-baseline --benchmark two_moons
python main.py experiment suite --exp all-baseline --benchmark two_moons
python main.py experiment suite --exp swap-ablation --benchmark two_moons
python main.py experiment suite --exp online-delta --benchmark two_moons --learning-rate 2
```

`--exp` accepts:

- `online-delta`: main Online-Delta-SA run from random mixed start layouts with `set_neuron` only
- `random-baseline`: the same random start-layout mechanism without SA
- `all-baseline`: homogeneous single-activation references (`all_relu`, `all_gelu`, `all_sigmoid`, `all_tanh`, `all_swish`, `all_identity`)
- `swap-ablation`: optional Online-Delta-SA run with `set_neuron` and `swap_neurons`

`--runs` controls the number of independent repetitions. Internally each run stores `layout_seed` and `training_seed`, but the reporting should primarily talk about independent runs.

`--learning-rate` is a preset-group index from `configs/experiment_suites.json`, not a numeric learning rate. Preset `2` expands to multiple concrete learning rates.

For calibration runs, the suite exposes direct annealing overrides:

```bash
python main.py experiment suite --exp online-delta --benchmark two_moons --runs 10 --start-temperature 0.02 --cooling-parameter 0.95
python main.py experiment suite --exp online-delta --benchmark two_moons --runs 10 --start-temperature 0.05 --cooling-parameter 0.95
```

## Online-Delta Reports

After a suite or Experiment Builder run, aggregate Online-Delta histories with:

```bash
python main.py experiment report-online-delta --path outputs/experiment_suites/<run-dir>
python main.py experiment report-online-delta --path outputs/experiments/<experiment-id>
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
  test_loss_by_layout.png
  hyperparameter_comparison.png
```

Ranking is by mean validation loss. Test loss and test accuracy are reported only after selection.

## Fast Smokes

Use these when checking plumbing, not for scientific claims:

```bash
python main.py experiment suite --exp online-delta --benchmark two_moons --runs 2 --max-steps 5 --epochs 1 --no-plots
python main.py experiment suite --exp random-baseline --benchmark two_moons --runs 2 --epochs 1 --no-plots
python main.py experiment report-online-delta --path outputs/experiment_suites/<smoke-run-dir>
```

Full claims require the default 10 runs or more, depending on runtime budget.
