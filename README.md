# Activation Playground

Activation Playground evaluates activation-function layouts in small neural
networks. The official experiment path follows a simple Online-Delta Simulated Annealing design:

- random mixed start layouts
- all six Basics activations: `relu`, `gelu`, `sigmoid`, `tanh`, `swish`, `identity`
- one uniformly selected hidden neuron changes per candidate step
- `set_neuron` as the only main-path neighborhood function
- matched random-layout and homogeneous baselines
- validation-loss ranking across independent runs

The repository has two public entry paths:

```text
CLI = reproducible scientific experiments
GUI = focused Online-Delta demonstration
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
mkdir -p .mplconfig
export MPLCONFIGDIR=.mplconfig
```

Show the available commands:

```bash
python main.py
```

## GUI Demo

```bash
python main.py gui
python main.py gui --benchmark concentric_circles
python main.py gui --benchmark concentric_circles --profile demo
```

The GUI visualizes one Online-Delta workflow with an in-memory decision
timeline. A normal GUI start loads `methodical_selected_20260613`, the
benchmark-specific configuration selected by the completed methodical tuning
run. Use `demo` for a faster explanation. The selected profile is useful for
reproducing the tuned setup, but its locked confirmation did not establish a
robust general Online-Delta-SA advantage. Reproducible multi-run evaluation
belongs to the CLI. The GUI always uses the fixed Basics topology of the
selected benchmark; free topology overrides remain available only in the manual
`run` command.

## CLI

Run one manually selected layout:

```bash
python main.py run --benchmark two_moons --hidden-sizes 8 --layout "relu"
```

Run the three official experiment types:

```bash
python main.py experiment suite --exp online-delta --benchmark two_moons
python main.py experiment suite --exp random-baseline --benchmark two_moons
python main.py experiment suite --exp all-baseline --benchmark two_moons
```

`swap-ablation` remains available as an explicitly optional comparison:

```bash
python main.py experiment suite --exp swap-ablation --benchmark two_moons
```

Use the fixed validation-selected profile for reportable follow-up evaluation:

```bash
python main.py experiment suite \
  --evaluation-profile tuned_20260530 \
  --exp online-delta \
  --benchmark two_moons \
  --runs 30
```

Suites without `--evaluation-profile` are validation-only calibration runs. The
held-out test split is evaluated only for explicit fixed-profile reporting.

Aggregate Online-Delta progress and activation statistics:

```bash
python main.py experiment report-online-delta \
  --path outputs/experiment_suites/<run-dir>
```

Normal runs write compact JSON, CSV, aggregate plots, and per-run layout
summaries. Export regenerable per-step layout frames only when needed:

```bash
python main.py experiment suite \
  --exp online-delta \
  --benchmark two_moons \
  --export-layout-frames
```

Run or resume staged validation-first hyperparameter tuning:

```bash
python main.py experiment tune --profile overnight --benchmark all --phase full --workers 4
python main.py experiment tune --resume outputs/hyperparameter_tuning/<run-dir> --benchmark all --phase full --workers 4
```

## Official Benchmarks

| Benchmark | Topology | Default epochs | Task |
| --- | --- | ---: | --- |
| `two_moons` | `2-8-1` | 100 | binary |
| `concentric_circles` | `2-8-8-1` | 150 | binary |
| `crossing_spirals` | `6-16-16-1` | 250 | binary |

The vendored CSV files live under `data/benchmarks/basics_group/`.

## Documentation

- [Terminal Experiments](docs/EXPERIMENTS.md)
- [Fixed Results](docs/RESULTS_REPORT.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Documentation Archive](docs/archive/README.md)
