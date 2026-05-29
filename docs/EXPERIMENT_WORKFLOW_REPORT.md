# Experiment Workflow Report

This report documents the current official experiment workflow of the Activation Playground. It is intended as a GitHub-facing technical reference: what the codebase does, which experiments are part of the main design, which metrics are produced, and how results should be interpreted.

The current main experiment design follows a deliberately simple Online-Delta Simulated Annealing setup:

- official binary Basics benchmarks only
- random mixed activation layouts as SA start points
- one neighborhood operation in the main path: `set_neuron`
- all six Basics activation functions
- matched random baselines
- homogeneous single-AF baselines
- validation loss as the primary ranking metric
- test metrics only for final reporting

`swap-ablation` exists in the codebase as an optional extension, but it is not part of the main experiment package described here.

## 1. Project Goal

The project investigates whether small neural networks can benefit from mixed activation-function layouts.

A normal MLP usually fixes one activation function, for example ReLU, before training. Here, each hidden neuron can independently use one activation function from:

```text
relu, gelu, sigmoid, tanh, swish, identity
```

The main research question is:

> Can Online-Delta Simulated Annealing find mixed activation layouts that, after retraining, outperform random mixed layouts and homogeneous single-activation baselines?

The project does not primarily try to produce the best possible classifier for each benchmark. The central goal is to compare layout-search strategies under controlled conditions.

## 2. Official Benchmarks

The official benchmark set is fixed to:

| Benchmark | Topology | Epochs | Task |
| --- | --- | ---: | --- |
| `two_moons` | `2-8-1` | 100 | binary classification |
| `concentric_circles` | `2-8-8-1` | 150 | binary classification |
| `crossing_spirals` | `6-16-16-1` | 250 | binary classification |

All official benchmarks use:

- one sigmoid output neuron
- binary cross entropy
- learning rate `0.01`
- batch size `32`
- Xavier/Glorot uniform initialization
- zero bias initialization
- train / validation / test split

`test_activation` remains an internal didactic/debug benchmark, not an official result benchmark.

## 3. Chronological Experiment Workflow

The full experiment workflow is:

1. Load an official benchmark CSV.
2. Split the data into train, validation, and test subsets.
3. Build an MLP with the benchmark topology.
4. Choose or generate an activation layout for the hidden neurons.
5. Train/evaluate that layout or run Online-Delta-SA to search for improved layouts.
6. Retrain selected layouts from scratch under identical conditions.
7. Aggregate results across 10 independent runs.
8. Rank by mean validation loss.
9. Report test loss and test accuracy only after selection.
10. Generate plots and CSV artifacts for progress, acceptance behavior, and activation statistics.

This order matters. Validation metrics guide selection. Test metrics are not used for search or tuning decisions.

## 4. Model And Training

Each run creates a modular MLP:

```text
input -> hidden layer(s) with configurable activation layout -> sigmoid output
```

The output layer is always sigmoid for the official binary benchmarks. Simulated Annealing only changes hidden-layer activation functions.

Training uses mini-batch gradient descent. For each epoch, the model sees batches from the training split and updates weights using binary cross entropy.

Important training hyperparameters:

| Hyperparameter | Current value | Meaning |
| --- | ---: | --- |
| `learning_rate` | `0.01` | Step size for weight updates |
| `batch_size` | `32` | Number of samples per gradient update |
| `epochs` | benchmark-specific | Number of full passes over training data |
| `weight_scale` | `1.0` | Scale used for initialization API; current model uses Xavier/Glorot |
| `shuffle` | `true` | Shuffle train batches between epochs |

Good training behavior means:

- train loss decreases over epochs
- validation loss also decreases or remains stable
- train and validation accuracy improve together
- train loss much better than validation loss can indicate overfitting
- both train and validation stuck near random performance means the benchmark is not learned under the current setup

For binary cross entropy, a loss around `0.69` often indicates near-random predictions on balanced data. That is why `crossing_spirals` currently needs careful interpretation: many results stay close to `0.69`.

## 5. Metrics

### `train_loss`

`train_loss` is the loss on the training split.

It answers:

> How well does the model fit the data it is allowed to learn from?

Lower is better, but train loss alone is not enough. A model can memorize or overfit training data.

### `val_loss`

`val_loss` is the loss on the validation split.

It answers:

> How well does the model generalize to data not used for gradient updates?

This is the primary metric for ranking layouts and experiment configurations. In the current suite, lower mean validation loss is the main success criterion.

### `test_loss`

`test_loss` is the loss on the held-out test split.

It answers:

> How well does the selected model perform on data that was not used for training or model selection?

Test loss is only reported after selection. It should not be used to tune hyperparameters or choose layouts.

### `train_accuracy`

`train_accuracy` is classification accuracy on the training split.

It is easy to understand but less sensitive than loss. Accuracy can stay unchanged even when the model becomes more confident or better calibrated.

### `val_accuracy`

`val_accuracy` is accuracy on the validation split.

It is useful as a sanity check, but the suite ranks primarily by validation loss. In some cases two layouts can have similar accuracy but different validation loss.

### `test_accuracy`

`test_accuracy` is final accuracy on the test split.

It is useful for reporting and presentation, but not for search. For binary balanced tasks, `0.5` is roughly random performance.

### `ranking_score`

`ranking_score` is the value used to sort result rows. In the current official suite it is mean validation loss.

Lower is better.

### `acceptance_rate`

`acceptance_rate` belongs to Online-Delta-SA.

It is the fraction of SA candidate layouts accepted during the run.

Interpretation:

- too close to `1.0`: search accepts almost everything and behaves too much like a random walk
- too close to `0.0`: search is frozen and barely moves
- middle range: search rejects some bad moves but still explores

The current calibrated defaults produced reasonable acceptance on `two_moons` and `concentric_circles`. `crossing_spirals` is still high, but the larger issue there is weak benchmark learning.

### `online_validation_loss_delta`

This is:

```text
final online validation loss - step 0 online validation loss
```

Negative values mean the continuously trained Online-Delta process improved over time.

This is not the same as final retrained layout quality. It describes the inherited online model during the SA process.

## 6. Experiment Types

The official main suite uses three experiment types.

### 6.1 `all-baseline`

Command:

```bash
python main.py experiment suite --exp all-baseline --benchmark two_moons
```

This trains six homogeneous layouts:

```text
all_relu
all_gelu
all_sigmoid
all_tanh
all_swish
all_identity
```

Purpose:

> Establish simple, interpretable reference baselines.

If Online-Delta cannot beat the best homogeneous baseline, the mixed-layout search is less convincing for that benchmark.

### 6.2 `random-baseline`

Command:

```bash
python main.py experiment suite --exp random-baseline --benchmark two_moons
```

This creates random mixed activation layouts and trains them normally, without SA.

Purpose:

> Check whether mixed layouts are already good just because they are random.

This is the most important fairness baseline for Online-Delta-SA, because Online-Delta also starts from random mixed layouts.

For every run index, Online-Delta and Random-Baseline use the same random start layout. That makes paired comparison possible.

### 6.3 `online-delta`

Command:

```bash
python main.py experiment suite --exp online-delta --benchmark two_moons
```

This is the main SA experiment.

Each run:

1. samples a random mixed start layout
2. initializes model weights
3. evaluates the start state as Step 0
4. repeatedly proposes one activation change
5. accepts or rejects the candidate using SA
6. trains after accepted candidates
7. stores the best and final layouts
8. retrains selected layouts from scratch for final comparison

The final comparison includes:

| Label | Meaning |
| --- | --- |
| `same_random_start_retrained` | Original random start layout retrained from scratch |
| `best_layout_from_sa_retrained` | Best layout found during SA, retrained from scratch |
| `end_layout_from_sa_retrained` | Final layout after SA, retrained from scratch |
| `best_inherited_model_from_sa` | Best model state inherited directly from online SA |
| `end_inherited_model_from_sa` | Final model state inherited directly from online SA |

The retrained rows are the main scientific comparison. The inherited rows are diagnostic.

## 7. Online-Delta Simulated Annealing

Online-Delta-SA evaluates a candidate layout change without retraining the candidate first.

At each SA step:

1. Take the current model weights.
2. Take the current mini-batch.
3. Evaluate the current layout on that mini-batch.
4. Propose exactly one hidden-neuron activation change.
5. Apply the candidate layout using the same weights.
6. Evaluate the candidate on the same mini-batch.
7. Compute the loss delta:

```text
delta = candidate_batch_loss - current_batch_loss
```

8. Accept the candidate if it improves the loss.
9. If it is worse, accept with probability:

```text
P(accept) = exp(-delta / temperature)
```

10. If accepted, keep the candidate layout and train the model on that mini-batch.
11. If rejected, restore the previous layout.
12. Continue with the next step and updated temperature.

This is called Online-Delta because the acceptance decision uses the immediate loss delta under the same mini-batch and same weights.

The candidate is not retrained before the acceptance decision. This avoids mixing the layout effect with extra training noise.

## 8. Neighborhood Function

The official main path uses only:

```text
set_neuron
```

Mechanism:

1. Choose one hidden neuron uniformly.
2. Choose one different activation function uniformly from all six Basics AFs.
3. Replace only that neuron activation.

Input neurons are never changed. The output sigmoid is never changed.

This follows the simplified design: one local AF change per step. Larger moves such as changing a whole layer are intentionally excluded from the main experiment.

## 9. Simulated Annealing Hyperparameters

Current suite defaults:

| Hyperparameter | Value | Meaning |
| --- | ---: | --- |
| `start_temperature` | `0.03` | Initial willingness to accept worse candidates |
| `cooling_parameter` | `0.95` | Geometric cooling multiplier |
| `iterations_per_temperature` | `5` | Number of steps before cooling |
| `max_steps` | `120` | Maximum SA steps per run |
| `min_temperature` | `0.001` | Stop threshold for temperature |
| `neighborhood_operations` | `set_neuron` | Allowed layout moves |

Interpretation:

- high temperature increases exploration
- low temperature increases exploitation
- too high acceptance rate means the search is too permissive
- too low acceptance rate means the search is too rigid

The current defaults were calibrated after observing that `start_temperature=1.5` accepted almost all candidates on `two_moons`.

## 10. Output Artifacts

Suite output directory:

```text
outputs/experiment_suites/<timestamp>_<benchmark>_<exp>_lr_preset_<n>/
```

Each learning-rate folder contains:

```text
learning_rate_0.010/
  config.json
  manifest.json
  runs/*.json
  summary.csv
  plots_single/
```

The `aggregate/` folder contains:

```text
summary.csv
mean_curves.png
val_loss_by_layout.png
test_loss_by_layout.png
hyperparameter_comparison.png
```

For Online-Delta runs it also contains:

```text
online_delta_steps.csv
online_delta_runs.csv
online_delta_progress_mean.png
online_delta_acceptance_rate.png
online_delta_delta_distribution.png
online_delta_temperature.png
activation_counts_best.csv
activation_counts_best.png
layout_similarity_best.csv
```

## 11. Current Result Snapshot

The following snapshot summarizes the current three-benchmark run set:

| Benchmark | Online best val loss | Random val loss | Best homogeneous baseline | Online vs Random | Online vs Best-All | Wins vs Random | Acceptance |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| `two_moons` | `0.3047` | `0.3138` | `all_identity` (`0.3105`) | `-0.0091` | `-0.0058` | `9/10` | `0.639` |
| `concentric_circles` | `0.0787` | `0.1454` | `all_gelu` (`0.0887`) | `-0.0667` | `-0.0100` | `7/10` | `0.777` |
| `crossing_spirals` | `0.6927` | `0.6941` | `all_swish` (`0.6897`) | `-0.0014` | `+0.0030` | `6/10` | `0.888` |

Interpretation:

- `two_moons`: positive but small Online-Delta signal
- `concentric_circles`: strongest Online-Delta result
- `crossing_spirals`: inconclusive; all methods remain close to random-level loss

The current result story is therefore not "Online-Delta wins everywhere". The accurate statement is:

> Online-Delta-SA improves retrained layouts on `two_moons` and `concentric_circles`. On `crossing_spirals`, the current training setup does not produce a meaningful separation between methods.

## 12. How To Interpret Good Results

A good Online-Delta result should satisfy most of these:

1. `best_layout_from_sa_retrained` has lower mean validation loss than `same_random_start_retrained`.
2. The paired win count against the same random start layout is clearly above chance.
3. Online-Delta beats or matches the best homogeneous baseline.
4. Test accuracy does not contradict the validation ranking.
5. Acceptance rate is neither almost `0` nor almost `1`.
6. Online progress curves improve over steps.
7. Best layouts show interpretable AF statistics.

The most important comparison is paired:

```text
same random start layout without SA
vs.
best layout found from that start by Online-Delta-SA
```

This avoids unfairly comparing unrelated random layouts.

## 13. CLI Commands For The Main Experiment Package

Run all three main experiments for one benchmark:

```bash
python main.py experiment suite --exp online-delta --benchmark two_moons
python main.py experiment suite --exp random-baseline --benchmark two_moons
python main.py experiment suite --exp all-baseline --benchmark two_moons
```

Repeat for:

```text
two_moons
concentric_circles
crossing_spirals
```

Generate Online-Delta aggregate reports:

```bash
python main.py experiment report-online-delta --path outputs/experiment_suites/<online-delta-run-dir>
```

Learning-rate preset sweeps:

```bash
python main.py experiment suite --exp all-baseline --benchmark crossing_spirals --learning-rate 2
python main.py experiment suite --exp online-delta --benchmark crossing_spirals --learning-rate 2
python main.py experiment suite --exp random-baseline --benchmark crossing_spirals --learning-rate 2
```

Use hyperparameter sweeps carefully. If tuning is performed on one benchmark, the corresponding baselines must be rerun with the same training hyperparameters.

## 14. Codebase Map

Important files:

| Path | Role |
| --- | --- |
| `configs.py` | Global defaults and supported names |
| `configs/experiment_suites.json` | Machine-readable suite setup |
| `benchmark_registry.py` | Official benchmark topology/epoch registry |
| `benchmarks.py` | CSV loading and dataset splitting |
| `model.py` | Modular MLP and activation layout support |
| `trainer.py` | Training loop and evaluation metrics |
| `activations.py` | Layout parsing, activation functions, neighbor operations |
| `annealing.py` | SA config and acceptance probability |
| `services/online_annealing_training_service.py` | Online-Delta-SA core loop |
| `services/experiment_suite_service.py` | Official terminal suite runner |
| `services/online_delta_reporting_service.py` | Online-Delta aggregate plots and CSVs |
| `cli/parser.py` | CLI subcommands and arguments |
| `docs/EXPERIMENTS.md` | Short operational experiment guide |

## 15. Reporting Discipline

When writing results:

- report mean and standard deviation over runs
- state the number of runs
- state the benchmark topology and epochs
- state learning rate, batch size, and SA parameters
- rank by validation loss
- mention test metrics only after selection
- distinguish retrained layouts from inherited online models
- do not overclaim `crossing_spirals` under current results

Recommended result wording:

> The Online-Delta-SA suite evaluates whether local activation-function changes can find mixed layouts that improve over matched random starts. On `two_moons` and `concentric_circles`, the best SA-found layouts improve mean validation loss after retraining. On `crossing_spirals`, all compared methods remain close to random-level loss, so this benchmark is currently inconclusive under the fixed training setup.

