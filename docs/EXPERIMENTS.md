# Terminal Experiments

## Official Scope

The active experiment path follows the Basics specification and Professor
Mayer's simplified Online-Delta-SA design.

| Benchmark | Topology | Default epochs | Task |
| --- | --- | ---: | --- |
| `two_moons` | `2-8-1` | 100 | binary |
| `concentric_circles` | `2-8-8-1` | 150 | binary |
| `crossing_spirals` | `6-16-16-1` | 250 | binary |

All official benchmarks use binary cross entropy, one sigmoid output neuron,
Xavier/Glorot uniform initialization, zero biases, and hidden activations from:

```text
relu, gelu, sigmoid, tanh, swish, identity
```

## Online-Delta-SA

Each run starts with random Xavier weights and a random mixed activation layout.
Only `set_neuron` is part of the main path:

1. Select one hidden neuron uniformly. Input neurons are excluded.
2. Draw one different activation function uniformly.
3. Evaluate current layout and candidate layout with the same weights and the
   same current training mini-batch.
4. Compute:

   ```text
   delta = candidate_batch_loss - current_batch_loss
   ```

5. Accept every `delta <= 0`.
6. For `delta > 0`, accept with:

   ```text
   exp(-delta / temperature)
   ```

7. Only after acceptance, train the retained candidate on that mini-batch and
   advance the mini-batch cursor.
8. After rejection, restore the old layout. Weights and mini-batch cursor stay
   unchanged.
9. Lower the temperature according to the geometric schedule after every
   proposal, accepted or rejected.

This is continuous online training during search. Weights are not frozen for
the full run. They are frozen only while comparing current layout and candidate
inside one proposal.

## Result Semantics

The primary SA result is the final retained layout when the search freezes or
hits its safety limit.

The pipeline also stores a diagnostic Validation-best intermediate layout. It
is useful for understanding the trajectory, but it is not the official SA
output because selecting it adaptively by Validation-Loss would change the
method.

The final comparison retrains layouts from scratch with fresh weights:

- `same_random_start_retrained`
- `end_layout_from_sa_retrained`
- `best_online_delta_value`: inherited diagnostic value, not a retraining curve

The first two use the same data split, initial weights, and mini-batch order.
Their difference therefore isolates the effect of the final SA layout as far as
possible. `best_online_delta_value` answers whether continuous online training
itself produced a useful inherited model.

## Run and Seed Design

An official suite uses:

```text
10 independent layout/split blocks x 3 independent replicates = 30 paired runs
```

For each run, the pipeline stores separate deterministic seeds for layout
generation, the benchmark split, Online-Delta weights and batches, SA proposals
and acceptance, and retraining weights and batches. Each block has its own
layout and benchmark split. The three replicates of one block share both, but
all stochastic training and SA streams differ. Random-start and final-layout
retraining deliberately share retraining seeds for a fair paired comparison.
The benchmark-level 95% confidence interval is calculated over the ten
block-level mean improvements, not naively over all thirty nested runs.

## GUI Comparison Designs

The GUI exposes the methodological design before benchmark selection:

- `Blocked Layout Comparison`: random start layouts differ, while split,
  Online-Delta weights and batches, SA proposal and acceptance streams, and
  retraining streams are shared across layouts. This isolates layout effects
  through common random numbers.
- `Robustheitsanalyse`: all random streams differ across runs. This measures
  robustness of the complete method rather than an isolated layout effect.

Within every run, random-start and final-SA-layout retraining remains paired.
For scientific inference from blocked comparisons, repeat complete blocks with
multiple master seeds and treat the block as the independent unit.

The complete mathematical specification is documented in
`docs/ONLINE_DELTA_SA_A_TO_Z.md`.

## Metrics

| Metric | Meaning | Interpretation |
| --- | --- | --- |
| `batch_loss_before` | BCE of retained model on current mini-batch | local proposal baseline |
| `candidate_loss_after` | BCE of candidate on the same mini-batch and same weights | local candidate quality |
| `delta` | candidate loss minus retained loss | negative is a local improvement |
| `post_training_batch_loss` | BCE after an accepted candidate was trained once | online fitness progress |
| `trained_batch_updates` | accepted proposals that caused one SGD update | actual online training amount |
| `effective_online_epochs` | consumed training examples divided by training-set size | comparable online training budget |
| `val_loss` | BCE on Validation split | tuning and trajectory diagnosis only |
| `val_accuracy` | classification accuracy on Validation split | secondary diagnosis |
| `test_loss` | BCE on untouched Test split | final reporting only |
| `test_accuracy` | classification accuracy on Test split | final reporting only |
| `acceptance_rate` | accepted proposals divided by all proposals | SA health indicator |

The SA decision uses `delta`, never Accuracy and never Test metrics.

## Public Suites

Use the promoted fixed profile for final reporting runs:

```bash
python main.py experiment suite \
  --evaluation-profile mayer_corrected_20260604 \
  --exp online-delta \
  --benchmark two_moons
```

The alias `--evaluation-profile default` resolves to the current promoted
profile from `configs/evaluation_profiles.json`.

```bash
python main.py experiment suite --exp online-delta --benchmark two_moons
python main.py experiment suite --exp random-baseline --benchmark two_moons
```

`swap-ablation` remains optional:

```bash
python main.py experiment suite --exp swap-ablation --benchmark two_moons
```

Useful calibration overrides:

```bash
python main.py experiment suite \
  --exp online-delta \
  --benchmark two_moons \
  --layouts 10 \
  --replicates 3 \
  --start-temperature 0.02 \
  --cooling-parameter 0.995 \
  --iterations-per-temperature 10 \
  --target-online-epochs 25
```

`--target-online-epochs` documents the calibrated minimum online-training
budget. A normal suite does not stop just because this value was reached.
Search stops through SA temperature or `--max-steps`.

## Reporting

```bash
python main.py experiment report-online-delta \
  --path outputs/experiment_suites/<run-dir>
```

The reporter writes:

- `online_delta_steps.csv`
- `online_delta_runs.csv`
- `online_delta_fitness_mean.png`
- `online_delta_validation_progress_mean.png`
- `online_delta_acceptance_rate.png`
- `online_delta_delta_distribution.png`
- `online_delta_temperature.png`
- `activation_counts_end.csv`
- `activation_counts_end.png`
- `layout_similarity_end.csv`
- `seed_manifest.csv`
- `paired_runs.csv`
- `layout_level_summary.csv`
- `benchmark_summary.csv`
- `paired_random_vs_sa_end.png`
- `paired_improvement_by_layout.png`
- `paired_improvement_distribution.png`

The fitness graph shows post-training Batch-Loss over actual SGD updates. The
Validation graph shows epoch-level diagnostics. Activation statistics and
layout similarity use final SA end layouts.

Per-run layout summaries are generated unless `--no-plots` is used. Large frame
series remain opt-in:

```bash
python main.py experiment suite \
  --exp online-delta \
  --benchmark two_moons \
  --export-layout-frames
```

## Mayer-Corrected Tuning

The earlier profile `tuned_20260530` is retained only as
`legacy_methodology_v1`. Do not use it for new claims.

The reviewed downstream run has been promoted as:

```text
mayer_corrected_20260604
```

It fixes benchmark-specific training and Online-Delta parameters for final
evaluation. The detailed interpretation lives in `docs/RESULTS_REPORT.md`.

Ordinary SGD training selections remain reusable because they do not depend on
the corrected Online-Delta cursor semantics. All SA-dependent stages must be
recomputed:

1. `delta-probe`
2. `online-budget-probe`
3. `sa-screen`
4. `sa-refine`
5. `confirm`

Run the targeted downstream pipeline:

```bash
python main.py experiment tune \
  --profile mayer-corrected \
  --benchmark all \
  --phase online-full \
  --import-training-from outputs/hyperparameter_tuning/20260530_143043_overnight \
  --workers 4
```

The compact budget probe checks `5, 10, 25, 50, 100` effective online epochs.
If no plateau is visible, it extends to `150, 250`. The subsequent SA screen
uses the selected budget as a health gate while the actual search still runs
until frozen or capped.

Use the smoke path only for plumbing checks:

```bash
python main.py experiment tune \
  --profile mayer-corrected \
  --benchmark two_moons \
  --phase online-full \
  --import-training-from outputs/hyperparameter_tuning/20260530_143043_overnight \
  --workers 2 \
  --smoke
```

## Legacy Results

The pre-correction report remains available for provenance:

```text
docs/archive/RESULTS_REPORT_tuned_20260530_legacy.md
```

The active report at `docs/RESULTS_REPORT.md` contains the reviewed
Mayer-corrected Confirmation. Its historical metrics predate the separated
seed schedule and remain reference values until a new fixed-profile
confirmation is completed.
