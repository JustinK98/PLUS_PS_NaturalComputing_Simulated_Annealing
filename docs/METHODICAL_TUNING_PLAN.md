# Methodical Online-Delta-SA Revalidation Plan

## Goal

Determine whether Online-Delta-SA improves a matched random activation layout
after training, across all official benchmarks, without selecting on the held-out
test split.

The reportable comparison is:

```text
same random start retrained vs final SA layout retrained
```

The diagnostic best layout observed during SA is not the official final result.

## Experimental Design

- Benchmarks: `two_moons`, `concentric_circles`, `crossing_spirals`
- Tuning profile: `methodical`
- Selection metric: validation loss
- Final metrics: paired validation-loss improvement, paired win rate, test loss,
  test accuracy, acceptance rate, and effective online epochs
- Confirmation design: 10 independent layout blocks x 3 replicates
- Inference unit: mean result per layout block, not each nested replicate
- Test split: locked until the fixed-parameter confirmation phase

Each layout block has its own deterministic data split and random start layout.
The three replicates inside a block share that split and layout, but use
independent online weights, mini-batches, SA proposals, SA acceptance draws, and
retraining streams.

## Factors

### Training

- Learning rate
- Epoch budget
- Batch size
- Weight scale

### Online-Delta-SA

- Start temperature derived from observed positive loss deltas
- Online learning-rate factor
- Online batch size
- Iterations per temperature
- Online-training budget
- Cooling schedule
- Target end-temperature ratio

Cooling schedules are compared using matched target end-temperature ratios.
This avoids comparing a geometric factor directly with an unrelated linear
decrement or logarithmic scale.

Included strategies:

- geometric
- linear
- logarithmic

Target end-temperature ratios:

- `0.01`
- `0.05`
- `0.20`

## Selection and Validation

1. Screen and refine SGD settings using validation metrics only.
2. Probe positive Online-Delta loss changes and derive start temperatures.
3. Select a sufficient online-training budget.
4. Screen SA candidates with stratified cooling-strategy coverage.
5. Refine the best healthy candidates using paired retraining.
6. Freeze one configuration per benchmark.
7. Run 10 layout blocks x 3 independent replicates.
8. Evaluate the held-out test split only during final confirmation.
9. Aggregate nested replicates to layout-block means before inference.

## Stop and Health Rules

- Reject non-finite configurations.
- Prefer acceptance rates between `0.15` and `0.80`.
- Require positive mean online progress during SA screening.
- Require the selected online-training budget to be reached.
- Keep `crossing_spirals` diagnostic-only if its training stop rule fails.
- Do not promote partial results or diagnostic best-observed layouts.

## Execution

End-to-end smoke:

```bash
MPLCONFIGDIR=.mplconfig ./.venv/bin/python main.py experiment tune \
  --profile methodical \
  --benchmark all \
  --phase full \
  --workers 4 \
  --smoke
```

Full resumable run:

```bash
MPLCONFIGDIR=.mplconfig ./.venv/bin/python main.py experiment tune \
  --profile methodical \
  --benchmark all \
  --phase full \
  --workers 4
```

Resume:

```bash
MPLCONFIGDIR=.mplconfig ./.venv/bin/python main.py experiment tune \
  --profile methodical \
  --benchmark all \
  --phase full \
  --workers 4 \
  --resume outputs/hyperparameter_tuning/<run-dir>
```

Refresh the progress and analysis report:

```bash
MPLCONFIGDIR=.mplconfig ./.venv/bin/python scripts/build_tuning_report.py \
  outputs/hyperparameter_tuning/<run-dir>
```

Expanded locked confirmation after the selected profiles are frozen:

```bash
MPLCONFIGDIR=.mplconfig ./.venv/bin/python main.py experiment tune \
  --profile methodical-confirmation-30 \
  --benchmark all \
  --phase confirm \
  --workers 4 \
  --resume outputs/hyperparameter_tuning/<run-dir>
```

This reuses completed confirmation blocks and expands the design to 30
independent layout/split blocks x 3 nested replicates.

## Artifacts

The tuning run writes checkpointed trial CSV files, rankings, selected
configurations, confirmation JSON, aggregate Online-Delta plots, and a final
summary. The report builder additionally writes:

- `analysis/METHODICAL_TUNING_REPORT.md`
- `analysis/phase_status.csv`
- `analysis/cooling_schedule_status.csv`
- `analysis/final_confirmation_summary.csv`
- `analysis/seed_audit.csv`
- `analysis/activation_effect_associations.csv`
- `analysis/trial_status_by_phase.png`
- `analysis/cooling_schedule_progress.png`
- `analysis/final_confirmation_effects.png`
- `analysis/activation_effect_associations.png`

Each benchmark aggregate also writes layout-level activation-share features and
exploratory associations between activation-share changes and paired
improvement. These associations guide controlled follow-up ablations; they are
not causal activation-function effects.
