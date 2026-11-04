# Experimental details

## Architecture

The common historical predictor uses a learned convex 12-position kernel and an
own-node MLP with two 32-dimensional GELU hidden layers and a zero-initialized
48-step head. The spatial backbone uses three attention layers and a
48-dimensional representation (input/node/time-of-day/weekday dimensions
16/16/8/8). History attention uses 8-dimensional node embeddings and keys/queries.
Spatial correction uses four neighbors, observed history shifts 0,1,2,3, and
32-dimensional hidden states. Read the model source for tensor-level operations.

The graph response subtracts a zero-input reference and a training-input linear
projection onto own relative history. It enters with scale 0.25. The neighbor
correction enters with scale 0.01. Static partitions are stored explicitly so
coordinate anonymization does not rebuild or alter the published model.

## Training and ablation

Training is staged. The supplied frozen base contains the trained historical
predictor, spatial backbone and history projection. History attention is trained
on this base, then spatial correction is trained with earlier stages frozen.
The stage kernel is `sharp/training.py`; its inputs are normalized histories,
training/validation targets, cached base predictions, feature normalization,
and the node-wise forward callback. `train_stage` expects `RUN` and `CONFIG_HASH`
to be set by its caller. It returns the selected model state and its history.

Adam uses learning rate 0.001, weight decay 0.0001 and gradient clipping 5.
History attention allows 60 epochs, minimum 20, patience 12 and node batch 4096.
Spatial correction allows 40 epochs, patience 8 and node batch 1024, using at most
8192 evenly spaced training windows. Epoch zero is eligible for validation-MAE
selection. The history residual training loss is scaled by 1/0.01. Node-stage
selection uses float64 error arithmetic; the main result tables use the recorded
float32 inverse-transform arithmetic. These two metric conventions are separate.

`configs/staged_training_records.json` contains actual completed/selected epochs,
and stage seconds. Shared baseline settings and recorded
epochs are in `configs/protocol.json` and `configs/comparison_training_records.json`.
These configurations specify a common-budget adapted comparison, not a rerun of
every baseline author's original paper configuration. Total upstream training
and model-development budgets are not claimed to be equal across methods.

| Variant | Common historical predictor | History attention | Spatial correction |
|---|---|---|---|
| C0 | yes | no | no |
| H | yes | yes | no |
| S | yes | no | yes |
| SHARP | yes | yes | yes |
| PatchSTG | independent matched comparator | its own native attention | its own native architecture |

H and S are parallel factor configurations, not consecutive additions to the
independent PatchSTG comparator. Retained correction stages in the ablation use
the corresponding full-model completed epoch budgets; omitted heads are zero.
The matched PatchSTG uses one layer and 24 dimensions (8/8/4/4), as recorded in
the original matched experiment. Component gains should be evaluated relative
to C0 and to the other full/partial configurations.

## Threshold replay

The predicted interval is the next five minutes. A node-window is positive if
any of these observations exceeds its training-only node threshold; it triggers
a predicted alarm if any of its forecasts exceeds that same threshold.
Current-above-threshold windows are excluded from pre-onset precision.

An event starts when a node changes from at-or-below threshold to above threshold.
Contiguous above-threshold readings belong to one event. An onset is evaluated
only if all five preceding issue times have complete replay predictions. It is
detected when at least one eligible pre-onset issue predicts crossing within
five minutes. Lead time is onset minus the first successful issue time, evaluated
only among detected events. The archived protocol does not define one-to-one
predicted-alarm episodes, so its window precision must not be labelled event
precision. Likewise, a new event F1 requires an explicitly defined matching rule.

## Input perturbations

Random missingness uses RNG seed 20261101, a 10% mask over history readings, and
keeps the first history position. Each masked position is filled from the prior
available value. Staleness replaces the last two history values with the value
at the preceding minute. The same perturbation is supplied to each model.
Archived results cover seven datasets, two neural models and two conditions.
They describe a single missingness realization and seed2026 models.

## Latency

Each group uses batch one, 50 warmups and 200 measured windows, with GPU
synchronization. Local processing includes normalization, device transfer,
48-step inference, inverse transformation and five-step result serialization.
CPU uses two PyTorch threads. Slow samples are retained. P99 is the empirical
99th percentile, not a worst-case deadline guarantee. Hardware and backend flags
are in `configs/evaluation_environment.json`; published latency summaries are in
`results/engineering/`; detailed timing records are retained locally.

## Baselines and detailed tables

The 19 retained comparators are AGCRN, ASTGCN, ASTGNN, D2STGNN, DCRNN, DSTAGNN,
GMAN, HimNet, MTGNN, PatchSTG, PDFormer, ST-Norm, STAEformer, STEP, STG-Mamba,
STGCN, STGODE, STSGCN and TFT. Tables retain the archived repository labels where
needed to match the recorded results. Per-seed test metrics, seven dataset
tables, full ablation metrics and engineering summaries are included.

## Extended evaluation

The formal engineering tables in `results/engineering/event_metrics.csv` and
`robustness_metrics.csv` cover three seeds and three missingness masks. The
episode matching rule and mask seeds are in `configs/engineering.json`.
`continuous_pipeline_summary.json` reports the 1,000-update accelerated WF1
software run per model. The generator is `experiments/engineering_extension.py`.
Intermediate caches, candidate-case lists, debug audits and duplicate figure
formats are not part of the public result directory.


## Monitoring interface validation

The manuscript expands SHARP as Selective Historical Attention and Residual
Prediction. The local monitoring prototype processes each numerical input update
through a frozen checkpoint, threshold evaluation, JSON output, and a browser
interface showing a selected-node forecast and an all-sensor heatmap. Browser
timing starts before input serialization and ends after canvas drawing and two
requestAnimationFrame callbacks. Loopback HTTP is included.

WF1 seed2026 uses 12 input steps, 48 output steps, and a displayed five-step
forecast. Each model has 50 inference warmups, 10 complete interface warmups,
and 1,000 measured sequential updates. Input indices advance by one minute;
the replay runs in accelerated wall time. The result is a local interface
prototype evaluation. Summary, hardware and browser version are in
`results/engineering/visual_pipeline_summary.json`.

| Model | Interface P50 (ms) | P95 (ms) | P99 (ms) | Maximum (ms) | Updates exceeding60s |
|---|---:|---:|---:|---:|---:|
| SHARP |33.500|66.900|67.201|100.100|0|
| PatchSTG |33.300|33.600|33.800|66.700|0|

Reproduce from the repository root. Start the backend in one terminal:

```bash
python experiments/ui_monitor_backend.py --repo . --out runs/ui_validation
```

In a second terminal, use Node.js with Playwright installed:

```bash
node experiments/ui_monitor_client.cjs runs/ui_validation
```

Windows uses installed Microsoft Edge; other systems use Playwright Chromium.
The client saves detailed timing samples and interface screenshots to the run
directory. Stop the backend with Ctrl+C after the client finishes.

## Descriptive spatial correlations

`results/engineering/spatial_correlation_diagnostic.csv` describes the final20%
of each dataset. Geometric four-neighbor pairs are deduplicated as unordered
pairs, and compared with all remaining sensor pairs. Pearson correlations are
averaged separately for observed stress levels and one-minute increments.
Undefined correlations involving zero-variance sequences are excluded, with
valid pair counts retained. These descriptive test-segment statistics explain
data characteristics; they do not estimate causal stress transmission.

```bash
python experiments/spatial_diagnostic.py --repo . --output runs/spatial_correlation_diagnostic.csv
```
