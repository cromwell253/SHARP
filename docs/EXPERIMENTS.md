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
validation curves and stage seconds. Shared baseline settings and recorded
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
are in `configs/evaluation_environment.json`; per-window measurements are in
`results/engineering/latency_samples/`.

## Baselines and detailed tables

The 19 retained comparators are AGCRN, ASTGCN, ASTGNN, D2STGNN, DCRNN, DSTAGNN,
GMAN, HimNet, MTGNN, PatchSTG, PDFormer, ST-Norm, STAEformer, STEP, STG-Mamba,
STGCN, STGODE, STSGCN and TFT. Tables retain the archived repository labels where
needed to match the recorded results. Per-seed test metrics, seven dataset
tables, full ablation metrics and engineering summaries are included.
