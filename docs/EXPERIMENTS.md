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

## Baselines and detailed tables

The 19 retained comparators are AGCRN, ASTGCN, ASTGNN, D2STGNN, DCRNN, DSTAGNN,
GMAN, HimNet, MTGNN, PatchSTG, PDFormer, ST-Norm, STAEformer, STEP, STG-Mamba,
STGCN, STGODE, STSGCN and TFT. Tables retain the archived repository labels where
needed to match the recorded results. Per-seed test metrics, seven dataset
tables, full ablation metrics and engineering summaries are included.


## Engineering evaluation

The protocols, experiment scripts, selected warning calibration parameters and
formal results are collected in [`engineering/`](../engineering/README.md).
This includes five-minute numerical accuracy, direct and calibrated threshold
replay, incomplete inputs, processing latency and the local monitoring interface.
