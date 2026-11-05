# Warning protocol and extended results

The manuscript keeps the complete q95 comparison for the validation-F1 and
validation-FA-constrained operating points. This page provides the scoring and
selection details and the other threshold scenarios.

## Fixed forecasts and warning score

Both forecasters use their original frozen checkpoints. For sensor i, the fixed
stress threshold u_i is its training-only 90th, 95th or 99th percentile. The
primary scenario is q95. A warning is eligible only while the current measured
stress is at or below u_i.

For a five-minute forecast, the score is:

$$z_{t,i}=\operatorname{clip}\left(\frac{\max_{h=1,\ldots,5}\widehat y_{t,h,i}-u_i}{\sigma_i},-20,20\right)+b_i.$$

Here sigma_i is the training-node standard deviation, floored at 0.01 MPa. The
bias b_i is the mean normalized residual between observed and predicted
five-minute maxima on the validation-fitting half, shrunk toward the cross-node
mean with 256 pseudo-samples. An eligible issue generates a warning if z exceeds
the selected score threshold tau. This calibration changes the alert rule; the
continuous forecasts, stress thresholds and observed event labels stay fixed.

## Validation fitting and threshold selection

The chronological validation set is divided into halves. Fitting targets must
remain entirely before the midpoint. Fitting histories have stride five and a
maximum of 150,000 sensor-window samples. The second half selects score
thresholds using minute-by-minute event replay.

The common candidates are margin-only, bias-plus-margin, logistic regression
and histogram gradient boosting. A single family is selected by mean q95
validation event F1 across both forecasters, all seven datasets and three seeds.
The selected family is bias-plus-margin. Exact classifier settings and the
threshold candidate grid are in
[`configs/warning_calibration_protocol.json`](configs/warning_calibration_protocol.json).
The four family-selection scores are in
[`results/validation_family_summary.csv`](results/validation_family_summary.csv).

The F1 operating point maximizes validation event F1. Constrained operating
points maximize recall subject to validation FA budgets of 0.01, 0.02 or 0.05
per sensor-hour. If the selection half has no observed onsets, the primary F1
point keeps direct thresholding for both forecasters. Budget points retain the
validation constraint. The constraint is applied on validation and does not
impose identical test false-alarm rates.

All parameters are fixed before calibrated reassessment of the previously used
chronological test segment. This is an additional study on that segment, not a
new external blind test. The 126 original direct-threshold case records were
reproduced by the event evaluator.

## Matching and aggregation

Consecutive eligible alerts at a sensor form a segment. Its first issue is
greedily matched to the earliest unmatched observed onset in the next five
minutes. Each segment and onset can be matched at most once. Only onsets with
five complete preceding forecast opportunities are evaluated.

Precision is matched segments divided by all alert segments; recall is matched
onsets divided by all eligible onsets. F1 is their harmonic mean. FA is unmatched
segments divided by all evaluated sensor-hours. Lead is observed onset minus
matched segment issue time, calculated only for detected events. When no alert
or event exists, the corresponding precision/recall/F1 is zero; lead is undefined.
Metrics average the three seeds within each dataset before the seven datasets
receive equal weight. Undefined lead values are omitted. Macro F1 is averaged
from case-level F1, rather than recomputed from macro precision and recall.

## Other threshold scenarios

The following table uses the validation-F1 operating point.

| Training percentile | Model | P (%) | R (%) | F1 (%) | FA / sensor-hour | Lead (min) |
|---|---|---:|---:|---:|---:|---:|
|90%|SHARP|49.14|70.30|57.76|0.0391|2.07|
|90%|PatchSTG|49.14|65.66|56.19|0.0363|2.02|
|95%|SHARP|53.81|72.72|60.44|0.0340|2.21|
|95%|PatchSTG|52.62|65.05|57.18|0.0300|1.87|
|99%|SHARP|53.39|71.34|59.97|0.0311|2.22|
|99%|PatchSTG|54.74|63.72|57.09|0.0273|2.17|

For the q95 direct-threshold reference, F1 is 26.44% for SHARP and 35.24% for
PatchSTG. With calibration, both improve. At the q95 F1 operating point, the
paired mean F1 difference favors SHARP by 3.27 percentage points, with a 95%
percentile bootstrap interval of [1.67, 5.84] points. Resampling uses seven
dataset differences after seed averaging, 20,000 draws and RNG seed 20261104.
All seven dataset mean differences are positive. At that operating point SHARP
also has a higher unmatched-alert frequency (0.0340 vs 0.0300 per sensor-hour).

Full case metrics, all budget operating points and confidence intervals are in
[`results/test_case_metrics.csv`](results/test_case_metrics.csv),
[`results/test_macro_summary.csv`](results/test_macro_summary.csv) and
[`results/paired_effect_intervals.csv`](results/paired_effect_intervals.csv).
Selected numeric parameters are in
[`configs/warning_operating_points.json`](configs/warning_operating_points.json).
Reproduction commands are in [the engineering README](README.md#reproduction).
No intermediate prediction caches are needed in the public repository.
