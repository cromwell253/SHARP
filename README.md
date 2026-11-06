# SHARP

**History-Selective Attention and Prediction-Guided Spatial Incremental Correction for Multi-Node, Multi-Step Mining-Induced Stress Forecasting**

SHARP is a spatiotemporal forecasting model for mining-induced stress prediction. It integrates history-selective attention and prediction-guided spatial incremental correction to capture temporal dependencies and spatial interactions among multiple monitoring points, enabling multi-node, multi-step stress forecasting.

## 1. Datasets

The project includes seven real-world mining-induced stress datasets:

- JAPD
- WF1
- WF13112
- WF15212
- WF1905
- WF2301
- WF2903

The datasets are stored in the `data/` directory. Each dataset contains the following files:

- `series.npz`: Stress measurements stored as a numerical array with shape `[time, nodes]`, using `float32` precision and MPa as the measurement unit.
- `coordinates.csv`: Local spatial coordinates of the monitoring points.

All datasets have a sampling interval of one minute. Spatial coordinates are defined relative to the first monitoring point, preserving pairwise distances between monitoring locations.

Calendar dates and original sensor identifiers are omitted, while the original temporal and spatial ordering is preserved.

### Dataset coverage

| Dataset | Sensors | Time steps | Start / end timestamps | Days |
|---|---:|---:|---|---:|
| JAPD | 12 | 21363 | 2025-07-01 12:01 to  2025-07-16 08:03 | 14 |
| WF1 | 36 | 30240 | 2025-07-01 12:01 to  2025-07-22 12:00 | 21 |
| WF13112 | 36 | 131040 | 2026-03-02 14:30 to  2026-06-01 14:29 | 91 |
| WF15212 | 26 | 175680 | 2026-04-15 09:12 to  2026-08-15 09:11 | 122 |
| WF1905 | 30 | 161280 | 2025-08-11 10:42 to  2025-12-01 10:41 | 112 |
| WF2301 | 43 | 172800 | 2025-08-01 11:02 to  2025-11-29 11:01 | 120 |
| WF2903 | 16 | 175680 | 2026-03-09 18:33 to  2026-07-09 18:32 | 122 |

The table reports the first and last timestamps retained in each experimental input. Calendar timestamps are omitted from the released numerical arrays.

## 2. Experimental Configuration

### 2.1 Data Splitting and Preprocessing

All datasets are chronologically divided into three subsets:

| Dataset Split | Ratio |
|---|---:|
| Training | 60% |
| Validation | 20% |
| Testing | 20% |

The model takes the most recent 12 minutes of stress measurements as input and predicts stress variations at all monitoring points over the subsequent 48 minutes.

Training and evaluation windows are generated with a five-minute stride, while event replay uses a one-minute stride.

Data normalization is performed using the mean and standard deviation calculated from the training set. Validation and test inputs may include preceding historical observations, while all prediction targets remain within their respective data splits.

### 2.2 Model Comparison

- **Input length:** 12 minutes
- **Forecast horizon:** 48 minutes
- **Sampling interval:** 1 minute
- **Random seeds:** 2026, 2027, 2028
- **Evaluation metrics:** MAE and RMSE
- **Forecasting leads:** 12, 24, and 48 minutes

The main comparison evaluates SHARP against 19 independent baseline models across all seven datasets. Results are averaged over three random seeds for each dataset.

### 2.3 Ablation and Engineering Evaluations

Five model configurations are included in the ablation study to investigate the contributions of individual model components.

Engineering evaluations cover the following aspects:

1. **Stress Threshold Event Replay:** Evaluates stress threshold crossing detection and warning lead times using thresholds derived from the 90th, 95th, and 99th percentiles of the training data, with both direct thresholding and shared validation-only calibration.
2. **Missing Data Robustness:** Evaluates forecasting performance under randomly masked historical observations.
3. **Input Staleness:** Examines model performance when the latest two minutes of monitoring data are delayed.
4. **Inference Latency:** Measures model execution efficiency on CPU and GPU, including preprocessing, inference, and result serialization.

Engineering protocols, experiment scripts, selected warning parameters, and detailed results are collected in [`engineering/`](engineering/README.md).

## 3. Experimental Results

Experimental results are organized as follows:

| File or Directory | Description |
|---|---|
| `results/comparison.xlsx` | Summary of model comparison results across seven datasets |
| `results/ablation.xlsx` | Summary of ablation study results |
| `results/comparison/` | Per-dataset, per-seed, and aggregated comparison results |
| `engineering/results/` | Engineering evaluation results, including five-minute forecasting accuracy, threshold event replay, input perturbation tests, and inference latency |

The experimental results cover:

- Multi-step stress forecasting performance across seven datasets.
- Ablation analysis of individual SHARP components.
- Forecasting errors at different prediction horizons.
- Stress threshold event detection under different threshold settings.
- Forecasting performance under missing and delayed input data.
- CPU and GPU inference efficiency.

## 4. Model Implementation

The project is organized as follows:

```text
SHARP/
├── data/                       # Seven mining-induced stress datasets
├── sharp/
│   ├── models/                 # SHARP architecture and model components
│   ├── training.py              # Training implementations for the correction stages
│   └── inference.py             # Model loading and inference interface
├── checkpoints/                # Model checkpoints
├── configs/                     # Data splits and experimental configurations
├── results/
│   ├── comparison/              # Model comparison results
│   ├── comparison.xlsx          # Comparison summary
│   └── ablation.xlsx             # Ablation summary
├── engineering/                # Engineering protocols, experiments, and results
├── evaluate.py                  # Forecasting performance evaluation
└── replay.py                    # Stress threshold event replay
```

The checkpoints cover all seven datasets, three random seeds, and five ablation configurations.

`evaluate.py` loads the specified dataset, model variant, and checkpoint to perform forecasting evaluation.

`replay.py` performs minute-by-minute historical stress event replay using predefined thresholds.

The inference interface accepts historical stress measurements with shape `[batch, 12, nodes]` and produces forecasts with shape `[batch, 48, nodes]`, where stress values are expressed in MPa.
