"""Apply published five-minute warning settings to a forecast (NumPy only)."""
from pathlib import Path
import json
import numpy as np

def calibrated_warning(current, forecast, dataset, seed=2026, model='SHARP',
                       quantile=.95, operating_point='f1', config=None):
    """Return node alarms for forecast shape [..., horizon>=5, nodes].

    current has shape [..., nodes]. Alarms are issued only below or at the
    fixed stress threshold. None is the stored representation of +infinity.
    """
    if config is None:
        config=Path(__file__).resolve().parents[1]/'configs/warning_operating_points.json'
    settings=json.loads(Path(config).read_text(encoding='utf-8'))
    data=settings['data'][dataset];entry=settings['models'][f'{dataset}_{seed}_{model}']
    point=entry['operating_points'][str(quantile)][operating_point]
    stress_threshold=np.asarray(data['stress_thresholds'][str(quantile)])
    scale=np.asarray(data['node_std'])
    # Match the study's float32 score followed by float64 bias addition.
    score=((np.max(forecast[..., :5, :],axis=-2)-stress_threshold)/scale).astype(np.float32)
    np.clip(score,-20,20,out=score)
    if not point.get('override'):score=score+np.asarray(entry['bias'])
    tau=np.inf if point['threshold'] is None else point['threshold']
    return (np.asarray(current)<=stress_threshold)&(score>tau)
