import json
import pandas as pd
import numpy as np
from typing import Any


class BenchmarkEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, pd.DataFrame):
            return {'__dataframe__': True, 'data': obj.to_dict(orient='list'), 'columns': list(obj.columns)}
        if isinstance(obj, pd.Series):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _deserialize_value(obj: Any) -> Any:
    if isinstance(obj, dict):
        if obj.get('__dataframe__'):
            return pd.DataFrame(obj['data'], columns=obj.get('columns'))
        return {k: _deserialize_value(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_deserialize_value(v) for v in obj]
    return obj


def save_results_json(path: str, data: dict) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, cls=BenchmarkEncoder, ensure_ascii=False)


def load_results_json(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return _deserialize_value(data)
