import numpy as np

__all__ = ["parse"]

def parse(path, delimiter=',', astype=float) -> np.ndarray:
    """解析CSV文件为Array."""
    data = []
    with open(path, 'r') as f:
        data = f.read()
        data = data.split(delimiter)
        for idx, data_ in enumerate(data):
            try:
                data[idx] = float(data_)
            except Exception as e:
                print(f"`{e}` from {path}: {idx}/{len(data)} - {data_}")
                data[idx] = 0
    return np.array(data).astype(float)
