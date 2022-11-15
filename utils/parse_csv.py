import numpy as np

__all__ = ["parse"]

def parse(path, delimiter=',', astype=float) -> np.ndarray:
    """解析CSV文件为Array."""
    data = []
    with open(path, 'r') as f:
        data = f.read()
        data = data.split(delimiter)

        data = data[:-1] if data[-1].strip() == '' else data

        for idx, data_ in enumerate(data):
            try:
                data[idx] = float(data_)
            except Exception as e:
                print(f"`{e}` from {path}: {idx}/{len(data)} - {data_}")
    return np.array(data).astype(float)
