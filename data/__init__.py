import os
import numpy as np
from utils import parse

__all__ = ["in_signal", "train_symbols", "symbol_pattern"]


cur_dir = os.path.dirname(__file__)


def quant(x: np.ndarray, quantiple: float = 1) -> np.ndarray:
    return (np.round(x / quantiple) * quantiple).astype(float)

def get_in_signal():
    in_signal = parse(os.path.join(cur_dir, "RX_X.csv"), delimiter='\n')
    in_signal =in_signal / np.mean(np.abs(in_signal))
    in_signal = quant(in_signal, np.max(in_signal) / (2 ** 6))
    in_signal = in_signal / np.mean(np.abs(in_signal)) * 3 / 2
    in_signal = 2 * in_signal - 3
    return in_signal

in_signal = get_in_signal()

def get_train_symbols():
    train_symbols = parse(os.path.join(cur_dir, "train_symbol.csv"), astype=int)
    train_symbols = 2 * train_symbols - 3
    return train_symbols

train_symbols = get_train_symbols()

def get_symbol_pattern():
    symbol_pattern = parse(os.path.join(cur_dir, "symbol_pattern.csv"), astype=int)
    symbol_pattern = np.concatenate([symbol_pattern, symbol_pattern], 0)
    return symbol_pattern

symbol_pattern = get_symbol_pattern()
