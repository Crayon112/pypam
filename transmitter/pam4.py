# !usr/bin/env python
# -*- encoding: utf-8 -*-
"""PAM4 信号发生器.

@File: pam4.py
@Time: 2022/11/13 14:42:19
@Author: Crayon112
@SoftWare: VSCode
@Description: PAM4 信号发生器.

"""


import numpy as np


class  PAM4Transmitter(object):

    def __init__(self, symbols: np.ndarray, n_samples: int) -> None:
        self._symbols = symbols
        self._n_samples = n_samples

    def signal(self) -> np.array:
        """PAM4 信号."""
        level_signal = self.to_level(self._symbols)

        signals = np.array([0] * self._n_samples)
        gap = self._n_samples // len(level_signal)

        ones = np.array([1] * gap)
        for i in range(len(level_signal)):
            signals[i * gap: (i + 1) * gap] = ones * level_signal[i]
        Idrive = signals / max(signals)

        Idrive = Idrive - np.mean(Idrive)
        PAM4 = Idrive[::32]
        return PAM4

    @staticmethod
    def to_level(symbol_pattern: np.ndarray) -> np.ndarray:
        """根据符号序列产生PAM4电平信号"""
        signals = np.array([0] * (len(symbol_pattern) // 2))
        for i in range(len(signals)):
            later, prev = symbol_pattern[2 * i + 1], symbol_pattern[2 * i]
            if prev == 0 and later == 0:
                signals[i] = 0
            elif prev == 1 and later == 0:
                signals[i] = 1
            elif prev == 1 and later == 1:
                signals[i] = 2
            elif prev == 0 and later == 1:
                signals[i] = 3
        return signals

    def to_csv(self, save_path: str):
        """保存到 CSV 文件."""
        signal: np.ndarray = self.signal()
        np.savetxt(save_path, signal, fmt="%.18f")
