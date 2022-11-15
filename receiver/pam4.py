# !usr/bin/env python
# -*- encoding: utf-8 -*-
"""PAM4 接受端.

@File: pam4.py
@Time: 2022/11/13 16:05:38
@Author: Crayon112
@SoftWare: VSCode
@Description: PAM4 接受端.

"""

import math
import numpy as np
from typing import Tuple


DEFAULT_SAMPLE_NUMBER = 2000

DEFUALT_DELAY = 1

DEFAULT_TAP = 31

DEFAULT_STEP = 0.0005

DEFAULT_PIECE = 500


def symbol2stream(symbols: np.ndarray) -> np.ndarray:
    """将PAM符号序列转换为电平信号表示."""
    symbol_map = {3: [0, 1], 1: [1, 1], -3: [0, 0], -1: [1, 0]}
    stream = []
    for symbol in symbols:
        symbol = int(symbol)
        stream.append(symbol_map[symbol])
    return np.array(stream)


def _shift_decison_signal(
    decision_signal: np.ndarray,
    train_symbols: np.ndarray,
    n_sample=None,
):
    """将决策信号频移消除，转为基带信号."""
    n_sample = DEFAULT_SAMPLE_NUMBER if n_sample is None else n_sample
    train_symbols = train_symbols[:n_sample]

    xp, xs, xm = [], [], []
    for idx in range(n_sample // 2):
        _xp = np.sum(np.conj(decision_signal[idx: idx+n_sample] * train_symbols))
        _xs = np.sqrt(np.sum(decision_signal[idx: idx+n_sample] ** 2) * np.sum(train_symbols ** 2))
        _xm = (_xp / _xs) ** 2
        xp.append(_xp)
        xs.append(_xs)
        xm.append(_xm)
    xm = np.array(xm)
    max_pos = np.argmax(xm)
    return np.concatenate([
        decision_signal[max_pos:],
        decision_signal[:max_pos],
    ], axis=0)


def power2symbol(
    power_signal: np.ndarray,
    train_symbols: np.ndarray,
    n_sample=None,
) -> np.ndarray:
    """将功率信号进行决策转换为符号."""
    threshold_x2 = np.mean(power_signal)
    threshold_x3 = np.mean(power_signal[power_signal > threshold_x2])
    threshold_x1 = np.mean(power_signal[power_signal < threshold_x2])

    decision_signal = np.zeros_like(power_signal)
    decision_signal[power_signal <= threshold_x1] = -3
    decision_signal[(power_signal > threshold_x1) & (power_signal <= threshold_x2)] = -1
    decision_signal[(power_signal > threshold_x2) & (power_signal <= threshold_x3)] = 1
    decision_signal[power_signal > threshold_x3] = 3
    return _shift_decison_signal(decision_signal, train_symbols, n_sample)


def summary(out_symbols: np.ndarray, train_symbols: np.ndarray):
    """输出符号与训练符号之间结果比较."""
    train_symbols = train_symbols[:len(out_symbols)]

    error_summary = dict()

    # symbol error
    symbol_error = (out_symbols != train_symbols).astype(int)
    error_summary["symbol_error"] = symbol_error

    # decision error
    out_stream = symbol2stream(out_symbols)
    train_stream = symbol2stream(train_symbols)
    decision_error = (out_stream[:, 0] != train_stream[:, 0]).astype(int)
    error_summary["decision_error"] = decision_error

    zeros = np.array([0] * len(out_symbols))
    # bit error
    low, middle, high = zeros.copy(), zeros.copy(), zeros.copy()
    for idx, (o_stream, t_stream) in enumerate(zip(out_stream, train_stream)):
        if o_stream[0] != t_stream[0]:
            if o_stream[-1] == 0:
                low[idx] = 1
            else:
                high[idx] = 1
        if o_stream[-1] != t_stream[-1]:
            middle[idx] = 1

    error_summary["low"] = low
    error_summary["middle"] = middle
    error_summary["high"] = high

    # BER
    error_summary["BER"] = 0.5 * np.sum(decision_error) / len(decision_error)
    return error_summary


class PAM4Receiver(object):

    def __init__(self, in_signal: np.ndarray) -> None:
        """接受信号为模拟信号.

        Notes
        =====
        需要预处理为 [-3, 3] 区间内

        """
        self.in_signal = in_signal
        self.out_signal = None

    def equalize(
        self,
        in_signal: np.ndarray,
        train_symbols: np.ndarray,
        hxx: np.ndarray,
        delay: int = None,
        tap: int = None,
        piece: int = None,
        step: float = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """均衡过程：模拟信号转换为功率信号."""
        delay = DEFUALT_DELAY if delay is None else delay
        tap = DEFAULT_TAP if tap is None else tap
        step = DEFAULT_STEP if step is None else step
        piece = DEFAULT_PIECE if piece is None else piece

        # 主逻辑
        init_id = 0
        slice_ = len(in_signal) // piece * piece

        zeros = np.array([0.0] * slice_)
        tap_signal = np.array([0.0] * tap)

        delay_ = math.ceil((tap - 1) / 4)
        power_signal = np.concatenate([
            np.array([0.0] * (delay + delay_)),
            train_symbols,
        ], axis=0)

        for _ in range(tap - 1):
            out_signal = zeros.copy()
            tap_signal[init_id] = in_signal[0]
            for idx in range(0, slice_, piece):
                for piece_id in range(piece):
                    cur_id = idx + piece_id
                    out_signal[cur_id] = tap_signal.dot(hxx)
                    if cur_id % 2 == 1:
                        update_value = power_signal[cur_id // 2] - out_signal[cur_id]
                        hxx += step * update_value * tap_signal
                    tap_signal[1:] = tap_signal[:-1]
                    tap_signal[init_id] = in_signal[cur_id + 1]
        return hxx, out_signal

    def _init_hxx(self,  hxx: np.ndarray = None, tap: int = None):
        """初始化抽头系数."""
        if tap is None:
            tap = DEFAULT_TAP
        if hxx is None:
            hxx_ = np.array([0.0] * tap)
            hxx_[tap - tap // 2 - 1] = 1
        else:
            hxx_ = np.array([0.0] * tap)
            hxx_[0: min(tap, len(hxx))] = hxx[0: min(tap, len(hxx))]
        return hxx_

    def run(
        self,
        train_symbols: np.ndarray,
        hxx: np.ndarray = None,
        delay: int = None,
        tap: int = None,
        piece: int = None,
        step: float = None,
    ):
        """接收机数字信号处理.
        Params
        ======
        train_symbols {np.ndarray}: 训练序列，需要预处理为 [-3, 3] 区间内
        h_xx {np.ndarray}: 传递函数
        delay {int}: 输入信号的初始频移
        tap {int}: 抽头系数
        piece {int}: 在均衡过程中的窗口包含的样本数
        step {float}: 均衡器的更新步长

        Returns
        =======
        {dict}: 接收机的误码率以及各项指标

        """

        hxx = self._init_hxx(hxx, tap)

        hxx, power_signal = self.equalize(
            self.in_signal, train_symbols,
            hxx, delay=delay, tap=tap,
            piece=piece, step=step,
        )

        self.out_signal = power_signal[1::2]
        out_symbols = power2symbol(self.out_signal, train_symbols)
        return summary(out_symbols, train_symbols)

    def to_csv(self, save_path: str):
        """保存到 CSV 文件."""
        if self.out_signal is None:
            return
        np.savetxt(save_path, self.out_signal, fmt="%.18f")
