import numpy as np
from typing import List


def get_pam_threshold(signal: np.ndarray, pam_level: int, up=False) -> List[float]:
    """获取 PAM 信号的判决门限.

    Paramters
    ========
    signal {np.ndarray}: PAM 电平信号
    pam_level {int}: PAM 信号的 Level - 必定为 2 的 K 次幂
    up {bool}: 是否使用上门限

    Returns
    =======
    {List[float]}: 长度为 Level 的判决上门限
    """

    def generate_threshold(s, level):
        mean = np.mean(s)
        if level == 2:
            return [mean]

        left, right = s[s < mean], s[s > mean]
        return generate_threshold(left, level // 2) + \
            [mean] + \
            generate_threshold(right, level // 2)

    if not up:
        return [0] + generate_threshold(signal, pam_level)
    return generate_threshold(signal, pam_level) + [np.max(signal)]


def q_square_factor(signal: np.ndarray, level: int = 4) -> float:
    r"""Q方因子

    Notes
    =====
        Q方因子定义如下，给定 K - 1 个判决门限，其中第 k 个门限内（包含上门限）的信号
    标准差为 :latex:`\sigma_k`， 均值为 :latex:`\mu_k`， 则Q方因子可以用
    :latex:`\frac{1}{K}\sum_{k=0}^{K-2}(\frac{\sigma_(k+1)+\sigma(k)}{\mu_(k+1)-\mu(k)})^{2}`
    """
    # 各个判断门限中的信号
    levels = np.array([
        np.array(signal[signal <= decision])
        for decision in get_pam_threshold(signal, level, up=True)
    ])

    sigma = np.var(levels, axis=0)  # 判决门限内的标准差
    means = np.mean(levels, axis=0)  # 判决门限内的均值

    q = (sigma[:-1, ...] + sigma[1:, ...]) / (means[:-1, ...] - means[1:, ...])
    q_square_factor = np.sum(q ** 2) / level
    return q_square_factor


def ecia(in_signal, train_symbols):
    """ECIA 算法."""
    ...


if __name__ == '__main__':
    n = 16
    print(get_pam_threshold(np.array([i for i in range(n)]), n, up=True))
