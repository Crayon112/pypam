import numpy as np


from config import n_samples
from utils.generate_pam import PamGenerator
from utils.parse_csv import parse
from transmitter.pam4 import PAM4Transmitter
from receiver.pam4 import PAM4Receiver


__choice__ = "R"  # T/R
# __choice__ = "T"  # T/R


def quant(x: np.ndarray, quantiple: float = 1) -> np.ndarray:
    return (np.round(x / quantiple) * quantiple).astype(float)


if __name__ == '__main__':
    pam4_signal = PamGenerator()
    symbol_pattern = np.array(pam4_signal.codes).astype(float)

    train_symbols = np.array(pam4_signal.symbols).astype(float)
    train_symbols = 2 * train_symbols - 3

    if __choice__ == "T":
        transmitter = PAM4Transmitter(symbol_pattern, n_samples)
        transmitter.to_csv("./out/pam4.csv")
    else:
        in_signal = parse("./out/RX_X.csv", delimiter='\n')
        in_signal = quant(in_signal, np.max(in_signal) / (2 ** 6))
        in_signal = in_signal / np.mean(np.abs(in_signal)) * 3 / 2
        in_signal = 2 * in_signal - 3

        receiver = PAM4Receiver(in_signal)
        summary = receiver.run(train_symbols)
        receiver.to_csv("./out/output.csv")
        print(summary)
