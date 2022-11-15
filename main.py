from config import n_samples
from data import symbol_pattern
from transmitter.pam4 import PAM4Transmitter
from data import in_signal, train_symbols
from receiver.pam4 import PAM4Receiver


# __choice__ = "T"  # T/R
__choice__ = "R"  # T/R


if __name__ == '__main__':
    if __choice__ == "T":
        transmitter = PAM4Transmitter(symbol_pattern, n_samples)
        transmitter.to_csv("./out/pam4.csv")
    else:
        receiver = PAM4Receiver(in_signal)
        summary = receiver.run(train_symbols)
        receiver.to_csv("./out/output.csv")
        print(summary)
