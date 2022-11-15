# from config import n_samples
# from data import symbol_pattern
# from transmitter.pam4 import PAM4Transmitter

# if __name__ == '__main__':
#     transmitter = PAM4Transmitter(symbol_pattern, n_samples)
#     transmitter.to_csv("E:\source\PAM4\PAM4.csv")

from data import in_signal, train_symbols
from receiver.pam4 import PAM4Receiver


if __name__ == '__main__':
    receiver = PAM4Receiver(in_signal)
    summary = receiver.run(train_symbols, round=1)
    print(summary["BER"])
