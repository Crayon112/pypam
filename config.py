# region Transmitter
prbs = 2 ** 16

n_bits_per_symbol = 2
n_samples_per_bit = 16
n_samples_per_symbol = n_samples_per_bit * n_bits_per_symbol  # 32

n_bits = n_bits_per_symbol * prbs  # 2 * 2 ** 16
n_symbols = n_bits // n_bits_per_symbol  # 2 ** 16
n_samples = n_samples_per_symbol * n_symbols  # 32 * 2 ** 16

# endregion Transmitter
