import math
import random

DEFAULT_LEVEL = 4

DEFAULT_N_SYMBOLS = 2 ** 16


class PamGenerator(object):

    def __init__(self, n_symbols: int = None, level: int = None) -> None:
        self.n_symbols = n_symbols or DEFAULT_N_SYMBOLS
        self.level = level or DEFAULT_LEVEL

        self.code2symbol = {
            code: symbol for symbol, code in
            enumerate(self.gray_code(int(math.log2(self.level))))
        }

        self.symbol2code = {
            symbol: code for code, symbol in
            self.code2symbol.items()
        }

        self.symbols, self.codes = [], []
        for code, symbol in self._generate():
            self.symbols.append(symbol)
            self.codes.extend(list(code))

    def gray_code(self, n: int):
        """长度为 n 的格雷码序列."""
        if n == 1:
            return ['0', '1']
        res = []

        last = self.gray_code(n - 1)
        for c in last:
            res.append('0' + c)
        for c in reversed(res):
            res.append(str(1 - int(c[0])) + c[1:])
        return res

    def _generate(self):
        for _ in range(self.n_symbols):
            seed = random.randint(0, self.level - 1)
            code = self.symbol2code[seed]
            yield code, seed
