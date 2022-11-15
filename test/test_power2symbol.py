import os
import sys
sys.path.append(".")
from data import train_symbols
from utils import parse
from receiver.pam4 import power2symbol, summary


curdir = os.path.dirname(__file__)


output = parse(os.path.join(curdir, '../out/output.csv'), delimiter='\n')

out = power2symbol(output, train_symbols)

print(summary(out, train_symbols))
