import numpy as np
from . import VanDerCorput as VDC
from primePy import primes

def Halton_Sequence(dimensions, amount):
    N, d = amount, dimensions
    Primes = primes.first(d)
    return np.array([VDC.VanDerCorput_Sequence(Primes[i], N) for i in range(d)])