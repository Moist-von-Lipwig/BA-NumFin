import numpy as np

def BasePExpansion(prime, Amount):
    k = 1
    # The numpy arrays all need to be the same size, thus we need to know their max size in advance
    while (prime**k-Amount <= 0):
        k = k+1
    Numbers = np.arange(Amount + 1).reshape(Amount + 1, 1)
    vec = prime ** np.arange(0, k).reshape(1, k)
    return (Numbers // vec) % prime

def VanDerCorput_Sequence(prime, amount):
    p, N = prime, amount
    BPExp = BasePExpansion(p, N)
    lengthList = np.size(BasePExpansion(p, N)[0])
    rad_vec = 1/p ** np.arange(1, lengthList+1)
    out = BPExp @ rad_vec
    return out

#print(VanDerCorput(2,7))