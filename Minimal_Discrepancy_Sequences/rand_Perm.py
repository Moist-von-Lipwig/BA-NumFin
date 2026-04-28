import numpy as np
from scipy.special import comb
from . import VanDerCorput as vdc
from . import Faure as fre
from . import Halton as hlt

def gen_rand_perm(n):
    #Fisher-Yates-Shuffle for 0 to n
    Permutation = np.array(range(n+1))
    for i in reversed(range(1, n+1)):
        z = np.random.randint(0, i)
        Permutation[[i-1, z-1]] = Permutation[[z-1, i-1]]
    return Permutation

def PrimaryGeneratorMatrix(mat_dim):
    i, j = np.indices((mat_dim, mat_dim))
    return comb(j, i)

def BasePExpansion(prime, Amount):
    k = 1
    while (prime**k-Amount <= 0):
        k = k+1
    Numbers = np.arange(Amount + 1).reshape(Amount + 1, 1)
    vec = prime ** np.arange(0, k).reshape(1, k)
    return (Numbers // vec) % prime

def permutate(List, Perm):
    mapped = np.vectorize(lambda x: Perm[x])
    return mapped(List)

def Rand_Perm_Faure_Sequence(prime, amount, dimension, Permutation):
    if prime < dimension:
        print('Error - prime too small')
        Output = 0
    else:
        p, N, d = prime, amount, dimension
        BPExp_orig = BasePExpansion(p, N)
        BPExp_orig_perm = permutate(np.int_(BPExp_orig), Permutation)
        mat_dim = np.size(BasePExpansion(p, N)[0])
        PrimaryGenMat = PrimaryGeneratorMatrix(mat_dim)
        G_Mat = np.eye(mat_dim)
        b_vec = 1 / p ** np.arange(1, mat_dim + 1)
        Output = np.array([BPExp_orig @ b_vec])
        for i in range(1, d):
            G_Mat = PrimaryGenMat @ G_Mat
            BPExp_twisted = BPExp_orig_perm @ G_Mat.T % p
            sub_Faure = BPExp_twisted @ b_vec
            Output = np.append(Output, np.array([sub_Faure]), axis=0)
    return Output