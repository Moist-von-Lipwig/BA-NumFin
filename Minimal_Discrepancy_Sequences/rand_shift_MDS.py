import numpy as np
from . import VanDerCorput as vdc
from . import Faure as fre
from . import Halton as hlt

def rand_shift_VDC_Sequence(prime, amount):
    VDCS = vdc.VanDerCorput_Sequence(prime, amount)
    rand_Vec = np.array([np.random.uniform(0,1,1)])
    return np.mod(VDCS + rand_Vec, 1)

def rand_shift_Halton_Sequence(dimensions, amount, rand_Vec):
    if rand_Vec == 'Null':
        rand_Vec = np.array([np.random.uniform(0, 1, dimensions)])
    Halt = hlt.Halton_Sequence(dimensions, amount).T
    Halt_shift = np.add(Halt, rand_Vec)
    return Halt_shift.T % 1

def rand_multi_shift_Halton_Sequence(dimensions, amount, rand_Vec):
    if rand_Vec == 'Null':
        rand_Vec = np.array([np.random.uniform(0, 1, dimensions)])
    Halt = hlt.Halton_Sequence(dimensions, amount).T
    Halt_shift = np.add(Halt, rand_Vec)
    return Halt_shift.T % 1

def rand_shift_Faure_Sequence(prime, amount, dimensions):
    Faur = fre.Faure_Sequence(prime, amount, dimensions)
    rand_Vec = np.array([np.random.uniform(0,1,dimensions)])
    Faur_shift = np.add(Faur, rand_Vec.T)
    return np.mod(Faur_shift, 1)