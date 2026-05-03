import matplotlib.pyplot as plt
import numpy as np
import time
from scipy.stats import norm
from Simulation_and_Valuation import (BrownianMotion as BrMo,
                                      Option_Pricing_Approximately as OpPrAp,
                                      Option_Pricing_Analyticaly as OpPrAn)

from Minimal_Discrepancy_Sequences import (Halton as Hlt,
                                           Faure as Fre,
                                           rand_Perm as rP)
dimensions = 4
Exercise_Time = 1 / 12
N = 10
Timepoints_for_BM = np.arange(1, dimensions + 1) / dimensions  # Excluding leading zeros
print(Timepoints_for_BM)
Timepoints_for_PP = np.arange(dimensions + 1) * (Exercise_Time / dimensions)  # Including leading zeros
print(Timepoints_for_PP)
Pseudo_Rand = np.random.normal(0, 1, (dimensions, N))
Brownian_Motion = BrMo.Forward_Method(Pseudo_Rand, Timepoints_for_BM)
print(Brownian_Motion)