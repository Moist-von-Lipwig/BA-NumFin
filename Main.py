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

total_time = time.time()

# Parameters for European Option
Start_Price = 100
Interest = 0.10
Volatility = 0.20
Strike = 100
Exercise_Time = 3/12

Look_Call_Val = OpPrAn.fix_Lookback_Opt_Analytic(Start_Price, Interest, Volatility, Strike, Exercise_Time)
print("The analytical Value of the fixed Lookback Option is: %.15g" % Look_Call_Val)

Paths = [1000000]
Number_Estimations = np.size(Paths)
dimensions = 1

Timepoints_for_BM = np.arange(1, dimensions + 1) / dimensions  # Excluding leading zeros
Timepoints_for_PP = np.arange(dimensions + 1) / dimensions  # Including leading zeros

# Allocating memory for the solutions
MC_Solutions1 = np.zeros(Number_Estimations)
MC_Variances1 = np.zeros(Number_Estimations)

i = 0  # For indexing the solution lists
for N in Paths:

    # # # Monte Carlo Method # # #
    start_time = time.time()
    # Generate pseudo random numbers
    Pseudo_Rand = np.abs(np.random.normal(0, 3/12, (dimensions, N)))
    # Transform them into a brownian motion
    Brownian_Motion = BrMo.Forward_Method(Pseudo_Rand, Timepoints_for_BM)
    # Generate price processes based on the brownian motion
    Price_Process = OpPrAp.PriceProcesses(Start_Price, Interest, Volatility, Timepoints_for_PP, Brownian_Motion)
    # Calculate payoff for each brownian motion and estimate mean and variance
    MC_Solutions1[i], MC_Variances1[i] = OpPrAp.Lookback_Opt_Approx(Price_Process, Strike, Interest,
                                                                  Exercise_Time, 'Call', 'Fixed')
    print("--- %d Simulations for MC-Method done in %s seconds ---" % (N, time.time() - start_time))

print(MC_Solutions1)