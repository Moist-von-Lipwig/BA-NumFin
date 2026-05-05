import matplotlib.pyplot as plt
import numpy as np
import time
from scipy.stats import norm
from Simulation_and_Valuation import (BrownianMotion as BrMo,
                                      Option_Pricing_Approximately as OpPrAp)

from Minimal_Discrepancy_Sequences import (Halton as Hlt,
                                           rand_Perm as rP)

total_time = time.time()

Start_Price = 100
Interest = 0.05
Volatility = 0.2
Strike = 60
Exercise_Time = 1
dimensions = 30

Paths = [100, 1000, 10000, 100000, 1000000]
# Paths = [100, 1000, 10000, 100000]
Number_Paths = np.size(Paths)
Number_Estimations = 100

Number_Shifts = 10

rand_Faure_prime = 31

Timepoints_for_BM = np.arange(1, dimensions + 1) / dimensions  # Excluding leading zeros
Timepoints_for_PP = np.arange(dimensions + 1) * (Exercise_Time / dimensions)  # Including leading zeros

MC_Solutions = np.zeros(Number_Estimations)
MC_Variances = np.zeros(Number_Paths)

rQMC_1Hlt_Solutions = np.zeros(Number_Estimations)
rQMC_1Hlt_Variances = np.zeros(Number_Paths)

rQMC_2Hlt_Solutions = np.zeros(Number_Estimations)
rQMC_2Hlt_Variances = np.zeros(Number_Paths)

rQMC_Fre_Solutions = np.zeros(Number_Estimations)
rQMC_Fre_Variances = np.zeros(Number_Paths)

i = 0
for N in Paths:
    start_time = time.time()
    # Generating Halton
    Halton_Sequence = Hlt.Halton_Sequence(dimensions, N)[:, 1:]
    for j in range(Number_Estimations):
        # # # Monte Carlo Method # # #
        # Generate pseudo random numbers
        Pseudo_Rand = np.random.normal(0, 1, (dimensions, N))
        # Transform them into a brownian motion
        Brownian_Motion = BrMo.Forward_Method(Pseudo_Rand, Timepoints_for_BM)
        # Generate price processes based on the brownian motion
        Price_Process = OpPrAp.PriceProcesses(Start_Price, Interest, Volatility, Timepoints_for_PP, Brownian_Motion)
        # Calculate payoff for each brownian motion and estimate mean and variance
        MC_Solutions[j], _ = OpPrAp.Lookback_Opt_Approx(Price_Process, Strike, Interest,
                                                        Exercise_Time, 'Call', 'Fixed')

        rQMC_shift_rand_Vector = np.array([np.random.uniform(0, 1, dimensions) for shift in range(Number_Shifts)])

        # # # Randomized Quasi Monte Carlo Method with Halton sequence shifted once # # #
        # Generate randomized quasi random numbers
        Halton_Sequence_transposed = Halton_Sequence.T
        r1_Halton_Sequence = np.add(Halton_Sequence_transposed, rQMC_shift_rand_Vector[0]).T % 1
        r1_Halton_Sequence = norm.ppf(r1_Halton_Sequence[:, 1:])
        # Transform them into a brownian motion
        Brownian_Motion = BrMo.Forward_Method(r1_Halton_Sequence, Timepoints_for_BM)
        # Generate price processes based on the brownian motion
        Price_Process = OpPrAp.PriceProcesses(Start_Price, Interest, Volatility, Timepoints_for_PP, Brownian_Motion)
        # Calculate payoff for each brownian motion and estimate mean and variance
        rQMC_1Hlt_Solutions[j], _ = OpPrAp.Lookback_Opt_Approx(Price_Process, Strike, Interest,
                                                               Exercise_Time, 'Call', 'Fixed')

        # # # Randomized Quasi Monte Carlo Method with Halton sequence shifted several times # # #
        # Generate randomized quasi random numbers
        Halton_Sequence_for_Shift = Hlt.Halton_Sequence(dimensions, N // Number_Shifts)[:, 1:]
        Halton_Sequence_norm = norm.ppf(Halton_Sequence_for_Shift)
        # Transform them into a brownian motion
        Brownian_Motion = BrMo.Forward_Method(Halton_Sequence_norm, Timepoints_for_BM)
        # Generate price processes based on the brownian motion
        Price_Process = OpPrAp.PriceProcesses(Start_Price, Interest, Volatility, Timepoints_for_PP, Brownian_Motion)
        for shift in range(Number_Shifts - 1):
            r2_Halton_Sequence = np.add(Halton_Sequence_for_Shift.T, rQMC_shift_rand_Vector[shift])
            r2_Halton_Sequence = norm.ppf(r2_Halton_Sequence.T % 1)
            # Transform them into a brownian motion
            Brownian_Motion = BrMo.Forward_Method(r2_Halton_Sequence, Timepoints_for_BM)
            # Generate price processes based on the brownian motion
            Price_Process_shifted = OpPrAp.PriceProcesses(Start_Price, Interest, Volatility, Timepoints_for_PP,
                                                          Brownian_Motion)
            Price_Process = np.append(Price_Process, Price_Process_shifted, axis=0)
        # Calculate payoff for each brownian motion and estimate mean and variance
        rQMC_2Hlt_Solutions[j], _ = OpPrAp.Lookback_Opt_Approx(Price_Process, Strike, Interest,
                                                               Exercise_Time, 'Call', 'Fixed')

        # # # Randomized Quasi Monte Carlo Method with Faure sequence using permutations # # #
        # Generate randomized quasi random numbers
        # Permutation = rP.gen_rand_perm(rand_Faure_prime - 1)
        Permutation = np.random.permutation(rand_Faure_prime)
        rand_Faure_Sequence = rP.Rand_Perm_Faure_Sequence(rand_Faure_prime, N, dimensions, Permutation)[:, 1:]
        rand_Faure_Sequence = norm.ppf(rand_Faure_Sequence)
        # Transform them into a brownian motion
        Brownian_Motion = BrMo.Forward_Method(rand_Faure_Sequence, Timepoints_for_BM)
        # Generate price processes based on the brownian motion
        Price_Process = OpPrAp.PriceProcesses(Start_Price, Interest, Volatility, Timepoints_for_PP, Brownian_Motion)
        # Calculate payoff for each brownian motion and estimate mean and variance
        rQMC_Fre_Solutions[j], _ = OpPrAp.Lookback_Opt_Approx(Price_Process, Strike, Interest,
                                                              Exercise_Time, 'Call', 'Fixed')

    MC_Variances[i] = np.var(MC_Solutions, ddof=1)
    rQMC_1Hlt_Variances[i] = np.var(rQMC_1Hlt_Solutions, ddof=1)
    rQMC_2Hlt_Variances[i] = np.var(rQMC_2Hlt_Solutions, ddof=1)
    rQMC_Fre_Variances[i] = np.var(rQMC_Fre_Solutions, ddof=1)
    i = i + 1
    print("--- %d Simulations done in %s seconds ---" % (N, time.time() - start_time))

print("--- Total time: %s seconds ---" % (time.time() - total_time))

np.set_printoptions(suppress=True)

print("--- Variance MC ---")
print(MC_Variances)

print("--- Variance rQMC_1Hlt ---")
print(rQMC_1Hlt_Variances)
print("Variance reduction")
print(rQMC_1Hlt_Variances / MC_Variances)

print("--- Variance rQMC_2Hlt ---")
print(rQMC_2Hlt_Variances)
print("Variance reduction")
print(rQMC_2Hlt_Variances / MC_Variances)

print("--- Variance rQMC_Fre ---")
print(rQMC_Fre_Variances)
print("Variance reduction")
print(rQMC_Fre_Variances / MC_Variances)

plt.rcParams["axes.titlesize"] = 30
plt.rcParams["axes.labelsize"] = 25
plt.rcParams["legend.fontsize"] = 20
plt.rcParams["xtick.labelsize"] = 20
plt.rcParams["ytick.labelsize"] = 20

x_Values = Paths

MC_Graph = MC_Variances
plt.loglog(x_Values, MC_Graph, color="b", marker="o", label='MC-Methode')

rQMC_1Hlt_Graph = rQMC_1Hlt_Variances
plt.loglog(x_Values, rQMC_1Hlt_Graph, color="y", marker="<",
           label='rQMC-Methode mit einfacher Verschiebung (Halton)')

rQMC_2Hlt_Graph = rQMC_2Hlt_Variances
plt.loglog(x_Values, rQMC_2Hlt_Graph, color="c", marker=">",
           label='rQMC-Methode mit multipler Verschiebung (Halton)')

rQMC_Fre_Graph = rQMC_Fre_Variances
plt.loglog(x_Values, rQMC_Fre_Graph, color="m", marker="^",
           label='rQMC-Methode mit Permutation (Faure)')

plt.title("Geschätzte Varianz des Schätzers für die fixierte Lookback Call Option")
plt.xlabel("Anzahl Simulationen")
plt.ylabel("Geschätzte Varianz")
plt.grid(axis='y')
plt.legend(loc="lower left")
plt.show()
