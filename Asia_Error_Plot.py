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

# Parameters for Asian Option
Start_Price = 100
Interest = 0.05
Volatility = 0.2
Strike = 60
Exercise_Time = 1
dimensions = 12  # The geom. average is taken over dimensions + 1 many timepoints

Asia_Call_Val = OpPrAn.disc_geom_Asian_Opt_Analytic(Start_Price, Interest, Volatility, Strike, Exercise_Time,
                                                    dimensions,
                                                    1)
print("The analytical Value of the discrete arithmetic Asian Call Option is:")
print(Asia_Call_Val)

Paths = [100, 1000, 10000, 100000, 1000000, 10000000]
# Paths = [100, 1000, 10000, 100000, 1000000]
Number_Paths = np.size(Paths)

# For Faure; must be greater than dimensions
Faure_prime = 13

# For shifted Halton
Number_Shifts = 10
rQMC_shift_rand_Vector = np.array([np.random.uniform(0, 1, dimensions) for shift in range(Number_Shifts)])

# For randomized Faure; must be greater than dimensions
rand_Faure_prime = 13
# Permutation = rP.gen_rand_perm(rand_Faure_prime - 1)
Permutation = np.random.permutation(rand_Faure_prime)

Timepoints_for_BM = np.arange(1, dimensions + 1) / dimensions  # Excluding leading zeros
Timepoints_for_PP = np.arange(dimensions + 1) * (Exercise_Time / dimensions)  # Including leading zeros

# Allocating memory for the solutions
MC_Solutions = np.zeros(Number_Paths)

QMC_Hlt_Solutions = np.zeros(Number_Paths)

QMC_Fre_Solutions = np.zeros(Number_Paths)

rQMC_1Hlt_Solutions = np.zeros(Number_Paths)

rQMC_2Hlt_Solutions = np.zeros(Number_Paths)

rQMC_Fre_Solutions = np.zeros(Number_Paths)

i = 0  # For indexing the solution lists
for N in Paths:

    # # # Monte Carlo Method # # #
    start_time = time.time()
    # Generate pseudo random numbers
    Pseudo_Rand = np.random.normal(0, 1, (dimensions, N))
    # Transform them into a brownian motion
    Brownian_Motion = BrMo.Forward_Method(Pseudo_Rand, Timepoints_for_BM)
    # Generate price processes based on the brownian motion
    Price_Process = OpPrAp.PriceProcesses(Start_Price, Interest, Volatility, Timepoints_for_PP, Brownian_Motion)
    # Calculate payoff for each brownian motion and estimate mean and variance
    MC_Solutions[i], _ = OpPrAp.Asian_Opt_Approx(Price_Process, Strike, Interest, Exercise_Time,
                                                 'Call', 'Discrete', 'Geometric')
    print("--- %d Simulations for MC-Method done in %s seconds ---" % (N, time.time() - start_time))

    # # # Quasi Monte Carlo Method with Halton sequence# # #
    start_time = time.time()
    # Generate quasi random numbers
    Halton_Sequence = Hlt.Halton_Sequence(dimensions, N)[:, 1:]
    Halton_Sequence_norm = norm.ppf(Halton_Sequence)
    # Transform them into a brownian motion
    Brownian_Motion = BrMo.Forward_Method(Halton_Sequence_norm, Timepoints_for_BM)
    # Generate price processes based on the brownian motion
    Price_Process = OpPrAp.PriceProcesses(Start_Price, Interest, Volatility, Timepoints_for_PP, Brownian_Motion)
    # Calculate payoff for each brownian motion and estimate mean and variance
    QMC_Hlt_Solutions[i], _ = OpPrAp.Asian_Opt_Approx(Price_Process, Strike, Interest, Exercise_Time,
                                                      'Call', 'Discrete', 'Geometric')
    print("--- %d Simulations for QMC-Method with Halton done in %s seconds ---" % (N, time.time() - start_time))

    # # # Quasi Monte Carlo Method with Faure sequence# # #
    start_time = time.time()
    # Generate quasi random numbers
    Faure_Sequence = Fre.Faure_Sequence(Faure_prime, N, dimensions)[:, 1:]
    Faure_Sequence = norm.ppf(Faure_Sequence)
    # Transform them into a brownian motion
    Brownian_Motion = BrMo.Forward_Method(Faure_Sequence, Timepoints_for_BM)
    # Generate price processes based on the brownian motion
    Price_Process = OpPrAp.PriceProcesses(Start_Price, Interest, Volatility, Timepoints_for_PP, Brownian_Motion)
    # Calculate payoff for each brownian motion and estimate mean and variance
    QMC_Fre_Solutions[i], _ = OpPrAp.Asian_Opt_Approx(Price_Process, Strike, Interest, Exercise_Time,
                                                      'Call', 'Discrete', 'Geometric')
    print("--- %d Simulations for QMC-Method with Faure done in %s seconds ---" % (N, time.time() - start_time))

    # # # Randomized Quasi Monte Carlo Method with Halton sequence shifted once # # #
    start_time = time.time()
    # Generate randomized quasi random numbers
    Halton_Sequence_transposed = Halton_Sequence.T
    r1_Halton_Sequence = np.add(Halton_Sequence_transposed, rQMC_shift_rand_Vector[0]).T % 1
    r1_Halton_Sequence = norm.ppf(r1_Halton_Sequence[:, 1:])
    # Transform them into a brownian motion
    Brownian_Motion = BrMo.Forward_Method(r1_Halton_Sequence, Timepoints_for_BM)
    # Generate price processes based on the brownian motion
    Price_Process = OpPrAp.PriceProcesses(Start_Price, Interest, Volatility, Timepoints_for_PP, Brownian_Motion)
    # Calculate payoff for each brownian motion and estimate mean and variance
    rQMC_1Hlt_Solutions[i], _ = OpPrAp.Asian_Opt_Approx(Price_Process, Strike, Interest,
                                                        Exercise_Time,
                                                        'Call', 'Discrete', 'Geometric')
    print("--- %d Simulations for rQMC-Method with Halton shifted once done in %s seconds ---"
          % (N, time.time() - start_time))

    # # # Randomized Quasi Monte Carlo Method with Halton sequence shifted several times # # #
    start_time = time.time()
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
    rQMC_2Hlt_Solutions[i], _ = OpPrAp.Asian_Opt_Approx(Price_Process, Strike, Interest,
                                                        Exercise_Time, 'Call',
                                                        'Discrete', 'Geometric')
    print("--- %d Simulations for rQMC-Method with Halton shifted several times done in %s seconds ---"
          % (N, time.time() - start_time))

    # # # Randomized Quasi Monte Carlo Method with Faure sequence using permutations # # #
    start_time = time.time()
    # Generate quasi random numbers
    rand_Faure_Sequence = rP.Rand_Perm_Faure_Sequence(rand_Faure_prime, N, dimensions, Permutation)[:, 1:]
    rand_Faure_Sequence = norm.ppf(rand_Faure_Sequence)
    # Transform them into a brownian motion
    Brownian_Motion = BrMo.Forward_Method(rand_Faure_Sequence, Timepoints_for_BM)
    # Generate price processes based on the brownian motion
    Price_Process = OpPrAp.PriceProcesses(Start_Price, Interest, Volatility, Timepoints_for_PP, Brownian_Motion)
    # Calculate payoff for each brownian motion and estimate mean and variance
    rQMC_Fre_Solutions[i], _ = OpPrAp.Asian_Opt_Approx(Price_Process, Strike, Interest,
                                                       Exercise_Time, 'Call',
                                                       'Discrete', 'Geometric')
    print("--- %d Simulations for rQMC-Method with Faure using permutations done in %s seconds ---"
          % (N, time.time() - start_time))

    i = i + 1

print("--- Total time: %s seconds ---" % (time.time() - total_time))

# # # Generating the error plot # # #

np.set_printoptions(suppress=True)

# Adjust font sizes for both plots
plt.rcParams["axes.titlesize"] = 30
plt.rcParams["axes.labelsize"] = 25
plt.rcParams["legend.fontsize"] = 20
plt.rcParams["xtick.labelsize"] = 20
plt.rcParams["ytick.labelsize"] = 20

x_Values = Paths

MC_Graph = np.abs(MC_Solutions - Asia_Call_Val)
plt.loglog(x_Values, MC_Graph, color="b", marker="o", label='MC-Methode')

print("--- Result MC ---")
print(MC_Solutions)
print("--- Error MC ---")
print(MC_Graph)

QMC_Hlt_Graph = np.abs(QMC_Hlt_Solutions - Asia_Call_Val)
plt.loglog(x_Values, QMC_Hlt_Graph, color="r", marker="D", label='QMC-Methode mit Halton')

print("--- Result QMC_Hlt ---")
print(QMC_Hlt_Solutions)
print("--- Error QMC_Hlt ---")
print(QMC_Hlt_Graph)

QMC_Fre_Graph = np.abs(QMC_Fre_Solutions - Asia_Call_Val)
plt.loglog(x_Values, QMC_Fre_Graph, color="g", marker="s", label='QMC-Methode mit Faure')

print("--- Result QMC_Fre ---")
print(QMC_Fre_Solutions)
print("--- Error QMC_Fre ---")
print(QMC_Fre_Graph)

rQMC_1Hlt_Graph = np.abs(rQMC_1Hlt_Solutions - Asia_Call_Val)
plt.loglog(x_Values, rQMC_1Hlt_Graph, color="y", marker="<",
           label='rQMC-Methode mit einfacher Verschiebung (Halton)')

print("--- Result rQMC_1Hlt ---")
print(rQMC_1Hlt_Solutions)
print("--- Error rQMC_1Hlt ---")
print(rQMC_1Hlt_Graph)

rQMC_2Hlt_Graph = np.abs(rQMC_2Hlt_Solutions - Asia_Call_Val)
plt.loglog(x_Values, rQMC_2Hlt_Graph, color="c", marker=">",
           label='rQMC-Methode mit multipler Verschiebung (Halton)')

print("--- Result rQMC_2Hlt ---")
print(rQMC_2Hlt_Solutions)
print("--- Error rQMC_2Hlt ---")
print(rQMC_2Hlt_Graph)

rQMC_Fre_Graph = np.abs(rQMC_Fre_Solutions - Asia_Call_Val)
plt.loglog(x_Values, rQMC_Fre_Graph, color="m", marker="^",
           label='rQMC-Methode mit Permutation (Faure)')

print("--- Result rQMC_Fre ---")
print(rQMC_Fre_Solutions)
print("--- Error rQMC_Fre ---")
print(rQMC_Fre_Graph)

plt.title("Geschätzte Fehlerraten für die diskrete geometrische Asiatische Call Option")
plt.xlabel("Anzahl Simulationen")
plt.ylabel("Geschätzter Fehler")
plt.grid(axis='y')
plt.legend(loc="lower left")
plt.show()