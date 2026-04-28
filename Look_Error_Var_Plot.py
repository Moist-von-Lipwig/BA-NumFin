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

# Parameters for Lookback Option
Start_Price = 100
Interest = 0.10
Volatility = 0.20
Strike = 100
Exercise_Time = 3/12

Look_Call_Val = OpPrAn.fix_Lookback_Opt_Analytic(Start_Price, Interest, Volatility, Strike, Exercise_Time)
print("The analytical Value of the fixed Lookback Call Option is: %.15g" % Look_Call_Val)

Paths = [10, 100, 1000, 10000, 100000, 1000000]  # , 10000000]
Number_Estimations = np.size(Paths)
dimensions = 12

# For Faure
Faure_prime = 13

# For shifted Halton
Number_Shifts = 10
rQMC_shift_rand_Vector = np.array([np.random.uniform(0, 1, dimensions) for shift in range(Number_Shifts)])

# For randomized Faure
rand_Faure_prime = 13
Permutation = rP.gen_rand_perm(rand_Faure_prime - 1)

Timepoints_for_BM = np.arange(1, dimensions + 1) / dimensions  # Excluding leading zeros
Timepoints_for_PP = np.arange(dimensions + 1) / dimensions  # Including leading zeros

# Allocating memory for the solutions
MC_Solutions = np.zeros(Number_Estimations)
MC_Variances = np.zeros(Number_Estimations)

QMC_Hlt_Solutions = np.zeros(Number_Estimations)
QMC_Hlt_Variances = np.zeros(Number_Estimations)

QMC_Fre_Solutions = np.zeros(Number_Estimations)
QMC_Fre_Variances = np.zeros(Number_Estimations)

rQMC_1Hlt_Solutions = np.zeros(Number_Estimations)
rQMC_1Hlt_Variances = np.zeros(Number_Estimations)

rQMC_2Hlt_Solutions = np.zeros(Number_Estimations)
rQMC_2Hlt_Variances = np.zeros(Number_Estimations)

rQMC_Fre_Solutions = np.zeros(Number_Estimations)
rQMC_Fre_Variances = np.zeros(Number_Estimations)

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
    MC_Solutions[i], MC_Variances[i] = OpPrAp.Lookback_Opt_Approx(Price_Process, Strike, Interest,
                                                                  Exercise_Time, 'Call', 'Fixed')
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
    QMC_Hlt_Solutions[i], QMC_Hlt_Variances[i] = OpPrAp.Lookback_Opt_Approx(Price_Process, Strike, Interest,
                                                                            Exercise_Time, 'Call', 'Fixed')
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
    QMC_Fre_Solutions[i], QMC_Fre_Variances[i] = OpPrAp.Lookback_Opt_Approx(Price_Process, Strike, Interest,
                                                                            Exercise_Time, 'Call', 'Fixed')
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
    rQMC_1Hlt_Solutions[i], rQMC_1Hlt_Variances[i] = OpPrAp.Lookback_Opt_Approx(Price_Process, Strike, Interest,
                                                                                Exercise_Time, 'Call', 'Fixed')
    print("--- %d Simulations for rQMC-Method with Halton shifted once done in %s seconds ---"
          % (N, time.time() - start_time))

    # # # Randomized Quasi Monte Carlo Method with Halton sequence shifted several times # # #
    start_time = time.time()
    # Generate randomized quasi random numbers
    shifted_Values = np.zeros(Number_Shifts)
    shifted_Variance = np.zeros(Number_Shifts)
    for shift in range(Number_Shifts):
        r2_Halton_Sequence = np.add(Halton_Sequence.T, rQMC_shift_rand_Vector[shift])
        r2_Halton_Sequence = norm.ppf(r2_Halton_Sequence.T % 1)
        # Transform them into a brownian motion
        Brownian_Motion = BrMo.Forward_Method(r2_Halton_Sequence, Timepoints_for_BM)
        # Generate price processes based on the brownian motion
        Price_Process = OpPrAp.PriceProcesses(Start_Price, Interest, Volatility, Timepoints_for_PP, Brownian_Motion)
        # Calculate payoff for each brownian motion and estimate mean and variance
        shifted_Values[shift], shifted_Variance[shift] = OpPrAp.Lookback_Opt_Approx(Price_Process, Strike, Interest,
                                                                                    Exercise_Time, 'Call', 'Fixed')
    rQMC_2Hlt_Solutions[i], rQMC_2Hlt_Variances[i] = np.mean(shifted_Values), np.var(shifted_Variance, ddof=1)
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
    rQMC_Fre_Solutions[i], rQMC_Fre_Variances[i] = OpPrAp.Lookback_Opt_Approx(Price_Process, Strike, Interest,
                                                                              Exercise_Time, 'Call', 'Fixed')
    print("--- %d Simulations for rQMC-Method with Faure using permutations done in %s seconds ---"
          % (N, time.time() - start_time))

    i = i + 1

print("--- Total time: %s seconds ---" % (time.time() - total_time))

print(MC_Solutions)

# # # Generating the error plot # # #

# Adjust font sizes for both plots
plt.rcParams["axes.titlesize"] = 30
plt.rcParams["axes.labelsize"] = 25
plt.rcParams["legend.fontsize"] = 20
plt.rcParams["xtick.labelsize"] = 20
plt.rcParams["ytick.labelsize"] = 20

x_Values = Paths

#Theoretical_Convergence = np.array(Paths) ** (-1 / 2)
#plt.loglog(x_Values, Theoretical_Convergence, color="b", marker="x", label='MC-Methode (theoretische Konvergenz)')

MC_Graph = np.abs(MC_Solutions - Look_Call_Val)
plt.loglog(x_Values, MC_Graph, color="b", marker="o", label='MC-Methode')

QMC_Hlt_Graph = np.abs(QMC_Hlt_Solutions - Look_Call_Val)
plt.loglog(x_Values, QMC_Hlt_Graph, color="r", marker="D", label='QMC-Methode mit Halton')

QMC_Fre_Graph = np.abs(QMC_Fre_Solutions - Look_Call_Val)
plt.loglog(x_Values, QMC_Fre_Graph, color="g", marker="s", label='QMC-Methode mit Faure')

rQMC_1Hlt_Graph = np.abs(rQMC_1Hlt_Solutions - Look_Call_Val)
plt.loglog(x_Values, rQMC_1Hlt_Graph, color="y", marker="<",
           label='rQMC-Methode mit einfacher Verschiebung (Halton)')

rQMC_2Hlt_Graph = np.abs(rQMC_2Hlt_Solutions - Look_Call_Val)
plt.loglog(x_Values, rQMC_2Hlt_Graph, color="c", marker=">",
           label='rQMC-Methode mit multipler Verschiebung (Halton)')

rQMC_Fre_Graph = np.abs(rQMC_Fre_Solutions - Look_Call_Val)
plt.loglog(x_Values, rQMC_Fre_Graph, color="m", marker="^",
           label='rQMC-Methode mit Permutation (Faure)')

plt.title("Geschätzte Fehlerraten für die fixed Lookback Call Option")
plt.xlabel("Anzahl Simulationen")
plt.ylabel("Geschätzter Fehler")
plt.grid(axis='y')
plt.legend(loc="lower left")
plt.show()

# # # Generating the variance plot # # #

x_Values = Paths

#Theoretical_Convergence = 1 / np.array(Paths)
#plt.loglog(x_Values, Theoretical_Convergence, color="b", marker="X", label='MC-Methode (theoretische Konvergenz)')

MC_Graph = MC_Variances / Paths
plt.loglog(x_Values, MC_Graph, color="b", marker="o", label='MC-Methode')

QMC_Hlt_Graph = QMC_Hlt_Variances / Paths
plt.loglog(x_Values, QMC_Hlt_Graph, color="r", marker="D", label='QMC-Methode mit Halton')

QMC_Fre_Graph = QMC_Fre_Variances / Paths
plt.loglog(x_Values, QMC_Fre_Graph, color="g", marker="s", label='QMC-Methode mit Faure')

rQMC_1Hlt_Graph = rQMC_1Hlt_Variances / Paths
plt.loglog(x_Values, rQMC_1Hlt_Graph, color="y", marker="<",
           label='rQMC-Methode mit einfacher Verschiebung (Halton)')

rQMC_2Hlt_Graph = rQMC_2Hlt_Variances / Paths
plt.loglog(x_Values, rQMC_2Hlt_Graph, color="c", marker=">",
           label='rQMC-Methode mit multipler Verschiebung (Halton)')

rQMC_Fre_Graph = rQMC_Fre_Variances / Paths
plt.loglog(x_Values, rQMC_Fre_Graph, color="m", marker="^",
           label='rQMC-Methode mit Permutation (Faure)')

plt.title("Geschätzte Varianzraten für die fixed Lookback Call Option")
plt.xlabel("Anzahl Simulationen")
plt.ylabel("Geschätzte Varianz")
plt.grid(axis='y')
plt.legend(loc="lower left")
plt.show()
