import numpy as np
import scipy.stats.mstats as geom
from scipy.stats import norm
import math


def EUR_Opt_Analytic(S0, mu, sigma, Strike, Endtime, Call_Put):
    K, T = Strike, Endtime
    # math.log(x) returns ln(x)
    d_1 = (math.log(S0 / K) + (mu + sigma ** 2 / 2) * T) / (sigma * math.sqrt(T))
    d_2 = d_1 - (sigma * math.sqrt(T))
    if Call_Put == 'Call':
        return S0 * norm.cdf(d_1) - math.exp(-mu * T) * K * norm.cdf(d_2)
    elif Call_Put == 'Put':
        return math.exp(-mu * T) * K * norm.cdf(-d_2) - S0 * norm.cdf(-d_1)
    else:
        print('Error')
        return 0


def disc_geom_Asian_Opt_Analytic(S0, mu, sigma, Strike, Endtime, number_timepoints, alpha):
    # Only for Calls
    K, T, n, a = Strike, Endtime, number_timepoints, alpha
    S_adj = S0 ** a
    sigma_adj = a * sigma * math.sqrt((2 * n + 1) / (6 * (n + 1)))
    mu_adj = (a / 2) * (mu - sigma ** 2 / 2) + (a ** 2 * sigma ** 2 * (2 * n + 1)) / (12 * (n + 1))
    d_1 = (math.log(S_adj / K)) / (sigma_adj * math.sqrt(T)) + (mu_adj / sigma_adj + sigma_adj / 2) * math.sqrt(T)
    d_2 = d_1 - sigma_adj * math.sqrt(T)
    return S_adj * math.exp((mu_adj - mu) * T) * norm.cdf(d_1) - K * math.exp(-mu * T) * norm.cdf(d_2)

def var_Lookback_Opt_Analytic(S0, mu, sigma, Endtime):
    # Only for Calls
    min = S0  # min could later be used to compute prices during the run time of the option
    T = Endtime
    d_1 = (math.log(S0/min) + mu * T + sigma ** 2 * T / 2) / (sigma * math.sqrt(T))
    d_2 = d_1 - sigma * math.sqrt(T)
    d_3 = - d_1 + (2 * mu * math.sqrt(T))/sigma
    Term = math.exp(-mu*T) * (sigma**2 * S0) / (2*mu) * (
           (S0/min)**(-2*mu/sigma**2)*norm.cdf(d_3) - math.exp(mu*T)*norm.cdf(-d_1))
    return S0*norm.cdf(d_1) - math.exp(-mu*T)*min*norm.cdf(d_2) + Term

def fix_Lookback_Opt_Analytic(S0, mu, sigma, Strike, Endtime):
    # Only for Calls
    Max = S0 # Max could later be used to compute prices during the run time of the option
    K, T = Strike, Endtime
    if K >= Max:
        d_1 = (math.log(S0 / K) + mu * T + sigma ** 2 * T / 2) / (sigma * math.sqrt(T))
        d_2 = d_1 - (2 * mu * math.sqrt(T)) / sigma
        Term = math.exp(-mu * T) * (sigma ** 2) / (2 * mu) * S0 * (
                    -(S0 / K) ** (-2 * mu / sigma ** 2) * norm.cdf(d_2) + math.exp(mu * T) * norm.cdf(d_1))
        return S0 * norm.cdf(d_1) - math.exp(-mu * T) * K * norm.cdf(d_1 - sigma * math.sqrt(T)) + Term
    else:
        d_1 = (math.log(S0 / Max) + mu * T + sigma ** 2 * T / 2) / (sigma * math.sqrt(T))
        d_2 = d_1 - (2 * mu * math.sqrt(T)) / sigma
        Term = math.exp(-mu * T) * (sigma ** 2) / (2 * mu) * S0 * (
                -(S0 / Max) ** (-2 * mu / sigma ** 2) * norm.cdf(d_2) + math.exp(mu * T) * norm.cdf(d_1))
        return math.exp(-mu * T) * (Max-K) + S0 * norm.cdf(d_1) - math.exp(-mu * T) * Max * norm.cdf(d_1 - sigma * math.sqrt(T)) + Term