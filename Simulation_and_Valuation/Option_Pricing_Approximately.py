import numpy as np
import scipy.stats.mstats as geom
from scipy.stats import norm
import math


def PriceProcesses(S0, mu, sigma, times, W):
    return S0 * np.exp((mu - 0.5 * sigma ** 2) * times + sigma * W)


def EUR_Opt_Approx(PriceProcesses, Strike, Interest, Endtime, Call_Put):
    PP, K, r, T = PriceProcesses, Strike, Interest, Endtime
    if Call_Put == 'Call':
        Value_uncensored = PP[:, -1] - K
        Value = (Value_uncensored) * (Value_uncensored >= 0)
        Value_mean = np.exp(-r * 1) * np.mean(Value)
        # Using sample variance -> meaning division by n-1 (n-ddof)
        Variance = np.var(Value, ddof=1)
        return Value_mean, Variance
    elif Call_Put == 'Put':
        Value_uncensored = K - PP[:, -1]
        Value = (Value_uncensored) * (Value_uncensored >= 0)
        Value_mean = np.exp(-r * 1) * np.mean(Value)
        Variance = np.var(Value, ddof=1)
        return Value_mean, Variance
    else:
        print('Error')
        return 0


def Asian_Opt_Approx(PriceProcesses, Strike, Interest, Time, Call_Put, Disc_Cont, Arith_Geom):
    PP, K, r, T = PriceProcesses, Strike, Interest, Time
    if Call_Put == 'Call':
        if Disc_Cont == 'Discrete':
            if Arith_Geom == 'Arithmetic':
                # Call-Discrete-Arithmetic
                AriAvg = np.mean(PP, axis=1)
                Value_uncensored = (AriAvg - K)
                Value = Value_uncensored * (Value_uncensored >= 0)
                Value_mean = np.exp(-r * T) * np.mean(Value)
                Variance = np.var(Value, ddof=1)
                return Value_mean, Variance
            elif Arith_Geom == 'Geometric':
                # Call-Discrete-Geometric
                GeoAvg = geom.gmean(PP, axis=1)
                Value_uncensored = (GeoAvg - K)
                Value = Value_uncensored * (Value_uncensored >= 0)
                Value_mean = np.exp(-r * T) * np.mean(Value)
                Variance = np.var(Value, ddof=1)
                return Value_mean, Variance
        elif Disc_Cont == 'Continuous':
            if Arith_Geom == 'Arithmetic':
                # Call-Continuous-Arithmetic
                print('To be implemented')
                return 0
            elif Arith_Geom == 'Geometric':
                # Call-Continuous-Geometric
                print('To be implemented')
                return 0
    elif Call_Put == 'Put':
        if Disc_Cont == 'Discrete':
            if Arith_Geom == 'Arithmetic':
                # Put-Discrete-Arithmetic
                print('To be implemented')
                return 0
            elif Arith_Geom == 'Geometric':
                # Put-Discrete-Geometric
                print('To be implemented')
                return 0
        elif Disc_Cont == 'Continuous':
            if Arith_Geom == 'Arithmetic':
                # Put-Continuous-Arithmetic
                print('To be implemented')
                return 0
            elif Arith_Geom == 'Geometric':
                # Put-Continuous-Geometric
                print('To be implemented')
                return 0


def Lookback_Opt_Approx(PriceProcesses, Strike, Interest, Time, Call_Put, Fix_Var):
    PP, K, r, T = PriceProcesses, Strike, Interest, Time
    if Call_Put == 'Call':
        if Fix_Var == 'Fixed':
            # Call-Fixed
            MaxVal = np.max(PP, axis=1)
            Value_uncensored = MaxVal - K
            Value = Value_uncensored * (Value_uncensored >= 0)
            Value_mean = np.exp(-r * T) * np.mean(Value)
            Variance = np.var(Value, ddof=1)
            return Value_mean, Variance
        elif Fix_Var == 'Variable':
            # Call-Variable
            MinVal = np.min(PP, axis=1)
            Value_uncensored = PP[:, -1] - MinVal
            Value = Value_uncensored * (Value_uncensored >= 0)
            Value_mean = np.exp(-r * T) * np.mean(Value)
            Variance = np.var(Value, ddof=1)
            return Value_mean, Variance
    elif Call_Put == 'Put':
        if Fix_Var == 'Fixed':
            # Put-Fixed
            print('To be implemented')
            return 0
        elif Fix_Var == 'Variable':
            # Put-Variable
            print('To be implemented')
            return 0


def Barrier_Opt_Approx(PriceProcesses, Strike, Barrier, Interest, Time, Call_Put, Down_Up, In_Out):
    PP, K, r, T, H = PriceProcesses, Strike, Interest, Time, Barrier
    if Call_Put == 'Call':
        if Down_Up == 'Down':
            if In_Out == 'In':
                # Call-Down-In
                In_Vector = np.transpose([np.any(PP < H, 1)])
                DownInPP = In_Vector * PP
                Value_uncensored = DownInPP[:, -1] - K
                Value = Value_uncensored * (Value_uncensored >= 0)
                Value_mean = np.exp(-r * T) * np.mean(Value)
                Variance = np.var(Value, ddof=1)
                return Value_mean, Variance
            elif In_Out == 'Out':
                # Call-Down-Out
                Out_Vector = np.transpose([np.all(PP > H, 1)])
                DownOutPP = Out_Vector * PP
                Value_uncensored = DownOutPP[:, -1] - K
                Value = (Value_uncensored) * (Value_uncensored >= 0)
                Value_mean = np.exp(-r * T) * np.mean(Value)
                Variance = np.var(Value, ddof=1)
                return Value_mean, Variance
        elif Down_Up == 'Up':
            if In_Out == 'In':
                # Call-Up-In
                print('To be implemented')
                return 0
            elif In_Out == 'Out':
                # Call-Up-Out
                print('To be implemented')
                return 0
    elif Call_Put == 'Put':
        if Down_Up == 'Down':
            if In_Out == 'In':
                # Put-Down-In
                print('To be implemented')
                return 0
            elif In_Out == 'Out':
                # Put-Down-Out
                print('To be implemented')
                return 0
        elif Down_Up == 'Up':
            if In_Out == 'In':
                # Put-Up-In
                print('To be implemented')
                return 0
            elif In_Out == 'Out':
                # Put-Up-Out
                print('To be implemented')
                return 0
