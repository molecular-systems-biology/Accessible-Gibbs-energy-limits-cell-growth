"""
Figure 1g: PTS-coupled model.
================================================================================

This code produces the data for panel g of Figure 1 from the manuscript
"Accessible Gibbs energy at metabolic activation can constrain steady-state growth".
"""

from scipy.optimize import brentq

import numpy as np

def ATP_equation (ATP, Sout, Q_0):

    ADP = Q_0 - ATP

    N2 = np.sqrt(BIc * (1 + KmADPc / ADP * (2 + KmADPc / ADP)))
    N3 = np.sqrt(BPa * (1 + KmATPa / ATP))

    x = (Stot - Q_0) / (N2 + N3)

    I = x * N2
    P = x * N3

    gamma = (BATPt * (1 + KmSt / Sout) + BATPa * (1 + KmPa / P)) / (BADPc * (1 + KmIc / I))

    return gamma - 2 * (ATP / ADP)**2 * (1 + KmADPc / ADP)

KmSt, KmATPt, KmIc, KmADPc, KmPa, KmATPa = 50e-6, 0.1e-3, 0.5e-3, 0.5e-3, 0.5e-3, 0.1e-3 # [M]
kt, kc, ka = 5*1/(5+1), 5, 0.1 # [1/s]

BSt, BATPt, BIc, BADPc, BPa, BATPa = KmSt/kt, KmATPt/kt, KmIc/kc, KmADPc/kc, KmPa/ka, KmATPa/ka # [M.s]
alpha_t, alpha_c, alpha_a = 1/kt, 1/kc, 1/ka # [s]

A = alpha_t + alpha_c + alpha_a # [s]

ADP_0, Pi = 1e-12, 10e-3 # [M]
Stot=100e-3 # [M]

const_hyd = np.exp(-12.1 + np.log(Pi))

Sout_vals = np.array([1e-6, 1e-5, 1e-4, 1e-1]) # [M]
ATP_0_vals = np.logspace(-5, np.log10(5e-3), num=10000)

for i, Sout in enumerate(Sout_vals):

    mu = [] # normalised growth rate

    for ATP_0 in ATP_0_vals:

        Q_0 = ADP_0 + ATP_0

        ATP = brentq (ATP_equation, 0.001*Q_0, 0.999*Q_0, args=(Sout, Q_0))
        ADP = Q_0 - ATP

        N2 = np.sqrt(BIc * (1 + KmADPc / ADP * (2 + KmADPc / ADP)))
        N3 = np.sqrt(BPa * (1 + KmATPa / ATP))

        x = (Stot - Q_0) / (N2 + N3)

        I = x * N2
        P = x * N3

        mu.append(A / (A + BSt/Sout + BATPt/ATP + BSt*BATPt*kt/(Sout*ATP) + BIc/I + (1+KmIc/I)*(BADPc/ADP)*(2+KmADPc/ADP) + BATPa/ATP + BPa/P + BPa*BATPa*ka/(P*ATP))) # normalised growth rate

    np.savez(f"g{i}.npz", x=1e3*ATP_0_vals, y=mu)
