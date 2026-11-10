"""
Figure 1f: uniport-ATP-coupled model.
================================================================================

This code produces the data for panel f of Figure 1 from the manuscript
"Accessible Gibbs energy at metabolic activation can constrain steady-state growth".
"""

from scipy.optimize import brentq

import numpy as np

def Sin_from_x (x, Sout, N1):

    def TE (Sin): # transcendental equation
        return x**(-2) - (N1 / Sin)**2 + (2 + (KmSt + 2 * Sin) / (Sout - Sin)) / (kt * (Sout - Sin))

    Sin_min = 1e-15 # [M]
    Sin_max = 0.999 * Sout # [M]

    return brentq (TE, Sin_min, Sin_max)

def x_from_ATP (Sout, ATP, Q_0):

    ADP = Q_0 - ATP

    N1 = np.sqrt(BSp * (1 + KmATPp / ATP))
    N2 = np.sqrt(BIc * (1 + KmADPc / ADP * (2 + KmADPc / ADP)))
    N3 = np.sqrt(BPa * (1 + KmATPa / ATP))

    def total_pool (x):

        Sin = Sin_from_x (x, Sout, N1)
        I = x * N2
        P = x * N3

        return Stot - Q_0 - Sin - I - P

    x_min = (Stot - Q_0) / (N1 + N2 + N3)
    x_max = (Stot - Q_0) / (N2 + N3)

    return brentq (total_pool, x_min, x_max)

def ATP_equation (ATP, Sout, Q_0):

    ADP = Q_0 - ATP

    x = x_from_ATP (Sout, ATP, Q_0)

    N1 = np.sqrt(BSp * (1 + KmATPp / ATP))
    N2 = np.sqrt(BIc * (1 + KmADPc / ADP * (2 + KmADPc / ADP)))
    N3 = np.sqrt(BPa * (1 + KmATPa / ATP))

    Sin = Sin_from_x (x, Sout, N1)
    I = x * N2
    P = x * N3

    gamma = (BATPp * (1 + KmSp / Sin) + BATPa * (1 + KmPa / P)) / (BADPc * (1 + KmIc / I))

    return gamma - 2 * (ATP / ADP)**2 * (1 + KmADPc / ADP)

KmSt, KmSp, KmATPp, KmIc, KmADPc, KmPa, KmATPa = 50e-6, 0.1e-3, 0.1e-3, 0.5e-3, 0.5e-3, 0.5e-3, 0.1e-3 # [M]
kt, kp, kc, ka = 5, 1, 5, 0.1 # [1/s]

BSt, BSp, BATPp, BIc, BADPc, BPa, BATPa = KmSt/kt, KmSp/kp, KmATPp/kp, KmIc/kc, KmADPc/kc, KmPa/ka, KmATPa/ka # [M.s]
alpha_t, alpha_p, alpha_c, alpha_a = 1/kt, 1/kp, 1/kc, 1/ka # [s]

A = alpha_t + alpha_p + alpha_c + alpha_a # [s]

ADP_0, Sin_0, Pi = 1e-12, 1e-12, 10e-3 # [M]
Stot = 100e-3 # [M]

const_hyd = np.exp(-12.1 + np.log(Pi))

Sout_vals = np.array([1e-6, 1e-5, 1e-4, 1e-1]) # [M]
ATP_0_vals = np.logspace(-5, np.log10(5e-3), num=10000)

for i, Sout in enumerate(Sout_vals):

    mu = [] # normalised growth rate

    for ATP_0 in ATP_0_vals:

        Q_0 = ADP_0 + ATP_0

        ATP = brentq (ATP_equation, 0.001*Q_0, 0.999*Q_0, args=(Sout, Q_0))
        ADP = Q_0 - ATP

        x = x_from_ATP (Sout, ATP, Q_0)

        N1 = np.sqrt(BSp * (1 + KmATPp / ATP))
        N2 = np.sqrt(BIc * (1 + KmADPc / ADP * (2 + KmADPc / ADP)))
        N3 = np.sqrt(BPa * (1 + KmATPa / ATP))

        Sin = Sin_from_x (x, Sout, N1)
        I = x * N2
        P = x * N3

        mu.append(A / (A + (KmSt+2*Sin)/(kt*(Sout-Sin)) + BSp*BATPp*kp/(Sin*ATP) + BSp/Sin + BATPp/ATP + BIc/I + (1+KmIc/I)*(BADPc/ADP)*(2+KmADPc/ADP) + BATPa/ATP + BPa/P + BPa*BATPa*ka/(P*ATP))) # normalised growth rate

    np.savez(f"f{i}.npz", x=1e3*ATP_0_vals, y=mu)
