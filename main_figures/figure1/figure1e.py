"""
Figure 1e: antiport-coupled model.
================================================================================

This code produces the data for panel e of Figure 1 from the manuscript
"Accessible Gibbs energy at metabolic activation can constrain steady-state growth".
"""

import numpy as np

KmSt, KmCt, KmSp, KmIc, KmPa = 50e-6, 0.1e-3, 0.1e-3, 0.5e-3, 0.5e-3 # [M]
kt, kp, kc, ka = 5, 1, 5, 0.1 # [1/s]

BSt, BCt, BSp, BIc, BPa = KmSt/kt, KmCt/kt, KmSp/kp, KmIc/kc, KmPa/ka # [M.s]
alpha_t, alpha_p, alpha_c, alpha_a = 1/kt, 1/kp, 1/kc, 1/ka # [s]

A = alpha_t + alpha_p + alpha_c + alpha_a # [s]

Cout, Sin_0 = 1e-12, 1e-12 # [M]
Stot=100e-3 # [M]

Sout_vals = np.array([1e-6, 1e-5, 1e-4, 1e-1]) # [M]
Cin_0_vals = np.logspace(-7, np.log10(500e-6), num=10000) # [M]

for i, Sout in enumerate(Sout_vals):

    mu = [] # normalised growth rate

    for Cin_0 in Cin_0_vals:

        Q_0 = Cin_0 + Sin_0 # [M]

        I, P = (Stot - Q_0) * np.sqrt(BIc) / (np.sqrt(BIc) + np.sqrt(BPa)), (Stot - Q_0) * np.sqrt(BPa) / (np.sqrt(BIc) + np.sqrt(BPa))

        gamma = (1 + (KmSt / Sout) * (1 + Q_0 / KmSt) * (1 + Cout / KmCt)) * BCt / BSp
        Cin, Sin = Q_0 / (1 + np.sqrt(1/gamma)), Q_0 / (1 + np.sqrt(gamma)) # [M]

        mu.append(A / (A + BPa/P + BIc/I + BSp/Sin + BSt/Sout + BCt/Cin + (BSt * BCt * kt / (Sout * Cin)) * (1 + Sin/KmSt) * (1 + Cout/KmCt))) # normalised growth rate

    np.savez(f"e{i}.npz", x=1e6*Cin_0_vals, y=mu)
