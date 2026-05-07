"""
Figure S3: Growth limitation under fixed proteome allocation.
================================================================================

This code produces Figure S3 from the manuscript
"Accessible Gibbs energy at metabolic activation limits long-term cell growth".

It compares two scenarios in the antiport-coupled model at saturating 
external substrate ([S]_out = 100 mM):

1. Optimized proteome: enzyme levels and metabolite pools are jointly 
   optimized for each initial countersubstrate concentration [C]_in^0.
2. Fixed proteome: enzyme allocation is determined once at high accessible 
   Gibbs energy ([C]_in^0 = 500 µM) and then held constant. Only metabolite 
   reorganization is allowed.
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def proteome (x):

    Sout, Cin, Sin, I, P = x

    Et = alpha_t * ((1 + Sout / KmSt) * (1 + Cin / KmCt) + (1 + Sin / KmSt) * (1 + Cout / KmCt) - 1) / (Sout / KmSt * Cin / KmCt )
    Ep = alpha_p * (KmSp + Sin) / Sin
    Ec = alpha_c * (KmIc + I) / I
    Ea = alpha_a * (KmPa + P) / P

    return np.array([Et, Ep, Ec, Ea])

#----------------------------------------
# parameters
#----------------------------------------

T = 30 # temperature [C]
T = T + 273.15 # temperature [K]
R = 8.31 / 1e3 # gas constant [kJ/(K.mol)]
RT = R * T # [J/mol]

KmSt, KmCt, KmSp, KmIc, KmPa = 50e-6, 0.1e-3, 0.1e-3, 0.5e-3, 0.5e-3 # Michaelis constants [M]
kt, kp, kc, ka = 5, 1, 5, 0.1 # catalytic rate constants [1/s]

BSt, BCt, BSp, BIc, BPa = KmSt/kt, KmCt/kt, KmSp/kp, KmIc/kc, KmPa/ka # [M.s]

alpha_t, alpha_p, alpha_c, alpha_a = 1/kt, 1/kp, 1/kc, 1/ka # [s]
A = alpha_t + alpha_p + alpha_c + alpha_a # [s]

Cout, Sin_0 = 1e-12, 1e-12 # [M]
Stot = 100e-3 # [M]

#----------------------------------------
# plotting
#----------------------------------------

plt.rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"]}) # Helvetica
plt.rc("text", usetex=True) # LaTeX
plt.rc("text.latex", preamble=r"\usepackage{sfmath, siunitx} \DeclareSIUnit\Molar{\textsc{m}} \renewcommand{\familydefault}{\sfdefault}") # LaTeX preamble
plt.rc("pdf", fonttype=42) # type 42 font

plt.rc("axes", labelsize=7, titlelocation="left", titlesize=8)
plt.rc("xtick", direction="in", labelsize=7)
plt.rc("xtick.major", size=2, width=0.5)
plt.rc("ytick", direction="in", labelsize=7)
plt.rc("ytick.major", size=2, width=0.5)

plt.rc("figure.constrained_layout", use=True)
plt.rc("legend", borderpad=0.4, columnspacing=1, fontsize=5, handletextpad=0.4, labelspacing=0.25, title_fontsize=5)
plt.rc("lines", linewidth=1, markeredgewidth=0, markersize=8)

colors=sns.color_palette("deep") # color palette

figure=plt.figure(figsize=(1.75, 1.75), dpi=450)

ax = plt.subplot()

#--------------------------------------------------
# high Q_0 reference (saturating Sout)
#--------------------------------------------------

Sout_high = 100e-3 # [M]

Q_0_high = 500e-6 + Sin_0 # [M]

I = (Stot - Q_0_high) * np.sqrt(BIc) / (np.sqrt(BIc) + np.sqrt(BPa))
P = (Stot - Q_0_high) * np.sqrt(BPa) / (np.sqrt(BIc) + np.sqrt(BPa))

gamma = (1 + (KmSt / Sout_high) * (1 + Q_0_high / KmSt) * (1 + Cout / KmCt)) * BCt / BSp
Cin = Q_0_high / (1 + np.sqrt(1/gamma))
Sin = Q_0_high / (1 + np.sqrt(gamma))

E_fixed = proteome ([Sout_high, Cin, Sin, I, P])
E_tot_fixed = np.sum(E_fixed)

Cin_0_vals = np.logspace(-7, np.log10(500e-6), num=10000) # [M]

mu_opt = []
mu_fix = []

for Cin_0 in Cin_0_vals:

# optimized case

    Q_0 = Cin_0 + Sin_0

    I = (Stot - Q_0) * np.sqrt(BIc) / (np.sqrt(BIc) + np.sqrt(BPa))
    P = (Stot - Q_0) * np.sqrt(BPa) / (np.sqrt(BIc) + np.sqrt(BPa))

    gamma = (1 + (KmSt / Sout_high) * (1 + Q_0 / KmSt) * (1 + Cout / KmCt)) * BCt / BSp
    Cin = Q_0 / (1 + np.sqrt(1/gamma))
    Sin = Q_0 / (1 + np.sqrt(gamma))

    E_opt = proteome ([Sout_high, Cin, Sin, I, P])
    E_tot_opt = np.sum(E_opt)

    mu_opt.append(A / E_tot_opt)

# fixed proteome case

    j_list = []
    Cin_vals = np.linspace(1e-12, Q_0 - 1e-12, num=10000)

    for Cin in Cin_vals:

        Sin = Q_0 - Cin

        j_t = E_fixed[0] * kt * ((Sout_high / KmSt) * (Cin / KmCt)) / ((1 + Sout_high / KmSt) * (1 + Cin / KmCt) + (1 + Sin / KmSt) * (1 + Cout / KmCt) - 1) # j_t = J_t / J
        j_p = E_fixed[1] * kp * (Sin / (KmSp + Sin)) # j_p = J_p / J
        j_c = E_fixed[2] * kc # j_c = J_c / J
        j_a = E_fixed[3] * ka # j_a = J_a / J

        j = min (j_t, j_p, j_c, j_a)
        j_list.append(j)

    j_max = max(j_list)

    mu_fix.append(A * j_max / E_tot_fixed)

ax.plot(1e6*Cin_0_vals, mu_opt, color="k", ls="solid", label=r"Optimized proteome")
ax.plot(1e6*Cin_0_vals, mu_fix, color="k", ls="dashed", label=r"Fixed proteome")

ax.set_xlim(-25, 500)
plt.xticks(np.arange(0, 500+100, 100))
plt.xlabel(r"[C]$^0_{\mathrm{in}}$ (\si{\micro\Molar})")
ax.set_ylim(-0.05, 1)
ax.set_yticks(np.arange(0, 1+0.2, 0.2))
ax.set_ylabel(r"$\mu / \mu_{\mathrm{max}}$")
plt.legend()

plt.savefig("Metabolite_memory_SI.pdf")
