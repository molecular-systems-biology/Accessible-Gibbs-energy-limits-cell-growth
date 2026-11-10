"""
Figure S1: The growth limitation is robust to kinetic parameter variation.
================================================================================

This code produces Figure S1 from the manuscript
"Accessible Gibbs energy at metabolic activation can constrain steady-state growth".
"""

from matplotlib.lines import Line2D

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def mu_curve (Cin_0_vals, kt, kp, kc, ka, KmSt, KmCt, KmSp, KmIc, KmPa):

    BSt, BCt, BSp, BIc, BPa = KmSt/kt, KmCt/kt, KmSp/kp, KmIc/kc, KmPa/ka # [M.s]
    A = 1/kt + 1/kp + 1/kc + 1/ka # [s]

    Q_0 = Cin_0_vals + Sin_0 # [M]

    I = (Stot - Q_0) * np.sqrt(BIc) / (np.sqrt(BIc) + np.sqrt(BPa))
    P = (Stot - Q_0) * np.sqrt(BPa) / (np.sqrt(BIc) + np.sqrt(BPa))

    gamma = (1 + (KmSt / Sout) * (1 + Q_0 / KmSt) * (1 + Cout / KmCt)) * BCt / BSp
    Cin = Q_0 / (1 + np.sqrt(1/gamma))
    Sin = Q_0 / (1 + np.sqrt(gamma))

    return A / (A + BPa/P + BIc/I + BSp/Sin + BSt/Sout + BCt/Cin + (BSt * BCt * kt / (Sout * Cin)) * (1 + Sin/KmSt) * (1 + Cout/KmCt)) # normalised growth rate

KmSt, KmCt, KmSp, KmIc, KmPa = 50e-6, 0.1e-3, 0.1e-3, 0.5e-3, 0.5e-3 # [M]
kt, kp, kc, ka = 5, 1, 5, 0.1 # [1/s]

Cout, Sin_0 = 1e-12, 1e-12 # [M]
Stot = 100e-3 # [M]
Sout = 1e-1 # [M]

N = 100 # number of random parameter sets
seed = 0 # RNG seed

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

Cin_0_vals = np.logspace(-7, np.log10(500e-6), num=10000) # [M]

#--------------------------------------------------
# random kinetic parameter sets
#--------------------------------------------------
# k_cat log-uniform in [0.1, 100] 1/s; K_m log-uniform in [0.01, 10] mM

rng = np.random.default_rng(seed)

random_mu = [] # plotted random curves (Source Data)

for n in range(N):

    kt, kp, kc, ka = 10**rng.uniform(-1, 2, size=4) # [1/s]
    KmSt, KmCt, KmSp, KmIc, KmPa = 10**rng.uniform(-5, -2, size=5) # [M]

    mu = mu_curve (Cin_0_vals, kt, kp, kc, ka, KmSt, KmCt, KmSp, KmIc, KmPa)
    random_mu.append(mu)

    ax.plot(1e6*Cin_0_vals, mu, color=colors[0], alpha=0.15, lw=0.5)
random_handle = Line2D([0], [0], color=colors[0], lw=1, label="Random")

#--------------------------------------------------
# default kinetic parameter set
#--------------------------------------------------

KmSt, KmCt, KmSp, KmIc, KmPa = 50e-6, 0.1e-3, 0.1e-3, 0.5e-3, 0.5e-3 # [M]
kt, kp, kc, ka = 5, 1, 5, 0.1 # [1/s]

mu_default = mu_curve (Cin_0_vals, kt, kp, kc, ka, KmSt, KmCt, KmSp, KmIc, KmPa)

ax.plot(1e6*Cin_0_vals, mu_default, color="k")
default_handle = Line2D([0], [0], color="k", lw=1, label="Default")

ax.set_xlim(-25, 500)
ax.set_xticks(np.arange(0, 500+100, 100))
ax.set_xlabel(r"[C]$^0_{\mathrm{in}}$ (\si{\micro\Molar})")
ax.set_ylim(-0.05, 1.15)
ax.set_yticks(np.arange(0, 1+0.2, 0.2))
ax.set_ylabel(r"$\mu / \mu_{\mathrm{max}}$")
ax.legend(handles=[default_handle, random_handle], loc="upper center", ncols=2)

plt.savefig("Param_robustness_SI.pdf")
