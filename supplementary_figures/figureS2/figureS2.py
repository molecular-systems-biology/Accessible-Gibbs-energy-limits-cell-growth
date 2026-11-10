"""
Figure S2: The growth limitation is robust to reversible kinetics and external countersubstrate concentration.
================================================================================

This code produces panels a–c of Figure S2 from the manuscript
"Accessible Gibbs energy at metabolic activation can constrain steady-state growth".
"""

from scipy.optimize import differential_evolution

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def penalized_objective (x, S_out, C_out, Q_0, w=1e12):

    S_in, I, P = 10**x
    C_in = Q_0 - S_in

    Gamma_t, Gamma_p, Gamma_c = mass_action (x, S_out, C_out, Q_0)

    ratio = np.array([Gamma_t / K_eq_t, Gamma_p / K_eq_p, Gamma_c / K_eq_c])
    if np.any(ratio > 1):
        return w * (1 + np.sum(np.maximum(ratio - 1, 0)**2))

    E = proteome (x, S_out, C_out, Q_0)
    E_tot = np.sum(E)

    penalty = 0

    if S_in < 0: penalty += w * (S_in / Q_0)**2
    if C_in < 0: penalty += w * (C_in / Q_0)**2
    if I < 0: penalty += w * (I / 100e-3)**2
    if P < 0: penalty += w * (P / 100e-3)**2

    if S_in > Q_0: penalty += w * (1 - S_in / Q_0)**2

    for enz in E:
        if enz < 0: penalty += w * (enz / E_tot)**2

    return E_tot + penalty

def proteome (x, S_out, C_out, Q_0):

    S_in, I, P = 10**x
    C_in = Q_0 - S_in

    Gamma_t, Gamma_p, Gamma_c = mass_action (x, S_out, C_out, Q_0)

    E_t = alpha_t * ((1 + S_out / K_mS_t) * (1 + C_in / K_mC_t) + (1 + S_in / K_mS_t) * (1 + C_out / K_mC_t) - 1)\
        /((S_out / K_mS_t) * (C_in / K_mC_t) * (1 - Gamma_t / K_eq_t))
    E_p = alpha_p * ((1 + S_in / K_mS_p) * (1 + Pi / K_mPi_p) + (1 + I / K_mI_p) - 1)\
        /((S_in / K_mS_p) * (Pi / K_mPi_p) * (1 - Gamma_p / K_eq_p))
    E_c = alpha_c * ((1 + I / K_mI_c) * (1 + ADP / K_mADP_c) + (1 + P / K_mP_c) * (1 + ATP / K_mATP_c) - 1)\
        /((I / K_mI_c) * (ADP / K_mADP_c) * (1 - Gamma_c / K_eq_c))
    E_a = alpha_a * (1 + K_mP_a / P)

    return np.array([E_t, E_p, E_c, E_a])

def mass_action (x, S_out, C_out, Q_0):

    S_in, I, P = 10**x
    C_in = Q_0 - S_in

    Gamma_t = S_in * C_out / (S_out * C_in)
    Gamma_p = I / (S_in * Pi)
    Gamma_c = P * ATP / (I * ADP)

    return np.array([Gamma_t, Gamma_p, Gamma_c])

K_mS_t, K_mC_t, K_mS_p, K_mI_c, K_mP_a = 50e-6, 0.1e-3, 0.1e-3, 0.5e-3, 0.5e-3 # [M]
k_t, k_p, k_c, k_a = 5, 1, 5, 0.1 # [1/s]

alpha_t, alpha_p, alpha_c, alpha_a = 1/k_t, 1/k_p, 1/k_c, 1/k_a # [s]

A = alpha_t + alpha_p + alpha_c + alpha_a # [s]

K_mPi_p, K_mI_p, K_mADP_c, K_mATP_c, K_mP_c = 0.5e-3, 0.5e-3, 0.5e-3, 0.5e-3, 0.5e-3 # [M]
K_eq_t, K_eq_p, K_eq_c = 1, 30, 100

ATP, ADP, Pi = 5e-3, 1e-3, 50e-3 # [M]

S_in_0 = 1e-12 # [M]

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

lss=["dotted", "dashdot", "dashed", "solid"]
alphas=[0.4, 0.6, 0.8, 1]
labels=[r"\SI{1}{\micro\Molar}", r"\SI{10}{\micro\Molar}", r"\SI{0.1}{\milli\Molar}", r"\SI{100}{\milli\Molar}"]

figure=plt.figure(figsize=(5.25, 1.75), dpi=450)

#----------------------------------------
# Panel a
#----------------------------------------

C_out = 1e-12 # [M]

ax = plt.subplot(131)

S_out_vals = np.array([1e-6, 1e-5, 1e-4, 1e-1]) # [M]
C_in_0_vals = np.logspace(-7, np.log10(500e-6), num=1000) # [M]

C_in_0_a = C_in_0_vals # [M] (Source Data)
mu_a = [] # normalised growth rate for each [S]out (Source Data)

for i, S_out in enumerate(S_out_vals):

    mu = [] # normalised growth rate

    for C_in_0 in C_in_0_vals:

        Q_0 = C_in_0 + S_in_0 # [M]

        bounds = [(-12, np.log10(Q_0 - 1e-12)), (-12, -1), (-12, -1)]
        result = differential_evolution (penalized_objective, bounds=bounds, args=(S_out, C_out, Q_0), tol=1e-14, seed=0)

        E = proteome (result.x, S_out, C_out, Q_0)
        mu.append(A / np.sum(E))

    mu_a.append(mu)

    ax.plot(1e6*C_in_0_vals, mu, color=colors[0], ls=lss[i], alpha=alphas[i], label=labels[i])

ax.set_xlim(0, 500)
ax.set_xticks(np.arange(0, 500+100, 100))
ax.set_xlabel(r"[C]$^0_{\mathrm{in}}$ (\si{\micro\Molar})")
ax.set_ylim(-0.05, 1)
ax.set_yticks(np.arange(0, 1+0.2, 0.2))
ax.set_ylabel(r"$\mu / \mu_{\mathrm{max}}$")
ax.set_title(r"\textbf{a}")
handles, labels = ax.get_legend_handles_labels()
handles, labels = [handles[0], handles[2], handles[1], handles[3]], [labels[0], labels[2], labels[1], labels[3]]
ax.legend(handles, labels, alignment="left", loc="upper center", title=r"[S]$_\mathrm{out}$", ncols=2)

#----------------------------------------
# Panel b
#----------------------------------------

ax = plt.subplot(132)

labels = ["10", "100", "500"]

C_in_0_vals = np.array([10e-6, 100e-6, 500e-6]) # [M]
C_out_vals = np.logspace(-13, -1, num=1000) # [M]

mu_b = [] # normalised growth rate for each [C]in0 (Source Data)

for i, C_in_0 in enumerate(C_in_0_vals):

    mu = [] # normalised growth rate

    for C_out in C_out_vals:

        Q_0 = C_in_0 + S_in_0 # [M]

        bounds = [(-12, np.log10(Q_0 - 1e-12)), (-12, -1), (-12, -1)]
        result = differential_evolution (penalized_objective, bounds=bounds, args=(S_out, C_out, Q_0), tol=1e-14, seed=0)

        E = proteome (result.x, S_out, C_out, Q_0)
        mu.append(A / np.sum(E))

    mu_b.append(mu)

    ax.plot(1e3*C_out_vals, mu, color=colors[2], ls=lss[i+1], alpha=alphas[i], label=labels[i])

ax.set_xscale("log")
ax.set_xlim(1e-10, 1e2)
ax.set_xticks([1e-10, 1e-6, 1e-2, 1e2])
ax.set_xlabel(r"[C]$_{\mathrm{out}}$ (\si{\milli\Molar})")
ax.set_ylim(-0.05, 1)
ax.set_yticks(np.arange(0, 1+0.2, 0.2))
ax.set_title(r"\textbf{b}")
ax.legend(alignment="left", loc="upper center", title=r"[C]$^0_{\mathrm{in}}$ (\si{\micro\Molar})", ncols=3)

#----------------------------------------
# Panel c
#----------------------------------------

ax = plt.subplot(133)

C_in_0 = 500e-6 # [M]

ratio_t, ratio_p, ratio_c = [], [], []

for C_out in C_out_vals:

    Q_0 = C_in_0 + S_in_0 # [M]

    bounds = [(-12, np.log10(Q_0 - 1e-12)), (-12, -1), (-12, -1)]
    result = differential_evolution (penalized_objective, bounds=bounds, args=(S_out, C_out, Q_0), tol=1e-14, seed=0)

    Gamma_t, Gamma_p, Gamma_c = mass_action (result.x, S_out, C_out, Q_0)

    ratio_t.append(Gamma_t / K_eq_t)
    ratio_p.append(Gamma_p / K_eq_p)
    ratio_c.append(Gamma_c / K_eq_c)

plt.plot(1e3*C_out_vals, ratio_p, color=colors[1], label="Phosphorylation")
plt.plot(1e3*C_out_vals, ratio_c, color=colors[2], label="Catabolism")
plt.plot(1e3*C_out_vals, ratio_t, color=colors[0], label="Transport")

ax.set_xscale("log")
ax.set_xlim(1e-10, 1e2)
ax.set_xticks([1e-10, 1e-6, 1e-2, 1e2])
ax.set_xlabel(r"[C]$_{\mathrm{out}}$ (\si{\milli\Molar})")
ax.set_ylim(-0.05, 1)
ax.set_yticks(np.arange(0, 1+0.2, 0.2))
ax.set_ylabel(r"$\Gamma_r / K^\mathrm{eq}_r$")
ax.set_title(r"\textbf{c}")
ax.legend(loc="upper center")

plt.savefig("Reversible_antiport_coupled_model.pdf")
