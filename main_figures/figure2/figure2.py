"""
Figure 2: Accessible Gibbs energy limits metabolic activation in the antiport model.
================================================================================

This code produces panels a–d of Figure 2 from the manuscript
"Accessible Gibbs energy at metabolic activation limits long-term cell growth".

Panels a & c: Low initial countersubstrate ([C]_in^0 = 10 uM), varying external substrate [S]_out
Panels b & d: Saturating external substrate ([S]_out = 100 mM), varying initial [C]_in^0

It computes:
1. Accessible Gibbs energy (initial, steady-state, and dissipated)
2. Proteome allocation (fractions of E_t, E_p, E_c, E_a relative to E_tot) and total enzyme cost per flux
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# ============================== HELPER FUNCTIONS ==============================

def accessible_Gibbs_energy (x, Q_0): # accessible Gibbs energy in kJ/l (equation 22 in manuscript).

    Sout, Cin, Sin = x

    return RT * (Q_0 * np.log((Sout + Cout) / Q_0) + Cin * np.log(Cin / Cout) + Sin * np.log(Sin / Sout)) # [kJ/l]

def proteome (x): # E_r / J for given metabolite concentrations

    Sout, Cin, Sin, I, P = x

    Et = alpha_t * ((1 + Sout / KmSt) * (1 + Cin / KmCt) + (1 + Sin / KmSt) * (1 + Cout / KmCt) - 1) / (Sout / KmSt * Cin / KmCt )
    Ep = alpha_p * (KmSp + Sin) / Sin
    Ec = alpha_c * (KmIc + I) / I
    Ea = alpha_a * (KmPa + P) / P

    return np.array([Et, Ep, Ec, Ea])

# ============================== CONSTANTS & PARAMETERS ==============================

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

# ============================== PLOTTING SETUP ==============================

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

figure=plt.figure(figsize=(3.5, 3.5), dpi=450)

####################
# Figure 2a
####################

ax = plt.subplot(221)

Sout_vals = np.logspace(-7, -1, num=10000) # [M]
Cin_0 = 10e-6 # [M]

epsilon_0, epsilon_ss, epsilon_diss = [], [], []

for Sout in Sout_vals: # Enzyme Cost Minimization (ECM) + accessible Gibbs energy for each Sout

    Q_0 = Cin_0 + Sin_0

    I = (Stot - Q_0) * np.sqrt(BIc) / (np.sqrt(BIc) + np.sqrt(BPa)) # ECM
    P = (Stot - Q_0) * np.sqrt(BPa) / (np.sqrt(BIc) + np.sqrt(BPa)) # ECM

    gamma = (1 + (KmSt / Sout) * (1 + Q_0 / KmSt) * (1 + Cout / KmCt)) * BCt / BSp # ECM
    Cin = Q_0 / (1 + np.sqrt(1/gamma)) # ECM
    Sin = Q_0 / (1 + np.sqrt(gamma)) # ECM

    e0 = accessible_Gibbs_energy([Sout, Cin_0, Sin_0], Q_0) # initial accessible Gibbs energy
    ess = accessible_Gibbs_energy([Sout, Cin, Sin], Q_0) # steady-state accessible Gibbs energy

    epsilon_0.append(e0)
    epsilon_ss.append(ess)
    epsilon_diss.append(e0 - ess) # dissipated accessible Gibbs energy

ax.plot(1e3*Sout_vals, epsilon_0, color="0", label=r"$\mathcal{E}_0$")
ax.plot(1e3*Sout_vals, epsilon_ss, color="0", linestyle="dashed", label=r"$\mathcal{E}_{\mathrm{ss}}$")
ax.fill_between(1e3*Sout_vals, epsilon_ss, epsilon_0, color="0.85", edgecolor="None", label=r"$\Delta\mathcal{E}$")

ax.set_xscale("log")
plt.xlim(1e-4, 1e2)
plt.xticks([1e-4, 1e-2, 1e0, 1e2])
plt.xlabel(r"[S]$_{\mathrm{out}}$ (\si{\milli\Molar})")

ax.set_yscale("log")
ax.tick_params(axis="y", which="minor", left=False)
plt.ylim(1e-4 * (3e-2 / 1e-4)**(-0.05), 3e-2 * (3e-2 / 1e-4)**(0.05))
plt.yticks([1e-4, 1e-3, 1e-2])
plt.ylabel(r"$\mathcal{E}$ (\si{\kilo\joule\per\liter})")

plt.legend()
plt.text(1e-4 * (1e2 / 1e-4)**(0.05), 3e-2, r"\SI{10}{\micro\Molar} [C]$^0_{\mathrm{in}}$", color=colors[3], va="top", fontsize=7)
plt.title(r"\textbf{a}")

####################
# Figure 2b
####################

ax = plt.subplot(222)

Sout = 100e-3 # [M]
Cin_0_vals = np.linspace(10e-6, 500e-6, num=10000) # [M]

epsilon_0, epsilon_ss, epsilon_diss = [], [], []

for Cin_0 in Cin_0_vals: # Enzyme Cost Minimization (ECM) + accessible Gibbs energy for each Cin_0

    Q_0 = Cin_0 + Sin_0

    I = (Stot - Q_0) * np.sqrt(BIc) / (np.sqrt(BIc) + np.sqrt(BPa)) # ECM
    P = (Stot - Q_0) * np.sqrt(BPa) / (np.sqrt(BIc) + np.sqrt(BPa)) # ECM

    gamma = (1 + (KmSt / Sout) * (1 + Q_0 / KmSt) * (1 + Cout / KmCt)) * BCt / BSp # ECM
    Cin = Q_0 / (1 + np.sqrt(1/gamma)) # ECM
    Sin = Q_0 / (1 + np.sqrt(gamma)) # ECM

    e0 = accessible_Gibbs_energy([Sout, Cin_0, Sin_0], Q_0) # initial accessible Gibbs energy
    ess = accessible_Gibbs_energy([Sout, Cin, Sin], Q_0) # steady-state accessible Gibbs energy

    epsilon_0.append(e0)
    epsilon_ss.append(ess)
    epsilon_diss.append(e0 - ess) # dissipated accessible Gibbs energy

ax.plot(1e6*Cin_0_vals, epsilon_0, color="0", label=r"$\mathcal{E}_0$")
ax.plot(1e6*Cin_0_vals, epsilon_ss, color="0", linestyle="dashed", label=r"$\mathcal{E}_{\mathrm{ss}}$")
ax.fill_between(1e6*Cin_0_vals, epsilon_ss, epsilon_0, color="0.85", edgecolor="None", label=r"$\Delta\mathcal{E}$")

plt.xlim(10, 500)
plt.xticks([10, 100, 200, 300, 400, 500])
plt.xlabel(r"[C]$^0_{\mathrm{in}}$ (\si{\micro\Molar})")

ax.set_yscale("log")
ax.tick_params(axis="y", which="minor", left=False)
plt.ylim(1e-4 * (3e-2 / 1e-4)**(-0.05), 3e-2 * (3e-2 / 1e-4)**(0.05))
plt.yticks([1e-4, 1e-3, 1e-2])

plt.legend()
plt.text(10 + (500 - 10) * 0.025, 3e-2, r"\SI{100}{\milli\Molar} [S]$_{\mathrm{out}}$", color=colors[0], va="top", fontsize=7)
plt.title(r"\textbf{b}")

####################
# Figure 2c
####################

ax = plt.subplot(223)

Sout_vals = np.logspace(-7, -1, num=10000) # [M]
Cin_0 = 10e-6 # [M]

Et, Ep, Ec, Ea = [], [], [], []
E_tot = []

for Sout in Sout_vals: # Enzyme Cost Minimization (ECM) + proteome allocation for each Sout

    Q_0 = Cin_0 + Sin_0

    I = (Stot - Q_0) * np.sqrt(BIc) / (np.sqrt(BIc) + np.sqrt(BPa)) # ECM
    P = (Stot - Q_0) * np.sqrt(BPa) / (np.sqrt(BIc) + np.sqrt(BPa)) # ECM

    gamma = (1 + (KmSt / Sout) * (1 + Q_0 / KmSt) * (1 + Cout / KmCt)) * BCt / BSp # ECM
    Cin = Q_0 / (1 + np.sqrt(1/gamma)) # ECM
    Sin = Q_0 / (1 + np.sqrt(gamma)) # ECM

    E = proteome ([Sout, Cin, Sin, I, P]) # proteome allocation
    E_tot.append(np.sum(E))

    Et.append(E[0]/E_tot[-1])
    Ep.append(E[1]/E_tot[-1])
    Ec.append(E[2]/E_tot[-1])
    Ea.append(E[3]/E_tot[-1])

ax.plot(1e3*Sout_vals, Et, color=colors[0], label=r"$E_{\mathrm{t}}$")
ax.plot(1e3*Sout_vals, Ep, color=colors[1], label=r"$E_{\mathrm{p}}$")
ax.plot(1e3*Sout_vals, Ec, color=colors[2], label=r"$E_{\mathrm{c}}$")
ax.plot(1e3*Sout_vals, Ea, color=colors[3], label=r"$E_{\mathrm{a}}$")

ax.set_xscale("log")
plt.xlim(1e-4, 1e2)
plt.xticks([1e-4, 1e-2, 1e0, 1e2])
plt.xlabel(r"[S]$_{\mathrm{out}}$ (\si{\milli\Molar})")

plt.ylim(-0.05, 1.05)
plt.yticks(np.arange(0, 1+0.2, 0.2))
plt.ylabel(r"$E_r / E_{\mathrm{tot}}$")

plt.legend()
plt.text(1e-4 * (1e2 / 1e-4)**(0.05), 1, r"\SI{10}{\micro\Molar} [C]$^0_{\mathrm{in}}$", color=colors[3], va="top", fontsize=7)
plt.title(r"\textbf{c}")

left, right = ax.get_xlim()
mid_left = left * (right / left)**(0.25)
mid = left * (right / left)**(0.5)
mid_right = left * (right / left)**(0.75)

left_val = f"{E_tot[0]:.0f}"
mid_left_val = f"{E_tot[int(0.25*len(Sout_vals))]:.0f}"
mid_val = f"{E_tot[int(0.5*len(Sout_vals))]:.0f}"
mid_right_val = f"{E_tot[int(0.75*len(Sout_vals))]:.0f}"
right_val = f"{E_tot[-1]:.0f}"

ax_top = ax.secondary_xaxis("top")
ax_top.set_xticks([], minor=False)
ax_top.set_xticks([left, mid_left, mid, mid_right, right], minor=True)
ax_top.set_xticklabels([left_val, mid_left_val, mid_val, mid_right_val, right_val], minor=True)
ax_top.tick_params(which="minor", direction="in", size=2, width=0.5)
ax_top.set_xlabel(r"$E_{\mathrm{tot}} / J$ (\si{\second})")

####################
# Figure 2d
####################

ax = plt.subplot(224)

Sout = 100e-3 # [M]
Cin_0_vals = np.linspace(10e-6, 500e-6, num=10000) # [M]

Et, Ep, Ec, Ea = [], [], [], []
E_tot = []

for Cin_0 in Cin_0_vals: # Enzyme Cost Minimization (ECM) + proteome allocation for each Cin_0

    Q_0 = Cin_0 + Sin_0

    I = (Stot - Q_0) * np.sqrt(BIc) / (np.sqrt(BIc) + np.sqrt(BPa)) # ECM
    P = (Stot - Q_0) * np.sqrt(BPa) / (np.sqrt(BIc) + np.sqrt(BPa)) # ECM

    gamma = (1 + (KmSt / Sout) * (1 + Q_0 / KmSt) * (1 + Cout / KmCt)) * BCt / BSp # ECM
    Cin = Q_0 / (1 + np.sqrt(1/gamma)) # ECM
    Sin = Q_0 / (1 + np.sqrt(gamma)) # ECM

    E = proteome ([Sout, Cin, Sin, I, P]) # proteome allocation
    E_tot.append(np.sum(E))

    Et.append(E[0]/E_tot[-1])
    Ep.append(E[1]/E_tot[-1])
    Ec.append(E[2]/E_tot[-1])
    Ea.append(E[3]/E_tot[-1])

ax.plot(1e6*Cin_0_vals, Et, color=colors[0], label=r"$E_{\mathrm{t}}$")
ax.plot(1e6*Cin_0_vals, Ep, color=colors[1], label=r"$E_{\mathrm{p}}$")
ax.plot(1e6*Cin_0_vals, Ec, color=colors[2], label=r"$E_{\mathrm{c}}$")
ax.plot(1e6*Cin_0_vals, Ea, color=colors[3], label=r"$E_{\mathrm{a}}$")

plt.xlim(10, 500)
plt.xticks([10, 100, 200, 300, 400, 500])
plt.xlabel(r"[C]$^0_{\mathrm{in}}$ (\si{\micro\Molar})")

plt.ylim(-0.05, 1.05)
plt.yticks(np.arange(0, 1+0.2, 0.2))

plt.legend()
plt.text(10 + (500 - 10) * 0.025, 1, r"\SI{100}{\milli\Molar} [S]$_{\mathrm{out}}$", color=colors[0], va="top", fontsize=7)
plt.title(r"\textbf{d}")

left, right = ax.get_xlim()
mid_left = 0.25 * (left + right)
mid = 0.5 * (left + right)
mid_right = 0.75 * (left + right)

left_val = f"{E_tot[0]:.0f}"
mid_left_val = f"{E_tot[int(0.25*len(Sout_vals))]:.0f}"
mid_val = f"{E_tot[int(0.5*len(Sout_vals))]:.0f}"
mid_right_val = f"{E_tot[int(0.75*len(Sout_vals))]:.0f}"
right_val = f"{E_tot[-1]:.0f}"

ax_top = ax.secondary_xaxis("top")
ax_top.set_xticks([], minor=False)
ax_top.set_xticks([left, mid_left, mid, mid_right, right], minor=True)
ax_top.set_xticklabels([left_val, mid_left_val, mid_val, mid_right_val, right_val], minor=True)
ax_top.tick_params(which="minor", direction="in", size=2, width=0.5)
ax_top.set_xlabel(r"$E_{\mathrm{tot}} / J$ (\si{\second})")

plt.savefig("Figure2.pdf")
