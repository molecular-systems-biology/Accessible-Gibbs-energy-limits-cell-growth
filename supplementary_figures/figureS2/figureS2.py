"""
Figure S2: Accessible Gibbs energy limits metabolic activation in the PTS model.
================================================================================

This code produces panels a–d of Figure S2 from the manuscript
"Accessible Gibbs energy at metabolic activation limits long-term cell growth".

Panels a & c: Low initial ATP ([ATP]_0 = 0.1 mM), varying external substrate [S]_out
Panels b & d: Saturating external substrate ([S]_out = 100 mM), varying initial [ATP]_0

It computes:
1. Accessible Gibbs energy (initial, steady-state, and dissipated)
2. Proteome allocation (fractions of E_t, E_c, E_a relative to E_tot) and total enzyme cost per flux
"""

from scipy.optimize import brentq

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# ============================== HELPER FUNCTIONS ==============================

def accessible_Gibbs_energy_hyd (ATP, Q_0): # accessible Gibbs energy of ATP hydrolysis in kJ/l (equation S.19 in manuscript)

    ADP = Q_0 - ATP

    return RT * (Q_0 * np.log((1 + const_hyd) / Q_0) + ATP * np.log(ATP / const_hyd) + ADP * np.log(ADP)) # [kJ/l]

def proteome (x, Q_0): # E_r / J for given metabolite concentrations

    Sout, I, P, ATP = x

    ADP = Q_0 - ATP

    Et = alpha_t * (KmATPt + ATP) / ATP * (KmSt + Sout) / Sout
    Ec = alpha_c * ((KmADPc + ADP) / ADP)**2 * (KmIc + I) / I
    Ea = alpha_a * (KmATPa + ATP) / ATP * (KmPa + P) / P

    return np.array([Et, Ec, Ea])

def ATP_equation (ATP, Sout, Q_0): # equation S.10 in manuscript

    ADP = Q_0 - ATP

    N2 = np.sqrt(BIc * (1 + KmADPc / ADP * (2 + KmADPc / ADP)))
    N3 = np.sqrt(BPa * (1 + KmATPa / ATP))

    x = (Stot - Q_0) / (N2 + N3)

    I = x * N2
    P = x * N3

    gamma = (BATPt * (1 + KmSt / Sout) + BATPa * (1 + KmPa / P)) / (BADPc * (1 + KmIc / I))

    return gamma - 2 * (ATP / ADP)**2 * (1 + KmADPc / ADP)

# ============================== CONSTANTS & PARAMETERS ==============================

T = 30 # temperature [C]
T = T + 273.15 # temperature [K]
R = 8.31 / 1e3 # gas constant [kJ/(K.mol)]
RT = R * T # [J/mol]

KmSt, KmATPt, KmIc, KmADPc, KmPa, KmATPa = 50e-6, 0.1e-3, 0.5e-3, 0.5e-3, 0.5e-3, 0.1e-3 # Michaelis constants [M]
kt, kc, ka = 5*1/(5+1), 5, 0.1 # catalytic rate constants [1/s]

BSt, BATPt, BIc, BADPc, BPa, BATPa = KmSt/kt, KmATPt/kt, KmIc/kc, KmADPc/kc, KmPa/ka, KmATPa/ka # [M.s]

alpha_t, alpha_c, alpha_a = 1/kt, 1/kc, 1/ka # [s]
A = alpha_t + alpha_c + alpha_a # [s]

ADP_0, Pi = 1e-12, 10e-3 # [M]
Stot=100e-3 # [M]

const_hyd = np.exp(-12.1 + np.log(Pi))

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
# Figure 3a
####################

ax = plt.subplot(221)

Sout_vals = np.logspace(-7, -1, num=10000) # [M]
ATP_0 = 0.1e-3 # [M]

e_0, e_ss, e_diss = [], [], []

for Sout in Sout_vals: # Enzyme Cost Minimization (ECM) + accessible Gibbs energy for each Sout

    Q_0 = ADP_0 + ATP_0

    ATP = brentq (ATP_equation, 0.001*Q_0, 0.999*Q_0, args=(Sout, Q_0)) # ECM
    ADP = Q_0 - ATP

    e_0.append( accessible_Gibbs_energy_hyd (ATP_0, Q_0) ) # initial accessible Gibbs energy
    e_ss.append( accessible_Gibbs_energy_hyd (ATP, Q_0) ) # steady-state accessible Gibbs energy
    e_diss.append( e_0[-1] - e_ss[-1] ) # dissipated accessible Gibbs energy

ax.plot(1e3*Sout_vals, e_0, color="0", label=r"$\mathcal{E}_0$")
ax.plot(1e3*Sout_vals, e_ss, color="0", linestyle="dashed", label=r"$\mathcal{E}_{\mathrm{ss}}$")
ax.fill_between(1e3*Sout_vals, e_ss, e_0, color=colors[1], edgecolor="None", alpha=0.5, label=r"$\Delta\mathcal{E}_{\mathrm{hyd}}$")

ax.set_xscale("log")
plt.xlim(1e-4, 1e2)
plt.xticks([1e-4, 1e-2, 1e0, 1e2])
plt.xlabel(r"[S]$_{\mathrm{out}}$ (\si{\milli\Molar})")

plt.ylim(-0.0015, 0.0315)
plt.yticks(np.arange(0, 0.03+0.01, 0.01))
plt.ylabel(r"$\mathcal{E}$ (\si{\kilo\joule\per\liter})")

plt.legend(loc="upper left")
plt.text(1e-4 * (1e2 / 1e-4)**(0.05), 0.015, r"\SI{0.1}{\milli\Molar} [ATP]$_0$", color=colors[3], va="center", fontsize=7)
plt.title(r"\textbf{a}")

####################
# Figure 3b
####################

ax = plt.subplot(222)

Sout = 100e-3 # [M]
ATP_0_vals = np.linspace(0.1e-3, 5e-3, num=10000) # [M]

e_0, e_ss, e_diss = [], [], []

for ATP_0 in ATP_0_vals: # Enzyme Cost Minimization (ECM) + accessible Gibbs energy for each ATP_0

    Q_0 = ADP_0 + ATP_0

    ATP = brentq (ATP_equation, 0.001*Q_0, 0.999*Q_0, args=(Sout, Q_0)) # ECM
    ADP = Q_0 - ATP

    e_0.append( accessible_Gibbs_energy_hyd (ATP_0, Q_0) ) # initial accessible Gibbs energy
    e_ss.append( accessible_Gibbs_energy_hyd (ATP, Q_0) ) # steady-state accessible Gibbs energy
    e_diss.append( e_0[-1] - e_ss[-1] ) # dissipated accessible Gibbs energy

ax.plot(1e3*ATP_0_vals, e_0, color="0", label=r"$\mathcal{E}_0$")
ax.plot(1e3*ATP_0_vals, e_ss, color="0", linestyle="dashed", label=r"$\mathcal{E}_{\mathrm{ss}}$")
ax.fill_between(1e3*ATP_0_vals, e_ss, e_0, color=colors[1], edgecolor="None", alpha=0.5, label=r"$\Delta\mathcal{E}_{\mathrm{hyd}}$")

plt.xlim(0.1, 5)
plt.xticks([0.1, 1, 2, 3, 4, 5])
plt.xlabel(r"[ATP]$_0$ (\si{\milli\Molar})")

plt.ylim(-0.015, 0.315)
plt.yticks(np.arange(0, 0.3+0.1, 0.1))

plt.legend(loc="upper left")
plt.text(0.1 + (5 - 0.1) * 0.05, 0.15, r"\SI{100}{\milli\Molar} [S]$_{\mathrm{out}}$", color=colors[0], va="center", fontsize=7)
plt.title(r"\textbf{b}")

####################
# Figure 3c
####################

ax = plt.subplot(223)

Sout_vals = np.logspace(-7, -1, num=10000) # [M]
ATP_0 = 0.1e-3 # [M]

Et, Ec, Ea = [], [], []
E_tot = []

for Sout in Sout_vals: # Enzyme Cost Minimization (ECM) + proteome allocation for each Sout

    Q_0 = ADP_0 + ATP_0

    ATP = brentq (ATP_equation, 0.001*Q_0, 0.999*Q_0, args=(Sout, Q_0)) # ECM
    ADP = Q_0 - ATP

    N2 = np.sqrt(BIc * (1 + KmADPc / ADP * (2 + KmADPc / ADP))) # ECM
    N3 = np.sqrt(BPa * (1 + KmATPa / ATP)) # ECM

    x = (Stot - Q_0) / (N2 + N3) # ECM

    I = x * N2 # ECM
    P = x * N3 # ECM

    E = proteome ([Sout, I, P, ATP], Q_0) # proteome allocation
    E_tot.append(np.sum(E))

    Et.append(E[0]/E_tot[-1])
    Ec.append(E[1]/E_tot[-1])
    Ea.append(E[2]/E_tot[-1])

ax.plot(1e3*Sout_vals, Et, color=colors[0], label=r"$E_{\mathrm{t}}$")
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
plt.text(1e-4 * (1e2 / 1e-4)**(0.05), 0.9, r"\SI{0.1}{\milli\Molar} [ATP]$_0$", color=colors[3], fontsize=7)
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
ATP_0_vals = np.linspace(0.1e-3, 5e-3, num=10000) # [M]

Et, Ep, Ec, Ea = [], [], [], []
E_tot = []

for ATP_0 in ATP_0_vals: # Enzyme Cost Minimization (ECM) + proteome allocation for each ATP_0

    Q_0 = ADP_0 + ATP_0

    ATP = brentq (ATP_equation, 0.001*Q_0, 0.999*Q_0, args=(Sout, Q_0)) # ECM
    ADP = Q_0 - ATP

    N2 = np.sqrt(BIc * (1 + KmADPc / ADP * (2 + KmADPc / ADP))) # ECM
    N3 = np.sqrt(BPa * (1 + KmATPa / ATP)) # ECM

    x = (Stot - Q_0) / (N2 + N3) # ECM

    I = x * N2 # ECM
    P = x * N3 # ECM

    E = proteome ([Sout, I, P, ATP], Q_0) # proteome allocation
    E_tot.append(np.sum(E))

    Et.append(E[0]/E_tot[-1])
    Ec.append(E[1]/E_tot[-1])
    Ea.append(E[2]/E_tot[-1])

ax.plot(1e3*ATP_0_vals, Et, color=colors[0], label=r"$E_{\mathrm{t}}$")
ax.plot(1e3*ATP_0_vals, Ec, color=colors[2], label=r"$E_{\mathrm{c}}$")
ax.plot(1e3*ATP_0_vals, Ea, color=colors[3], label=r"$E_{\mathrm{a}}$")

plt.xlim(0.1, 5)
plt.xticks([0.1, 1, 2, 3, 4, 5])
plt.xlabel(r"[ATP]$_0$ (\si{\milli\Molar})")

plt.ylim(-0.05, 1.05)
plt.yticks(np.arange(0, 1+0.2, 0.2))

plt.legend()
plt.text(0.1 + (5 - 0.1) * 0.05, 0.5, r"\SI{10}{\milli\Molar} [S]$_{\mathrm{out}}$", color=colors[0], va="center", fontsize=7)
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

plt.savefig("PTS_SI.pdf")
