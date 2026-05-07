"""
Figure 5: Dynamic intravesicular metabolite reorganization links
the initial conditions to the resulting steady-state ATP production rate.
================================================================================

This code produces panels a–d of Figure 5 and Figure S.11 from the manuscript
"Accessible Gibbs energy at metabolic activation limits long-term cell growth".
"""

from matplotlib.lines import Line2D
from scipy.integrate import odeint
from scipy.optimize import root_scalar

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

#----------------------------------------
# helper functions
#----------------------------------------

def ternary_to_cartesian (orn, cit, arg):
    x = arg + cit / 2
    y = cit * np.sqrt(3) / 2
    return x, y

def blend_with_white (color, alpha):
    r, g, b = color[:3]
    return (1 - alpha + alpha * r, 1 - alpha + alpha * g, 1 - alpha + alpha * b)

#----------------------------------------
# initial conditions
#----------------------------------------

def IC (alpha): # initial conditions

    ADP_tot = A_tot / (1 + alpha) # ADP + MgADP
    ATP_tot = alpha * A_tot / (1 + alpha) # ATP + MgATP

    def cubic (M): # cubic equation for y0_Mg_in

        a = K_d_MgADP + K_d_MgATP + A_tot - Mg_tot # auxiliar parameter [M]
        b = (K_d_MgADP * K_d_MgATP + \
            (A_tot / (1 + alpha)) * (K_d_MgATP + alpha * K_d_MgADP) - \
            Mg_tot * (K_d_MgADP + K_d_MgATP)) # auxiliar parameter [M^2]
        c = - Mg_tot * K_d_MgADP * K_d_MgATP # auxiliar parameter [M^3]

        return M**3 + a * M**2 + b * M + c

    sol = root_scalar (cubic, bracket=[1e-12, Mg_tot-1e-12], xtol=1e-12) # solution of the cubic equation for y0_Mg_in

    if not sol.converged:
        raise ValueError ("Solver did not converge")

    Mg = sol.root
    ADP = ADP_tot * K_d_MgADP / (K_d_MgADP + Mg)
#    MgADP = ADP_tot * Mg / (K_d_MgADP + Mg)
#    ATP = ATP_tot * K_d_MgATP / (K_d_MgATP + Mg)
    MgATP = ATP_tot * Mg / (K_d_MgATP + Mg)

    return [ADP, MgATP]

#----------------------------------------
# enzyme velocities
#----------------------------------------

def ArcD_vel (y, nArcD): # ArcD velocity [mol/s]

    arg_out, orn_out, arg_in, orn_in = y

    f_arg_out = (arg_out / KmDarg) / (1 + arg_out / KmDarg + orn_out / KmDorn) # fractional concentration
    f_orn_out = (orn_out / KmDorn) / (1 + arg_out / KmDarg + orn_out / KmDorn) # fractional concentration
    f_arg_in = (arg_in / KmDarg) / (1 + arg_in / KmDarg + orn_in / KmDorn) # fractional concentration
    f_orn_in = (orn_in / KmDorn) / (1 + arg_in / KmDarg + orn_in / KmDorn) # fractional concentration

    V_ArcD = nArcD * kDp * kDm * (f_arg_out * f_orn_in - f_arg_in * f_orn_out) / \
    (kDp * (f_arg_out + f_orn_out) + kDm * (f_orn_in + f_arg_in)) # ArcD velocity [mol/s]

    return V_ArcD

def ArcA_vel (arg, ArcA): # ArcA velocity [M/s]
    return 4 * ArcA * kA * (arg / KmAarg) / (1 + arg / KmAarg + arg**2 / (KmAarg * KiAarg))

def ArcB_vel (y, ArcB):  # ArcB velocity [M/s]

    cit, CP, HPO4, orn = y

    N = kBunfav / (KiBhpo4 * KmBcit) * HPO4 * cit - kBfav / (KmBorn * KiBcp) * orn * CP # numerator

    D = 1 + \
        cit * orn * CP / (KiBcit * KmBorn * KiBcp) + \
        orn * CP / (KmBorn * KiBcp) + \
        CP / KiBcp + \
        orn * KmBcp / (KmBorn * KiBcp) # denominator

    V_ArcB = 6 * ArcB * N / D

    return V_ArcB

def ArcC_vel (y, ArcC): # ArcC velocity [M/s]

    carb, CP, MgADP, MgATP = y

    alp = MgADP / KmCadp # auxiliar parameter
    bta = CP / KmCcp # auxiliar parameter
    pi = MgATP / KmCatp # auxiliar parameter
    rho = carb / KmCcarb # auxiliar parameter

    V_ArcC = 2 * ArcC * (kCp * alp * bta - kCm * pi * rho) * \
        (alp + pi) * (1 + bta + rho) / ((1 + (alp + pi)**2) * \
        (1 + (bta + rho) * (2 + bta + rho))) # ArcC velocity [M/s]

    return V_ArcC

def AAC_vel (y, nT): # AAC velocity [mol/s]

    ADP_out, ATP_out, ADP_in, ATP_in = y

    kTp = 1.9 * kTm / (kTm - 1.9) # [1/s]
    KmTadp = 2.2e-6 * (kTp + kTm) / kTm # [M]

    f_adp_out = (ADP_out / KmTadp) / (1 + ADP_out / KmTadp + ATP_out / KmTatp) # fractional concentration
    f_atp_out = (ATP_out / KmTatp) / (1 + ADP_out / KmTadp + ATP_out / KmTatp) # fractional concentration
    f_adp_in = (ADP_in / KmTadp) / (1 + ADP_in / KmTadp + ATP_in / KmTatp) # fractional concentration
    f_atp_in = (ATP_in / KmTatp) / (1 + ADP_in / KmTadp + ATP_in / KmTatp) # fractional concentration

    V_AAC = nT * kTp * kTm * \
        (f_adp_out * f_atp_in - f_adp_in * f_atp_out) / \
        (kTp * (f_adp_out + f_atp_out) + kTm * (f_atp_in + f_adp_in)) # AAC velocity [mol/s]

    return V_AAC

#----------------------------------------
# system of mass-balance equations
#----------------------------------------

def SDE_wT (y, t, Q_0, arg_out): # system of differential equations with AAC

    ADP_in, cit_in, CP_in, MgATP_in, orn_in = y # inside concentrations [M]

# concentrations [M]

    arg_in = Q_0 - orn_in - cit_in
    Mg_in = (Mg_tot - MgATP_in) / (1 + ADP_in / K_d_MgADP)
    ATP_in = K_d_MgATP * MgATP_in / Mg_in
    MgADP_in = Mg_in * ADP_in / K_d_MgADP

# ArcD, ArcA, ArcB, ArcC, and AAC velocities

    V_ArcD = ArcD_vel([arg_out, 0, arg_in, orn_in], n_ArcD) # ArcD velocity [mol/s]
    V_ArcA = ArcA_vel(arg_in, ArcA) # ArcA velocity [M/s]
    V_ArcB = ArcB_vel([cit_in, CP_in, HPO4_in, orn_in], ArcB) # ArcB velocity [M/s]
    V_ArcC = ArcC_vel([0, CP_in, MgADP_in, MgATP_in], ArcC) # ArcC velocity [M/s]
    V_AAC = AAC_vel([ADP_out, 0, ADP_in, ATP_in], n_AAC) # AAC velocity [mol/s]

# CP degradation

    vCPh = kCPh * CP_in # CP hydrolysis velocity [M/s]

# ADP, MgATP

    a11 = K_d_MgADP + Mg_in + ADP_in # auxiliar parameter [M]
    a12 = ADP_in # auxiliar parameter [M]
    a21 = ATP_in # auxiliar parameter [M]
    a22 = K_d_MgATP + Mg_in + ATP_in # auxiliar parameter [M]

    b1 = - K_d_MgADP * V_ArcC - Mg_in * V_AAC / Vi # auxiliar parameter [M^2/s]
    b2 = K_d_MgATP * V_ArcC + Mg_in * V_AAC / Vi # auxiliar parameter [M^2/s]

    v_MgADP, v_MgATP = (a22 * b1 - a12 * b2) / (a11 * a22 - a12 * a21), (a21 * b1 - a11 * b2) / (a12 * a21 - a11 * a22) # [M/s]

# differential equations

    d_ADP_in = V_AAC / Vi + v_MgADP
    d_cit_in = V_ArcA - V_ArcB
    d_CP_in = V_ArcB - V_ArcC - vCPh
    d_MgATP_in = V_ArcC - v_MgATP
    d_orn_in = - V_ArcD / Vi + V_ArcB

    return [d_ADP_in, d_cit_in, d_CP_in, d_MgATP_in, d_orn_in]

#----------------------------------------
# parameters
#----------------------------------------

T = 30 # temperature [C]
T = T + 273.15 # temperature [K]
R = 8.31 / 1e3 # gas constant [kJ/(K.mol)]
RT = R * T # [J/mol]

n_ArcD = 31.4e-12 # ArcD amount [mol]
n_AAC = 51.0e-12 # AAC amount [mol]

ArcA = 1e-6 # ArcA concentration [M]
ArcB = 2e-6 # ArcB concentration [M]
ArcC = 5e-6 # ArcC concentration [M]

kCPh=1.5e-4 # CP hydrolysis rate [1/s]

K_a_buff = 10**(-6.87) # acid dissociation constant of the reaction H2PO4- = HPO4-2 + H+ [M]
K_a_nh4 = 10**(-9.084) # acid dissociation constant of the reaction NH4+ = NH3 + H+ [M]
K_a_co2 = 10**(-6.33) # acid dissociation constant of the reaction CO2 + H2O = HCO3- + H+ [M]

K_eq_carb=0.53 # equilibrium constant of the reaction carb = NH3 + HCO3- [M]

K_d_MgADP = 0.6e-3 # dissociation constant of the reaction MgADP = Mg+2 + ADP [M]
K_d_MgATP = 0.1e-3 # dissociation constant of the reaction MgATP = Mg+2 + ATP [M]

A_tot = 10e-3 # total nucleotide concentration inside [M]
Mg_tot = 10e-3 # total Mg+2 concentration [M]
KPi_tot = 50e-3 # total buffer concentration [M]

H = 10**(-7) # H+ concentration [M]
HPO4_in = KPi_tot / (1 + H / K_a_buff) # initial HPO4 concentration [M]

kDm, KmDarg, KmDorn = 87, 28.7e-6, 1.402e-3 # ArcD parameters
kA, KmAarg, KiAarg = 6.93, 5.1e-6, 3.2e-3 # ArcA parameters
kBunfav, KmBcp, KiBcp, KmBorn, KmBcit, KiBcit, KiBhpo4, KBeq = 89/6, 0.898e-3, 1.37e-6, 1.095e-3, 0.255e-3, 3.168e-3, 109.00e-3, 13.00e-6 # ArcB parameters
VfvC, kCm, KmCadp, KmCcp, KmCatp, KCeq = 1.50, 573, 2.54e-3, 0.618e-3, 9.5e-3, 6 # ArcC parameters
kTm, KmTatp = 8.5, 4.05e-6 # AAC parameters

kDp = 6.0 * kDm / (kDm - 6.0) # ArcD
kBfav = kBunfav * KmBorn * KiBcp / (KBeq * KiBhpo4 * KmBcit) # Haldane relationship (ArcB)
kCp = 7.3e7 * VfvC / (2 * 60e3) # ArcC
KmCcarb = kCp * KmCcp * KmCadp / (KCeq * kCm * KmCatp) # Haldane relationship (ArcC)
kTp = 1.9 * kTm / (kTm - 1.9) # AAC
KmTadp = 2.2e-6 * (kTp + kTm) / kTm # AAC

Vt = 125e-6 # total volume [l]
Vi = 0.893e-6 # internal volume [l]
Vo = Vt - Vi # external volume [l]

ADP_out = 100e-3 # ADP concentration outside [M]
y0_ADP_in, y0_MgATP_in = IC(0) # initial ADP and MgATP concentrations inside [M]

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

colors = sns.color_palette("deep")
comp_colors = [colors[1], colors[3], colors[4], colors[5]]

compositions = [
    (1, 0, 0),
    (0, 1, 0),
    (0, 0, 1),
    (1/3, 1/3, 1/3)]

Q0_vals = [0.5e-3, 1e-3, 5e-3, 10e-3] # [M]
Q0_alphas = [0.4, 0.6, 0.8, 1]
Q0_labels = ["0.5", "1", "5", "10"]

Q0_range = np.logspace(np.log10(1e-6), np.log10(10e-3), num=3000)

arg_out_fixed = 100e-3 # [M]

t_s_traj = np.arange(0, 24*60*60 + 0.1, 0.1) # 24 h
t_s_short = np.arange(0, 30*60 + 0.1, 0.1) # 30 min
t_min_short = t_s_short / 60 # [min]

figure = plt.figure(figsize=(3, 3), dpi=450)

#----------------------------------------
# Panel a
#----------------------------------------

ax1 = plt.subplot(221)
ax1.axis("off")
ax1.set_aspect("equal")

Q0_traj = Q0_vals[0]

traj = []
sols = []

for (f_orn, f_cit, f_arg) in compositions:

    orn0 = f_orn * Q0_traj
    cit0 = f_cit * Q0_traj

    y0 = [y0_ADP_in, cit0, 0, y0_MgATP_in, orn0]
    sol = odeint (SDE_wT, y0, t_s_traj, args=(Q0_traj, arg_out_fixed), rtol=1e-10, atol=1e-10)

    orn_f = sol[:, 4] / Q0_traj
    cit_f = sol[:, 1] / Q0_traj
    arg_f = 1 - orn_f - cit_f

    x_t, y_t = ternary_to_cartesian (orn_f, cit_f, arg_f)
    traj.append((x_t, y_t))
    sols.append(sol)

# steady state from all-ornithine
orn_ss_f = sols[0][-1, 3] / Q0_traj
cit_ss_f = sols[0][-1, 1] / Q0_traj
arg_ss_f = 1 - orn_ss_f - cit_ss_f

x_ss, y_ss = ternary_to_cartesian (orn_ss_f, cit_ss_f, arg_ss_f)

# grid lines at 1/3 and 2/3
for frac in [1/3, 2/3]:
    p1 = np.array(ternary_to_cartesian(frac, 1-frac, 0))
    p2 = np.array(ternary_to_cartesian(frac, 0, 1-frac))
    ax1.plot([p1[0], p2[0]], [p1[1], p2[1]], "-", color="0.75", linewidth=0.5)

    p1 = np.array(ternary_to_cartesian(0, frac, 1-frac))
    p2 = np.array(ternary_to_cartesian(1-frac, frac, 0))
    ax1.plot([p1[0], p2[0]], [p1[1], p2[1]], "-", color="0.75", linewidth=0.5)

    p1 = np.array(ternary_to_cartesian(0, 1-frac, frac))
    p2 = np.array(ternary_to_cartesian(1-frac, 0, frac))
    ax1.plot([p1[0], p2[0]], [p1[1], p2[1]], "-", color="0.75", linewidth=0.5)

# triangle edges
verts = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3)/2], [0, 0]])
ax1.plot(verts[:, 0], verts[:, 1], "k", linewidth=1)

# vertex labels
offset = 0.1
ax1.text(-offset, -offset, r"\textbf{Orn}", ha="center", va="center", fontsize=7)
ax1.text(1 + offset, -offset, r"\textbf{Arg}", ha="center", va="center", fontsize=7)
ax1.text(0.5, np.sqrt(3)/2 + offset, r"\textbf{Cit}", ha="center", va="center", fontsize=7)

den = [12000, 0, 6000, 16000]
# trajectories
for c_idx, clr in enumerate(reversed(comp_colors)):

    true_idx = len(comp_colors) - 1 - c_idx

    x_t, y_t = traj[true_idx]

    ax1.plot(x_t, y_t, color=clr)

    if true_idx != 1:
        mid = len(x_t) // den[true_idx]
        ax1.annotate("", xy=(x_t[mid+1], y_t[mid+1]), xytext=(x_t[mid], y_t[mid]), arrowprops=dict(arrowstyle="->", color=clr, lw=1, shrinkA=0, shrinkB=0))
    ax1.plot(x_t[0], y_t[0], color=clr, ls="None", marker=".", zorder=100)

ax1.annotate(r"Metabolite space", xy=(0.5, -0.05), xytext=(0.5, -0.35), ha="center", fontsize=7, arrowprops=dict(arrowstyle="->", lw=0.5, shrinkA=0, shrinkB=0))
plt.text(-0.25, 0.75, r"$Q_0 = \SI{0.5}{\milli\Molar}$", fontsize=7)
plt.title(r"\textbf{a}")

#----------------------------------------
# Panel b
#----------------------------------------

ax2 = plt.subplot(222)

cit_ss_range = []

for Q0 in Q0_range:

    y0_ss = [y0_ADP_in, 0, 0, y0_MgATP_in, Q0]
    sol_ss = odeint(SDE_wT, y0_ss, t_s_traj, args=(Q0, arg_out_fixed), rtol=1e-10, atol=1e-10)

    cit_ss_range.append(1e3*sol_ss[-1, 1]) # [mM]

cit_ss_range = np.array(cit_ss_range)

ax2.plot(1e3*Q0_range, cit_ss_range, color="k")
plt.xlim(-0.5, 10)
plt.xticks(np.arange(0, 10+2, 2))
plt.xlabel(r"$Q_0 (\si{\milli\Molar})$")
plt.ylim(-0.5, 10)
plt.yticks(np.arange(0, 10+2, 2))
plt.ylabel(r"Steady-state $[\mathrm{Cit}]_{\mathrm{in}}$ (\si{\milli\Molar})")
plt.title(r"\textbf{b}")

#----------------------------------------
# Panel c
#----------------------------------------

ax3 = plt.subplot(223)

ArcB_velocity = []
ArcC_velocity = []

for Q0 in Q0_vals:

    y0 = [y0_ADP_in, 0, 0, y0_MgATP_in, Q0]
    sol = odeint(SDE_wT, y0, t_s_short, args=(Q0, arg_out_fixed), rtol=1e-10, atol=1e-10, hmax=0.1)

    Mg_in = (Mg_tot - sol[:, 3]) / (1 + sol[:, 0] / K_d_MgADP)
    MgADP_in = Mg_in * sol[:, 0] / K_d_MgADP

    ArcB_velocity.append(60e3 * ArcB_vel([sol[:, 1], sol[:, 2], HPO4_in, sol[:, 4]], ArcB)) # [mM/min]
    ArcC_velocity.append(60e3 * ArcC_vel([0, sol[:, 2], MgADP_in, sol[:, 3]], ArcC)) # [mM/min]

for i in range(len(Q0_vals) - 1, -1, -1):
    blended = blend_with_white (colors[2], Q0_alphas[i])
    ax3.plot(t_min_short, ArcB_velocity[i], color=blended)
    ax3.plot(t_min_short, ArcC_velocity[i], color=blended, label=Q0_labels[i])

ax3.set_xlim(-1.5, 30)
ax3.set_xticks(np.arange(0, 30+5, 5))
ax3.set_xlabel(r"Time (\si{\minute})")
ax3.set_ylim(-0.075, 1.5)
ax3.set_yticks(np.arange(0, 1.5 + 0.3, 0.3))
ax3.set_ylabel(r"ArcB or ArcC velocity (\si{\milli\Molar\per\minute})")
ax3.set_title(r"\textbf{c}")
handles, labels = ax3.get_legend_handles_labels()
handles, labels = [handles[3], handles[1], handles[2], handles[0]], [labels[3], labels[1], labels[2], labels[0]]
ax3.legend(handles, labels, alignment="left", loc="lower right", title=r"$Q_0 (\si{\milli\Molar})$", ncols=2)

#----------------------------------------
# Panel d
#----------------------------------------

ax4 = plt.subplot(224)

arg_out_vals = [1e-6, 1e-5, 1e-4, 1e-1] # [M]

J = {i: [] for i in range(len(arg_out_vals))}

for i, arg_out in enumerate(arg_out_vals):
    for Q0 in Q0_range:

        y0  = [y0_ADP_in, 0, 0, y0_MgATP_in, Q0]
        sol = odeint(SDE_wT, y0, t_s_traj, args=(Q0, arg_out), rtol=1e-10, atol=1e-10)

        Mg_ss = (Mg_tot - sol[-1, 3]) / (1 + sol[-1, 0] / K_d_MgADP)
        MgADP_ss = Mg_ss * sol[-1, 0] / K_d_MgADP
        J_ss = 60e3 * ArcC_vel([0, sol[-1:, 2], np.array([MgADP_ss]), sol[-1:, 3]], ArcC)[0] # [mM/min]

        J[i].append(J_ss) # [mM/min]

for i in range(len(arg_out_vals) - 1, -1, -1):
    blended = blend_with_white (colors[0], Q0_alphas[i])
    ax4.plot(1e3*Q0_range, J[i], color=blended)

ax4.set_xlim(-0.5, 10)
ax4.set_xticks(np.arange(0, 10+2, 2))
ax4.set_xlabel(r"$Q_0$ (\si{\milli\Molar})")
ax4.set_ylim(-0.075, 1.5)
ax4.set_yticks(np.arange(0, 1.5 + 0.3, 0.3))
ax4.set_ylabel(r"ATP production rate (\si{\milli\Molar\per\minute})")
ax4.set_title(r"\textbf{d}")

plt.savefig("Figure5.pdf")

t_s_supp = np.arange(0, 10*60 + 0.1, 0.1)
t_min_supp = t_s_supp / 60

figure_s, axes_s = plt.subplots(3, 2, figsize=(3.5 * 4/5.25, 4), dpi=450)

Q0_supp = [Q0_vals[0], Q0_vals[2]]
Q0_lbl_supp = [Q0_labels[0], Q0_labels[2]]

sols_supp = []
for Q0 in Q0_supp:
    sols_Q0 = []
    for (f_orn, f_cit, f_arg) in compositions:

        orn0 = f_orn * Q0
        cit0 = f_cit * Q0

        y0 = [y0_ADP_in, cit0, 0, y0_MgATP_in, orn0]
        sol = odeint (SDE_wT, y0, t_s_supp, args=(Q0, arg_out_fixed), rtol=1e-10, atol=1e-10)
        sols_Q0.append(sol)

    sols_supp.append(sols_Q0)

panel_labels_s = iter("adbecf")

for row, (key, ylabel) in enumerate(zip(reversed(["orn", "cit", "arg"]), reversed([r"$[\mathrm{Orn}]_\mathrm{in} \, (\si{\milli\Molar})$", r"$[\mathrm{Cit}]_\mathrm{in} \, (\si{\milli\Molar})$", r"$[\mathrm{Arg}]_\mathrm{in} \, (\si{\milli\Molar})$"]))):

    for col, (Q0, Q0_lbl) in enumerate(zip(Q0_supp, Q0_lbl_supp)):

        ax = axes_s[row, col]
        lbl = next(panel_labels_s)

        for c_idx, clr in enumerate(comp_colors):

            sol = sols_supp[col][c_idx]
            orn_t = 1e3 * sol[:, 4]
            cit_t = 1e3 * sol[:, 1]
            arg_t = 1e3 * Q0 - orn_t - cit_t

            if key == "orn": data = orn_t
            elif key == "cit": data = cit_t
            else: data = arg_t

            if col==0:
                ax.plot(t_min_supp[:300], data[:300], color=clr)
                ax.set_ylabel(ylabel)
                ax.set_xlim(0, 0.5)
                ax.set_xticks(np.arange(0, 0.5+0.1, 0.1))

            else:
                ax.plot(t_min_supp, data, color=clr)
                ax.set_xlim(0, 10)
                ax.set_xticks(np.arange(0, 10+2, 2))

            ax.set_ylim(-0.05 * 1e3*Q0, 1.05 * 1e3*Q0)
            ax.set_yticks(np.arange(0, 1e3*Q0 + 1e3*Q0/5, 1e3*Q0/5))

        ax.set_title(r"\textbf{" + lbl + r"}")
        ax.text(0.95, 0.5, rf"$Q_0 = {Q0_lbl} \si{{\milli\Molar}}$", transform=ax.transAxes, ha="right", va="center", fontsize=7)

        if row ==2: ax.set_xlabel(r"Time (\si{\minute})")

figure_s.savefig("Figure5_SI.pdf")
