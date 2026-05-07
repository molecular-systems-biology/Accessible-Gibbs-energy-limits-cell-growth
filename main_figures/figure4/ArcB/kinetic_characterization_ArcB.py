"""
ArcB kinetic characterization
============================================

This code performs global fitting of the ornithine transcarbamoylase (ArcB) model and generates:

a) ArcB_main.pdf: Figures 4b and 4c
b) ArcB_SI.pdf: Figure S.5
c) fit_report_ArcB.txt: best-fit parameters + uncertainties

Data files required in the same directory: ArcB_orn.csv, ArcB_CP.csv, ArcB_cit.csv
"""

from scipy.integrate import odeint
from scipy.optimize import differential_evolution
from scipy.stats import f, iqr

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# ============================== KINETIC MODEL ==============================

def ArcB_vel (y, KmBcp, KiBcp, KmBorn, KmBcit, KiBcit, KiBhpo4, KBeq, ArcB):  # ArcB velocity [M/s]

    cit, CP, HPO4, orn = y

    kBfav = kBunfav * KmBorn * KiBcp / (KBeq * KiBhpo4 * KmBcit) # Haldane relationship (ArcB)

    N = kBunfav / (KiBhpo4 * KmBcit) * HPO4 * cit - kBfav / (KmBorn * KiBcp) * orn * CP # numerator

    D = 1 + \
        cit * orn * CP / (KiBcit * KmBorn * KiBcp) + \
        orn * CP / (KmBorn * KiBcp) + \
        CP / KiBcp + \
        orn * KmBcp / (KmBorn * KiBcp) # denominator

    V_ArcB = 6 * ArcB * N / D

    return V_ArcB

def SDE (y, t, params, ArcB): # system of differential equations

    KmBcp, KiBcp, KmBorn, KmBcit, KiBcit, KiBhpo4, KBeq = params

    V_ArcB=ArcB_vel(y, KmBcp, KiBcp, KmBorn, KmBcit, KiBcit, KiBhpo4, KBeq, ArcB) # ArcB velocity [M/s]

    vCPh=kCPh*y[1] # CP hydrolysis velocity [M/s]
    vbuff=(V_ArcB-vCPh)/(1+K_a_buff/H) # buffer velocity [M/s]

    return [-V_ArcB, V_ArcB-vCPh, -V_ArcB+vCPh+vbuff, V_ArcB]

def SSDE (t, y0, params, ArcB): # solution of the system of differential equations
    return odeint (SDE, y0, t, args=(params, ArcB))

def RSS (p, t, data): # residual sum of squares

    rss=[] # residual sum of squares

    for i in range (lf):

        for j in range (lic[i]):

            if i==0: y0=[0, 5e-3, y0_HPO4, y0_orn[j]] # t=0
            if i==1: y0=[0, y0_CP[j], y0_HPO4, 5e-3] # t=0
            if i==2: y0=[y0_cit[j], 0, y0_HPO4, 0] # t=0

            model=SSDE(t[i][j], y0, p, ArcB[i])

            if i==0: res=(model[:, 0]-data[i][j])/data[i][4][-1]
            if i==1: res=(model[:, 0]-data[i][j])/data[i][5][-1]
            if i==2: res=(model[:, 3]-data[i][j])/data[i][3][-1]

            rss.append(np.sum(res**2))

    x.append(p)
    res_sum_sq.append(np.sum(rss))

    return res_sum_sq[-1]

# ============================== LOAD DATA ==============================

fnam=["ArcB_orn.csv", "ArcB_CP.csv", "ArcB_cit.csv"] # file names
lf=len(fnam)

y0_orn=[0.25e-3, 0.5e-3, 0.7e-3, 1e-3, 5e-3, 10e-3] # initial ornithine concentrations [M]
y0_CP=[0.015e-3, 0.075e-3, 0.5e-3, 1e-3, 2.5e-3, 5e-3] # initial CP concentrations [M]
y0_cit=[5e-3, 20e-3, 35e-3, 50e-3] # initial citrulline concentrations [M]
lic=[len(y0_orn), len(y0_CP), len(y0_cit)]

data_t_s, data_t_min = [[], [], []], [[], [], []]
data = [[], [], []]
model_t_s, model_t_min = [[], [], [], []], [[], [], []]

Dt=0.5 # integration time step [s]
mr=8 # max rows
tnp=0 # total nunmber of points

for i in range (lf):

    sr=0 # skip rows

    for j in range (lic[i]):

        if i==0 and j==1: mr=6
        if i==2 and (j==0 or j==1): mr=9
        if i==2 and j==3: mr=10

        data_t_s[i].append(np.loadtxt(fnam[i], delimiter=",", usecols=0, skiprows=1+sr, max_rows=mr))
        data_t_min[i].append(data_t_s[i][j]/60)

        Data=np.loadtxt(fnam[i], delimiter=",", usecols=1, skiprows=1+sr, max_rows=mr)
        data[i].append(1e-6*Data)

        model_t_s[i].append(np.arange(0, data_t_s[i][j][-1]+Dt, Dt))
        model_t_min[i].append(model_t_s[i][j]/60)

        sr+=mr
        mr=8

    tnp+=sr

ArcB=[15e-9, 10e-9, 15e-9] # ArcB concentration [M]

H=10**(-7) # H+ concentration [M]
K_a_buff=10**(-6.87) # acid dissociation constant of the reaction H2PO4- <---> HPO4-2 + H+ [M]
kCPh=1.5e-4 # CP hydrolysis rate [1/s]
KPi_tot=50e-3 # total buffer concentration [M]
y0_HPO4=KPi_tot/(1+H/K_a_buff) # initial HPO4-2 concentration [M]

kBunfav=89/6 # [1/s]

# ============================== FITTING ==============================

x, res_sum_sq = [], []

bounds=[(0.1e-3, 1e-3), (1e-6, 100e-6), (1e-3, 5e-3), (0.1e-3, 1e-3), (1e-3, 5e-3), (50e-3, 150e-3), (1e-6, 50e-6)]

result = differential_evolution (RSS, bounds, args=(data_t_s, data), disp=True, polish=False)

op=result.x # optimized parameters
parameters=["KmBcp", "KiBcp", "KmBorn", "KmBcit", "KiBcit", "KiBhpo4", "KBeq"]
lp=len(parameters)

indexes=np.where(res_sum_sq <= result.fun*(1+lp/(tnp-lp)*f.ppf(q=0.95, dfn=lp, dfd=tnp)))
x=np.array(x)
x=x[indexes]

err_params=[iqr(x[:, i]) for i in range (lp)]

fitting_file=open("fit_report_ArcB.txt", "w")
fitting_file.write("Value of the objective function = %s\n" % str(result.fun))
fitting_file.write("Parameter,value,error\n")
for i in range (lp):
    fitting_file.write("%s,%s,%s\n" % (parameters[i], str(op[i]), str(err_params[i])))

#op=np.loadtxt("fit_report_ArcB.txt", delimiter=",", skiprows=1, usecols=1) # Load previously fitted parameters

# ============================== PLOTTING ==============================

model_cit=[[], []]
model_orn=[]

for i in range (lf):

    for j in range (lic[i]):

        if i==0: y0=[0, 5e-3, y0_HPO4, y0_orn[j]] # t=0
        if i==1: y0=[0, y0_CP[j], y0_HPO4, 5e-3] # t=0
        if i==2: y0=[y0_cit[j], 0, y0_HPO4, 0] # t=0

        model=SSDE(model_t_s[i][j], y0, op, ArcB[i])

        if i<=1: model_cit[i].append(1e3*model[:, 0]) # citrulline concentration [mM]
        if i==2: model_orn.append(1e6*model[:, 3]) # ornithine concentration [uM]

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
plt.rc("legend", borderpad=0.25, columnspacing=0, fontsize=5, handletextpad=0, labelspacing=0.25, title_fontsize=5)
plt.rc("lines", linewidth=1, markeredgewidth=0, markersize=8)

colors=sns.color_palette("deep") # color palette

lorn=["0.25", "0.5", "0.7", "1", "5", "10"]
lcp=["0.015", "0.075", "0.5", "1", "2.5", "5"]
lcp_2=["2.5", "5"]
lcit=["5", "20", "35", "50"]

figure=plt.figure(figsize=(1.75, 3.5), dpi=450)

ax=plt.subplot(211)

# 5 mM orn, 5 mM CP
ax1, = ax.plot(data_t_min[1][5], 1e3*data[1][5], color=colors[5], ls="None", marker=".", label="%s" % lcp[5])
ax.plot(model_t_min[1][5], model_cit[1][5], color=colors[5])
# 5 mM orn, 2.5 mM CP
ax2, = ax.plot(data_t_min[1][4], 1e3*data[1][4], color=colors[4], ls="None", marker=".", label="%s" % lcp[4])
ax.plot(model_t_min[1][4], model_cit[1][4], color=colors[4])
# 5 mM CP, 0.7 mM orn
ax3, = ax.plot(data_t_min[0][2], 1e3*data[0][2], color=colors[2], ls="None", marker=".", label="%s" % lorn[2])
ax.plot(model_t_min[0][2], model_cit[0][2], color=colors[2])
# 5 mM CP, 0.25 mM orn
ax4, = ax.plot(data_t_min[0][0], 1e3*data[0][0], color=colors[0], ls="None", marker=".", label="%s" % lorn[0])
ax.plot(model_t_min[0][0], model_cit[0][0], color=colors[0])

legend=plt.legend(alignment="left", handles=[ax1, ax2], loc="upper left", ncols=2, title=r"\SI{5}{\milli\Molar} Orn" "\n" r"CP (\si{\milli\Molar})")
ax.add_artist(legend)
plt.legend(alignment="left", handles=[ax3, ax4], loc="lower right", ncols=2, title=r"\SI{5}{\milli\Molar} CP" "\n" r"Orn (\si{\milli\Molar})")

plt.xlim(-1.5, 31.5)
plt.xticks(np.arange(0, 30+6, 6))
plt.xlabel("Time (min)")
plt.ylim(-0.175, 3.5)
plt.yticks(np.arange(0, 3.5+0.5, 0.5))
plt.ylabel(r"Citrulline concentration (\si{\milli\Molar})")
plt.text(24, 1, r"\textbf{ArcB}", fontsize=7, ha="center")
plt.title(r"\textbf{c}")

ax=plt.subplot(212)
for j in range (lic[2]-1, -1, -1):
    ax.plot(data_t_min[2][j], 1e6*data[2][j], color=colors[j], ls="None", marker=".", label="%s" % lcit[j])
    ax.plot(model_t_min[2][j], model_orn[j], color=colors[j])
plt.xlim(-6.25, 131.25)
plt.xticks(np.arange(0, 125+25, 25))
plt.xlabel("Time (min)")
plt.ylim(-12.5, 250)
plt.yticks(np.arange(0, 250+50, 50))
plt.ylabel(r"Ornithine concentration (\si{\micro\Molar})")
plt.legend(alignment="left", loc="upper left", ncols=4, title=r"Cit (\si{\milli\Molar})")
plt.text(100, 12.5, r"\textbf{ArcB}", fontsize=7, ha="center")
plt.title(r"\textbf{d}")

figure.tight_layout()
plt.savefig("ArcB_main.pdf", transparent=True)

figure=plt.figure(figsize=(4.5, 2.25), dpi=450)

ax=plt.subplot(121)
for j in range (lic[0]-1, -1, -1):
    ax.plot(data_t_min[0][j], 1e3*data[0][j], color=colors[j], ls="None", marker=".", label="%s" % lorn[j])
    ax.plot(model_t_min[0][j], model_cit[0][j], color=colors[j])
plt.xlim(-0.75, 15.75)
plt.xticks(np.arange(0, 15+3, 3))
plt.xlabel("Time (min)")
plt.ylim(-0.175, 3.5)
plt.yticks(np.arange(0, 3.5+0.5, 0.5))
plt.ylabel(r"Citrulline concentration (\si{\milli\Molar})")
plt.title(r"\textbf{a}")
plt.legend(alignment="left", loc="upper left", title=r"\SI{5}{\milli\Molar} CP" "\n" r"Orn (\si{\milli\Molar})")

ax=plt.subplot(122)
for j in range (lic[1]-1, -1, -1):
    ax.plot(data_t_min[1][j], 1e3*data[1][j], color=colors[j], ls="None", marker=".", label="%s" % lcp[j])
    ax.plot(model_t_min[1][j], model_cit[1][j], color=colors[j])
plt.xlim(-1.5, 31.5)
plt.xticks(np.arange(0, 30+6, 6))
plt.xlabel("Time (min)")
plt.ylim(-0.175, 3.5)
plt.yticks(np.arange(0, 3.5+0.5, 0.5))
plt.title(r"\textbf{b}")
plt.legend(alignment="left", loc="upper left", title=r"\SI{5}{\milli\Molar} Orn" "\n" r"CP (\si{\milli\Molar})")

figure.tight_layout()
plt.savefig("ArcB_SI.pdf", transparent=True)
