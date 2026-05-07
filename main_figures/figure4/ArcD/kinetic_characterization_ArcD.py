"""
ArcD kinetic characterization
============================================

This code performs global fitting of the arginine-ornithine antiporter (ArcD) model and generates:

a) ArcD_main.pdf: Figures 4e and 4f
b) ArcD_SI.pdf: Figure S.8
c) fit_report_ArcD.txt: best-fit parameters + uncertainties

Data file required in the same directory: ArcD.csv
"""

from scipy.integrate import odeint
from scipy.optimize import differential_evolution
from scipy.stats import f, iqr

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# ============================== KINETIC MODEL ==============================

def DE (y, t, params, consts): # differential equation

    n_arg_in = y

    kDm, KmDarg, KmDorn = params # ArcD parameters

    kDp=6.0*kDm/(kDm-6.0) # [1/s]

    n_arg_out=consts[0]-n_arg_in # conservation equation
    n_orn_out=consts[1]+n_arg_in # conservation equation
    n_orn_in=consts[2]-n_arg_in # conservation equation

    arg_out, orn_out, arg_in, orn_in = n_arg_out/Vo, n_orn_out/Vo, n_arg_in/Vi, n_orn_in/Vi # concentrations [M]

    f_arg_out=(arg_out/KmDarg)/(1+arg_out/KmDarg+orn_out/KmDorn) # fractional concentration
    f_orn_out=(orn_out/KmDorn)/(1+arg_out/KmDarg+orn_out/KmDorn) # fractional concentration
    f_arg_in=(arg_in/KmDarg)/(1+arg_in/KmDarg+orn_in/KmDorn) # fractional concentration
    f_orn_in=(orn_in/KmDorn)/(1+arg_in/KmDarg+orn_in/KmDorn) # fractional concentration

    V_ArcD=nArcD*kDp*kDm*(f_arg_out*f_orn_in-f_arg_in*f_orn_out)/(kDp*(f_arg_out+f_orn_out)+kDm*(f_orn_in+f_arg_in)) # ArcD velocity [mol/s]

    return V_ArcD

def SDE (t, y0, params, consts): # solution of the differential equation
    return odeint (DE, y0, t, args=(params, consts), hmax=Dt)

def RSS (p, t, data): # residual sum of squares

    rss=[] # residual sum of squares

    for i in range (lul):

        for j in range (ul[i]):

            consts=[y0_n_arg_out[j], y0_n_orn_out[i], y0_n_orn_in[i]]

            model_n_arg_in=SDE(t, 0, p, consts) # arginine amount inside [mol]

            res=model_n_arg_in[:, 0]-data[i][j]
            rss.append(np.sum(res**2))

    x.append(p)
    res_sum_sq.append(np.sum(rss))

    return res_sum_sq[-1]

# ============================== LOAD DATA ==============================

fnam="ArcD.csv" # file name

data=[[], [], [], [], []]

Dt=0.1 # integration time step [s]
sr=0 # skip rows
tnp=0 # total number of points

ul=[3, 7, 7, 7, 7] # upper limits
lul=len(ul)

for i in range (lul):

    for j in range (7):

        Data=np.loadtxt(fnam, delimiter=",", usecols=7, skiprows=1+sr, max_rows=7)
        data[i].append(1e-9*Data)

        sr+=7
        tnp+=len(data[i][j])

data_t_s=np.loadtxt(fnam, delimiter=",", usecols=3, skiprows=1, max_rows=7)
model_t_s=np.arange(0, data_t_s[-1]+Dt, Dt)

#nArcD=6.08e-12 # ArcD amount [mol]
nArcD=1.52e-11 # ArcD amount [mol]

Vt=100e-6 # total volume [l]
Vi=8.829e-7 # internal volume [l]
Vo=Vt-Vi # external volume [l]

y0_arg_out=np.array([1e-6, 2e-6, 5e-6, 10e-6, 25e-6, 50e-6, 100e-6]) # initial arginine concentration outside [M]
y0_orn_out=np.array([23e-9, 110e-9, 110e-9, 230e-9, 460e-9]) # initial ornithine concentration outside [M]
y0_orn_in=np.array([0.5e-3, 2.5e-3, 2.5e-3, 5e-3, 10e-3]) # initial ornithine concentration inside [M]

y0_n_arg_out, y0_n_orn_out, y0_n_orn_in = y0_arg_out*Vo, y0_orn_out*Vo, y0_orn_in*Vi # amounts [mol]

# ============================== FITTING ==============================

x, res_sum_sq = [], []

_kDm=(60, 100) # [1/s]
_KmDarg=(0, 100e-6) # [M]
_KmDorn=(0, 5e-3) # [M]

bounds=[_kDm, _KmDarg, _KmDorn]

result = differential_evolution (RSS, bounds, args=(data_t_s, data), disp=True, polish=False)

op=result.x # optimized parameters
parameters=["kDm", "KmDarg", "KmDorn"]
lp=len(parameters)

indexes=np.where(res_sum_sq <= result.fun*(1+lp/(tnp-lp)*f.ppf(q=0.95, dfn=lp, dfd=tnp)))
x=np.array(x)
x=x[indexes]

err_params=[iqr(x[:, i]) for i in range (lp)]

fitting_file=open("fit_report_ArcD.txt", "w")
fitting_file.write("Value of the objective function = %s\n" % str(result.fun))
fitting_file.write("Parameter,value,error\n")
for i in range (lp):
    fitting_file.write("%s,%s,%s\n" % (parameters[i], str(op[i]), str(err_params[i])))

#op=np.loadtxt("fit_report_ArcD.txt", delimiter=",", skiprows=2, usecols=1) # Load previously fitted parameters

# ============================== PLOTTING ==============================

model_n_arg_in=[[], [], [], [], []]

for i in range (lul):

    for j in range (ul[i]):

        consts=[y0_n_arg_out[j], y0_n_orn_out[i], y0_n_orn_in[i]]

        model=SDE(model_t_s, 0, op, consts)
        model_n_arg_in[i].append(1e9*model[:, 0]) # arginine amount inside [nmol]

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

labels=["1", "2", "5", "10", "25", "50", "100"]

figure=plt.figure(figsize=(1.75, 3.5), dpi=450)

ax=plt.subplot(211)

# 10 mM Orn in, 25 uM Arg out
ax.plot(data_t_s, 1e9*data[4][4], color=colors[4], ls="None", marker=".", label="%s" % labels[4])
ax.plot(model_t_s, model_n_arg_in[4][4], color=colors[4])
plt.text(90, model_n_arg_in[4][4][-1]+0.01, r"Orn in" "\n" r"\SI{10}{\milli\Molar}", fontsize=5, ha="right")
# 10 mM Orn in, 10 uM Arg out
ax.plot(data_t_s, 1e9*data[4][3], color=colors[3], ls="None", marker=".", label="%s" % labels[3])
ax.plot(model_t_s, model_n_arg_in[4][3], color=colors[3])
plt.text(90, model_n_arg_in[4][3][-1]+0.01, r"\SI{10}{\milli\Molar}", fontsize=5, ha="right")
# 5 mM Orn in, 5 uM Arg out
ax.plot(data_t_s, 1e9*data[3][2], color=colors[2], ls="None", marker=".", label="%s" % labels[2])
ax.plot(model_t_s, model_n_arg_in[3][2], color=colors[2])
plt.text(90, model_n_arg_in[3][2][-1]+0.01, r"\SI{5}{\milli\Molar}", fontsize=5, ha="right")
# 5 mM Orn in, 1 uM Arg out
ax.plot(data_t_s, 1e9*data[3][0], color=colors[0], ls="None", marker=".", label="%s" % labels[0])
ax.plot(model_t_s, model_n_arg_in[3][0], color=colors[0])
plt.text(90, model_n_arg_in[3][0][-1]+0.01, r"\SI{5}{\milli\Molar}", fontsize=5, ha="right")

plt.xlim(-4.5, 94.5)
plt.xticks(np.arange(0, 90+15, 15))
plt.ylim(-0.05, 1.05)
plt.yticks(np.arange(0, 1+0.2, 0.2))
plt.ylabel("Arginine amount inside (nmol)")
plt.text(15, 0.75, r"\textbf{ArcD}", fontsize=7, ha="center")
plt.title(r"\textbf{e}")
legend=plt.legend(alignment="left", loc="upper left", ncols=4, title=r"Arg out (\si{\micro\Molar})")

ax=plt.subplot(212)

# 2.5 mM Orn in, 25 uM Arg out
ax.plot(data_t_s, 1e9*data[1][4], color=colors[4], ls="None", marker=".")
ax.plot(model_t_s, model_n_arg_in[1][4], color=colors[4])
plt.text(90, model_n_arg_in[1][4][-1]+0.005, r"\SI{2.5}{\milli\Molar}", fontsize=5, ha="right")
# 2.5 mM Orn in, 10 uM Arg out
ax.plot(data_t_s, 1e9*data[1][3], color=colors[3], ls="None", marker=".")
ax.plot(model_t_s, model_n_arg_in[1][3], color=colors[3])
plt.text(90, model_n_arg_in[1][3][-1]+0.005, r"\SI{2.5}{\milli\Molar}", fontsize=5, ha="right")
# 0.5 mM Orn in, 5 uM Arg out
ax.plot(data_t_s, 1e9*data[0][2], color=colors[2], ls="None", marker=".")
ax.plot(model_t_s, model_n_arg_in[0][2], color=colors[2])
plt.text(90, model_n_arg_in[0][2][-1]+0.005, r"\SI{0.5}{\milli\Molar}", fontsize=5, ha="right")
# 0.5 mM Orn in, 1 uM Arg out
ax.plot(data_t_s, 1e9*data[0][0], color=colors[0], ls="None", marker=".")
ax.plot(model_t_s, model_n_arg_in[0][0], color=colors[0])
plt.text(90, model_n_arg_in[0][0][-1]+0.005, r"\SI{0.5}{\milli\Molar}", fontsize=5, ha="right")

plt.xlim(-4.5, 94.5)
plt.xticks(np.arange(0, 90+15, 15))
plt.xlabel("Time (s)")
plt.ylim(-0.025, 0.525)
plt.yticks(np.arange(0, 0.5+0.1, 0.1))
plt.ylabel("Arginine amount inside (nmol)")
plt.text(15, 0.425, r"\textbf{ArcD}", fontsize=7, ha="center")
plt.title(r"\textbf{f}")

figure.tight_layout()
plt.savefig("ArcD_main.pdf", transparent=True)

figure=plt.figure(figsize=(4.5, 4.5), dpi=450)

ax=plt.subplot(221)
for j in range (ul[4]-1, -1, -1):
    ax.plot(data_t_s, 1e9*data[4][j], color=colors[j], ls="None", marker=".", label="%s" % labels[j])
    ax.plot(model_t_s, model_n_arg_in[4][j], color=colors[j])
plt.xlim(-4.5, 94.5)
plt.xticks(np.arange(0, 90+15, 15))
plt.ylim(-0.075, 1.5)
plt.yticks(np.arange(0, 1.5+0.3, 0.3))
plt.ylabel("Arginine amount inside (nmol)")
plt.title(r"\textbf{a}")
plt.legend(alignment="left", loc="upper left", title=r"Arg out (\si{\micro\Molar})")
plt.text(45, 1.395, r"10 m\textsc{m} Orn in", fontsize=5, ha="center")

ax=plt.subplot(222)
for j in range (ul[3]-1, -1, -1):
    ax.plot(data_t_s, 1e9*data[3][j], color=colors[j], ls="None", marker=".")
    ax.plot(model_t_s, model_n_arg_in[3][j], color=colors[j])
plt.xlim(-4.5, 94.5)
plt.xticks(np.arange(0, 90+15, 15))
plt.ylim(-0.05, 1)
plt.yticks(np.arange(0, 1+0.2, 0.2))
plt.ylabel("Arginine amount inside (nmol)")
plt.title(r"\textbf{b}")
plt.text(45, 0.93, r"5 m\textsc{m} Orn in", fontsize=5, ha="center")

ax=plt.subplot(223)
for j in range (ul[1]-1, -1, -1):
    ax.plot(data_t_s, 1e9*data[1][j], color=colors[j], ls="None", marker=".")
    ax.plot(model_t_s, model_n_arg_in[1][j], color=colors[j])
plt.xlim(-4.5, 94.5)
plt.xticks(np.arange(0, 90+15, 15))
plt.xlabel("Time (s)")
plt.ylim(-0.035, 0.7)
plt.yticks(np.arange(0, 0.7+0.1, 0.1))
plt.ylabel("Arginine amount inside (nmol)")
plt.title(r"\textbf{c}")
plt.text(45, 0.651, r"2.5 m\textsc{m} Orn in", fontsize=5, ha="center")

ax=plt.subplot(224)
for j in range (ul[0]-1, -1, -1):
    ax.plot(data_t_s, 1e9*data[0][j], color=colors[j], ls="None", marker=".")
    ax.plot(model_t_s, model_n_arg_in[0][j], color=colors[j])
plt.xlim(-4.5, 94.5)
plt.xticks(np.arange(0, 90+15, 15))
plt.xlabel("Time (s)")
plt.ylim(-0.007, 0.14)
plt.yticks(np.arange(0, 0.14+0.02, 0.02))
plt.ylabel("Arginine amount inside (nmol)")
plt.title(r"\textbf{d}")
plt.text(45, 0.1302, r"0.5 m\textsc{m} Orn in", fontsize=5, ha="center")

figure.tight_layout()
plt.savefig("ArcD_SI.pdf")
