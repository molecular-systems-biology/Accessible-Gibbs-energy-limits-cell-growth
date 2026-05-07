"""
ArcA kinetic characterization
====================================

This code performs global fitting of the arginine deiminase (ArcA) kinetic model and generates:

a) ArcA_main.pdf: Figure 4a
b) ArcA_SI.pdf: Figure S.4
c) fit_report_ArcA.txt: best-fit parameters + uncertainties

Data files required in the same directory: ArcA_Arg*.csv (8 files)
"""

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.integrate import odeint
from scipy.optimize import differential_evolution
from scipy.stats import f, iqr, linregress

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# ============================== KINETIC MODEL ==============================

def ArcA_vel (y, params): # ArcA velocity

    arg = y

    kA, KmAarg = params # ArcA parameters

    V_ArcA=4*ArcA*kA*(arg/KmAarg)/(1+arg/KmAarg+arg**2/(KmAarg*KiAarg)) # ArcA velocity [M/s]

    return V_ArcA

def SDE (y, t, params): # system of differential equations

    V_ArcA=ArcA_vel(y[0], params) # ArcA velocity [M/s]

    return [-V_ArcA, V_ArcA]

def SSDE (t, y0, params): # solution of the system of differential equations
    return odeint (SDE, y0, t, args=(params,))

def RSS (p, t, data): # residual sum of squares

    rss=[] # residual sum of squares

    for i in range (lf):

        y0=[y0_arg[i], 0] # t=0

        model=SSDE(t[i], y0, p)
        model_cit=model[:, 1] # citruline concentration [M]

        res=model_cit-data[i]
        rss.append(np.sum(res**2))

    x.append(p)
    res_sum_sq.append(np.sum(rss))

    return res_sum_sq[-1]

# ============================== LOAD DATA ==============================

fnam=["ArcA_Arg1uM.csv", "ArcA_Arg2d5uM.csv", "ArcA_Arg4uM.csv", "ArcA_Arg5uM.csv", "ArcA_Arg10uM.csv", "ArcA_Arg25uM.csv", "ArcA_Arg50uM.csv", "ArcA_Arg250uM.csv"] # file names
lf=len(fnam)

data_t_s, data_t_min = [], []
data = []
model_t_s, model_t_min = [], []

Dt=0.5 # integration time step [s]
tnp=0 # total nunmber of points

for i in range (lf):

    data_t_s.append(np.loadtxt(fnam[i], delimiter=",", usecols=0))
    data_t_min.append(data_t_s[i]/60)

    Data=np.loadtxt(fnam[i], delimiter=",", usecols=1)
    data.append(1e-6*Data)

    model_t_s.append(np.arange(0, data_t_s[i][-1]+Dt, Dt))
    model_t_min.append(model_t_s[i]/60)

    tnp+=len(data_t_s[i])

ArcA=0.02e-6 # ArcA concentration [M]
KiAarg=3.2e-3 # [M]

y0_arg=np.array([1e-6, 2.5e-6, 4e-6, 5e-6, 10e-6, 25e-6, 50e-6, 250e-6]) # arginine concentration at t=0 [M]

#  ============================== FITTING ==============================

x, res_sum_sq = [], []

_kA=(1, 100) # [1/s]
_KmAarg=(0.5e-6, 50e-6) # [M]

bounds=[_kA, _KmAarg]
initial_guess=[10, 5e-6]

result = differential_evolution (RSS, bounds, args=(data_t_s, data), disp=True, polish=True, x0=initial_guess)

op=result.x # optimized parameters
parameters=["kA", "KmAarg"]
lp=len(parameters)

indexes=np.where(res_sum_sq <= result.fun*(1+lp/(tnp-lp)*f.ppf(q=0.95, dfn=lp, dfd=tnp)))
x=np.array(x)
x=x[indexes]

err_params=[iqr(x[:, i]) for i in range (lp)]

fitting_file=open("fit_report_ArcA.txt", "w")
fitting_file.write("Value of the objective function = %s\n" % str(result.fun))
fitting_file.write("Parameter,value,error\n")
for i in range (lp):
    fitting_file.write("%s,%s,%s\n" % (parameters[i], str(op[i]), str(err_params[i])))

#op=np.loadtxt("fit_report_ArcA.txt", delimiter=",", skiprows=2, usecols=1) # Load previously fitted parameters

# ============================== PLOTTING ==============================

model_y0_arg=np.arange(0, 250e-6+0.25e-6, 0.25e-6)
model_activity=60e6*ArcA_vel(model_y0_arg, op)/(ArcA*1.94e8) # ArcA activity [umol/(min.mg)]

activity=[]
for i in range (lf):
    lr=linregress(data_t_min[i][:3], data[i][:3])
    activity.append(1e6*lr.slope/(ArcA*1.94e8)) # ArcA activity [umol/(min.mg)]

model_cit=[]

for i in range (lf):

    y0=[y0_arg[i], 0] # t=0

    model=SSDE(model_t_s[i], y0, op)
    model_cit.append(1e6*model[:, 1]) # citrulline concentration [uM]

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

labels=["1", "2.5", "4", "5", "10", "25", "50", "250"]

figure=plt.figure(figsize=(1.75, 1.75), dpi=450)

ax=plt.subplot()
ax.plot(1e6*model_y0_arg, model_activity, color=colors[0])
ax.plot(1e6*y0_arg, activity, color=colors[0], ls="None", marker=".")
plt.xlim(-12.5, 262.5)
plt.xticks(np.arange(0, 250+50, 50))
plt.xlabel(r"Arginine concentration (\si{\micro\Molar})")
plt.ylim(0, 10)
plt.yticks(np.arange(0, 10+2, 2))
plt.ylabel(r"ArcA activity (\si{\micro\mol\per\minute\per\milli\gram})")
plt.text(125, 9, r"\textbf{ArcA}", fontsize=7, ha="center")
plt.title(r"\textbf{a}")

figure.tight_layout()
plt.savefig("ArcA_main.pdf")

figure=plt.figure(figsize=(4.5, 2.25), dpi=450)

ax=plt.subplot(121)
for i in range (lf-1, -1, -1):
    ax.plot(data_t_min[i], 1e6*data[i], color=colors[i], ls="None", marker=".", label="%s" % labels[i])
    ax.plot(model_t_min[i], model_cit[i], color=colors[i])
plt.xlim(-0.6, 12)
plt.xticks(np.arange(0, 12+2, 2))
plt.xlabel("Time (min)")
plt.ylim(-15, 300)  
plt.yticks(np.arange(0, 300+50, 50))
plt.ylabel(r"Citrulline concentration (\si{\micro\Molar})")
plt.title(r"\textbf{a}")
legend=plt.legend(alignment="left", loc="upper left", title=r"Arg (\si{\micro\Molar})")
rectangle=patches.Rectangle((0, 0), 2, 30, edgecolor="k", facecolor="None", linewidth=0.5, zorder=1000)
ax.add_patch(rectangle)
plt.annotate("", xytext=(2.5, 15), xy=(5.5, 50), arrowprops=dict(arrowstyle="->, head_length=0.3, head_width=0.2", linewidth=0.5))

ax_ins=inset_axes(ax, height="100%", width="100%", bbox_to_anchor=(0.6, 0.15, 0.385, 0.385), bbox_transform=ax.transAxes)
for i in range (lf-3, -1, -1):
    ax_ins.plot(data_t_min[i], 1e6*data[i], color=colors[i], ls="None", marker=".")
    ax_ins.plot(model_t_min[i], model_cit[i], color=colors[i])
plt.xlim(0, 2)
plt.xticks(np.arange(0, 2+1, 1), fontsize=5)
plt.ylim(0, 30)
plt.yticks(np.arange(0, 30+10, 10), fontsize=5)

ax=plt.subplot(122)
ax.plot(1e6*x[:, 1], x[:, 0], color=colors[0], ls="None", marker=".", alpha=0.5)
ax.errorbar(1e6*op[1], op[0], xerr=1e6*err_params[1], yerr=err_params[0], color="k", elinewidth=1, capthick=1, capsize=2.5)
plt.xlim(0, 12)
plt.xticks(np.arange(0, 12+2, 2))
plt.xlabel(r"$K^\mathrm{mA}_\mathrm{arg}$ (\si{\micro\Molar})", labelpad=0)
plt.ylim(6.5, 7.5)
plt.yticks(np.arange(6.5, 7.5+0.25, 0.25))
plt.ylabel(r"$k^+_\mathrm{A}$ (\si{\per\second})")
plt.title(r"\textbf{b}")

figure.tight_layout()
plt.savefig("ArcA_SI.pdf")
