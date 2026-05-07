"""
ArcC kinetic characterization
============================================

This code performs global fitting of the carbamate kinase (ArcC) model and generates:

a) ArcC_main.pdf: Figure 4d
b) fit_report_ArcC.txt: best-fit parameters + uncertainties

Data files required in the same directory: ArcC_ADP.csv, ArcC_ADP_stdev.csv, ArcC_CP.csv, ArcC_CP_stdev.csv
"""

from scipy.optimize import differential_evolution
from scipy.stats import f, iqr

import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, HPacker, VPacker
import numpy as np
import seaborn as sns

# ============================== KINETIC MODEL ==============================

def HE (x, a, b): # Hill equation
    return a*x**2/(b**2+x**2)

def MM (x, a, b): # Michaelis-Menten
    return a*x/(b+x)

def RSS (p, t, data): # residual sum of squares

    KiCadp, KiCcp, VfvC = p

    rss=[] # residual sum of squares

    for i in range (lf):

        if i==0:

            a=VfvC/(1+KiCcp/5e-3) # [mmol/(min.mg)]
            model=HE(ini_con, a, KiCadp)

            res=model-activity[i]
            rss.append(np.sum(res**2))

        if i==1:

            a=VfvC/(1+(KiCadp/5e-3)**2) # [mmol/(min.mg)]
            model=MM(ini_con, a, KiCcp)

            res=model-activity[i]
            rss.append(np.sum(res**2))

    x.append(p)
    res_sum_sq.append(np.sum(rss))

    return res_sum_sq[-1]

ArcC=50e-9 # ArcC concentration [M]

ini_con=np.array([0, 0.1e-3, 0.25e-3, 0.5e-3, 1e-3, 2.5e-3, 5e-3, 10e-3]) # initial concentrations [M]
lic=len(ini_con)

model_ini_con=np.arange(0, ini_con[-1]+0.01e-3, 0.01e-3) # [M]

# ============================== LOAD DATA ==============================

fnam=[["ArcC_ADP.csv", "ArcC_CP.csv"], ["ArcC_ADP_stdev.csv", "ArcC_CP_stdev.csv"]] # file names
lf=len(fnam[0])

data, stdev = [[], []], [[], []]

tnp=0 # total number of points

for i in range (lf):

    for j in range (lic):

        Data=np.loadtxt(fnam[0][i], delimiter=",", usecols=j+1)
        data[i].append(1e-3*Data)

        STDEV=np.loadtxt(fnam[1][i], delimiter=",", usecols=j+1)
        stdev[i].append(1e-3*STDEV)

        tnp+=len(data[i][j])

activity, activity_stdev = [[], []], [[], []]
model_activity = []

for i in range (lf):

    for j in range (lic):

        activity[i].append(60e3*data[i][j][1]/(10*ArcC*7.3e7)) # [mmol/(min.mg)]
        activity_stdev[i].append(60e3*stdev[i][j][1]/(10*ArcC*7.3e7)) # [mmol/(min.mg)]

# ============================== FITTING ==============================

x, res_sum_sq = [], []

_KiCadp=(1e-3, 5e-3) # [M]
_KiCcp=(0.1e-3, 5e-3) # [M]
_VfvC=(0.1, 5) # [mmol/(min.mg)]

bounds=[_KiCadp, _KiCcp, _VfvC]

result = differential_evolution (RSS, bounds, args=(ini_con, data), disp=True, polish=False)

op=result.x # optimized parameters
parameters=["KiCadp", "KiCcp", "VfvC"]
lp=len(parameters)

indexes=np.where(res_sum_sq <= result.fun*(1+lp/(tnp-lp)*f.ppf(q=0.95, dfn=lp, dfd=tnp)))
x=np.array(x)
x=x[indexes]

err_params=[iqr(x[:, i]) for i in range (lp)]

fitting_file=open("fit_report_ArcC.txt", "w")
fitting_file.write("Value of the objective function = %s\n" % str(result.fun))
fitting_file.write("Parameter,value,error\n")
for i in range (lp):
    fitting_file.write("%s,%s,%s\n" % (parameters[i], str(op[i]), str(err_params[i])))

# ============================== PLOTTING ==============================

KiCadp, KiCcp, VfvC = op
kCp=7.3e7*VfvC/(2*60e3) # [1/s]

model_activity=[]

for i in range (lf):

    if i==0:

        a=VfvC/(1+KiCcp/5e-3) # [mmol/(min.mg)]
        model_activity.append(HE(model_ini_con, a, KiCadp))

    if i==1:

        a=VfvC/(1+(KiCadp/5e-3)**2) # [mmol/(min.mg)]
        model_activity.append(MM(model_ini_con, a, KiCcp))

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

labels=["0", "0.1", "0.25", "0.5", "1", "2.5", "5", "10"]

figure=plt.figure(figsize=(1.75, 1.75), dpi=450)

ax=plt.subplot()
ax.plot(1e3*model_ini_con, model_activity[1], color=colors[1])
ax.errorbar(1e3*ini_con, activity[1], yerr=activity_stdev[1], color=colors[1], ls="None", marker=".", elinewidth=1, capthick=1, capsize=2.5)
ax.plot(1e3*model_ini_con, model_activity[0], color=colors[0], zorder=1000)
ax.errorbar(1e3*ini_con, activity[0], yerr=activity_stdev[0], color=colors[0], ls="None", marker=".", elinewidth=1, capthick=1, capsize=2.5, zorder=1001)
plt.xlim(-0.5, 10.5)
plt.xticks(np.arange(0, 10+2, 2))
#
plt.xlabel(r"CP / MgADP concentration (\si{\milli\Molar})", alpha=0)
xl1=TextArea(r"CP", textprops=dict(color="#dd8452", fontsize=7))
xl2=TextArea(r"/", textprops=dict(fontsize=7))
xl3=TextArea(r"MgADP", textprops=dict(color="#4c72b0", fontsize=7))
xl4=TextArea(r"concentration (\si{\milli\Molar})", textprops=dict(fontsize=7))
xl=HPacker(children=[xl1, xl2, xl3, xl4], pad=0, sep=1.95)
anchored_xl=AnchoredOffsetbox(loc="center", pad=0, borderpad=0, child=xl, frameon=False, bbox_to_anchor=(0.5, -0.1965), bbox_transform=ax.transAxes)
ax.add_artist(anchored_xl)
#
plt.ylim(-0.075, 1.575)
plt.yticks(np.arange(0, 1.5+0.3, 0.3))
plt.ylabel(r"ArcC activity (\si{\milli\mol\per\minute\per\milli\gram})")
plt.text(8, 0.3, r"\textbf{ArcC}", fontsize=7, ha="center")
plt.title(r"\textbf{b}")

figure.tight_layout()
plt.savefig("ArcC_main.pdf", transparent=True)
