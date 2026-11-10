"""
Figure S11: Calibration curve of PercevalHR in vesicles.
================================================================================

This code produces Figure S11 from the manuscript
"Accessible Gibbs energy at metabolic activation can constrain steady-state growth".
"""

from scipy.optimize import curve_fit

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def func (x, start, end, k):
    return start + (end - start) * x / (k + x)

fnam = "data.csv" # file name

data_x = np.loadtxt(fnam, delimiter=",", skiprows=1, usecols=0) # ATP/ADP

data_y = np.loadtxt(fnam, delimiter=",", skiprows=1, usecols=4) # F500/F430
data_y_err = np.loadtxt(fnam, delimiter=",", skiprows=1, usecols=5) # F500/F430 (stdev)

popt, pcov = curve_fit (func, data_x, data_y, p0=(0.8, 1.6, 5))
perr = np.sqrt(np.diag(pcov))

parameters=["start", "end", "k"]
lp=len(parameters)

fitting_file = open("fit_report.txt", "w")
fitting_file.write("Parameter,value,error\n")
for i in range (lp):
    fitting_file.write("%s,%s,%s\n" % (parameters[i], str(popt[i]), str(perr[i])))

model_x = np.logspace(-2, 3, num=10000)
model_y = func (model_x, popt[0], popt[1], popt[2])

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

figure=plt.figure(figsize=(1.75, 1.75), dpi=450)

ax=plt.subplot()
ax.plot(model_x, model_y, color="k")
ax.errorbar(data_x, data_y, yerr=data_y_err, color=colors[0], ls="None", marker=".", elinewidth=1, capthick=1, capsize=2.5)
ax.set_xscale("log")
plt.xlim(0.1 * (1e2 / 0.1)**(-0.05), 1e2 * (1e2 / 0.1)**(0.05))
plt.xticks([1e-1, 1e0, 1e1, 1e2])
plt.xlabel(r"ATP-to-ADP ratio")
plt.ylim(0.6, 2)
plt.yticks(np.arange(0.6, 2+0.2, 0.2))
plt.ylabel(r"$\mathrm{F}_{500} / \mathrm{F}_{430}$")

plt.savefig("Calibration_curve_SI.pdf")
