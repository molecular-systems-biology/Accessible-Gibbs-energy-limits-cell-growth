"""
Figure 3b -- 3e: Higher ornithine inside drives faster ATP production despite identical arginine outside.
================================================================================

This code produces panels b -- e of Figure 3 from the manuscript
"Accessible Gibbs energy at metabolic activation can constrain steady-state growth".
"""

from matplotlib.lines import Line2D

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def DWU (time, values, stdev): # derivative with uncertainty

    N=len(values)
    dvals=np.zeros(N)
    dstd=np.zeros(N)

# forward difference at first point
    dt=time[1]-time[0]
    dvals[0]=(values[1]-values[0])/dt
    dstd[0]=np.sqrt(stdev[1]**2+stdev[0]**2)/abs(dt)

# central differences for interior points
    for i in range(1, N-1):
        dt=time[i+1]-time[i-1]
        dvals[i]=(values[i+1]-values[i-1])/dt
        dstd[i]=np.sqrt(stdev[i+1]**2+stdev[i-1]**2)/abs(dt)

# backward difference at last point
    dt=time[-1]-time[-2]
    dvals[-1]=(values[-1]-values[-2])/dt
    dstd[-1]=np.sqrt(stdev[-1]**2+stdev[-2]**2)/abs(dt)

    return dvals, dstd

def load (fnam): # time, mean and sample standard deviation of the replicates in a CSV file

    data=np.genfromtxt(fnam, delimiter=",", skip_header=1)

    time=data[:, 0] # [min]
    reps=data[:, 1:] # F500/F430 of each replicate

    return time, np.nanmean(reps, axis=1), np.nanstd(reps, axis=1, ddof=1)

y0_orn_in=[0.5e-3, 10e-3] # initial ornithine concentration inside [M]
y0_ADP_out=[0.1e-3, 0.5e-3] # initial ADP concentration outside [M]

fnam=["0d5_orn_0_adp.csv", "10_orn_0_adp.csv"] # file name

data_t_s_woT, data_t_min_woT = [], []
data_woT, stdev_woT = [], []

dF_dt_woT, dF_dt_std_woT = [], []

for i in range (len(y0_orn_in)):

    t, mean, stdev = load (fnam[i])

    data_t_min_woT.append(t)
    data_t_s_woT.append(data_t_min_woT[i]*60)

    data_woT.append(mean)
    stdev_woT.append(stdev)

    dvals, dstd = DWU (data_t_min_woT[i], data_woT[i], stdev_woT[i])

    dF_dt_woT.append(dvals)
    dF_dt_std_woT.append(dstd)

fnam=[["0d5_orn_0d1_adp.csv", "0d5_orn_0d5_adp.csv"], ["10_orn_0d1_adp.csv", "10_orn_0d5_adp.csv"]] # file name

data_t_s_wT, data_t_min_wT = [[], []], [[], []]
data_wT, stdev_wT = [[], []], [[], []]

dF_dt_wT, dF_dt_std_wT = [[], []], [[], []]

for i in range (len(y0_orn_in)):

    for j in range (len(y0_ADP_out)):

        t, mean, stdev = load (fnam[i][j])

        data_t_min_wT[i].append(t)
        data_t_s_wT[i].append(data_t_min_wT[i][j]*60)

        data_wT[i].append(mean)
        stdev_wT[i].append(stdev)

        dvals, dstd = DWU (data_t_min_wT[i][j], data_wT[i][j], stdev_wT[i][j])

        dF_dt_wT[i].append(dvals)
        dF_dt_std_wT[i].append(dstd)

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
plt.rc("legend", borderpad=0.25, columnspacing=1, fontsize=5, labelspacing=0.25, title_fontsize=5)
plt.rc("lines", linewidth=1, markeredgewidth=0, markersize=8)

colors=sns.color_palette("deep") # color palette

labels=["0", "0.1", "0.5"]
n_subp=1

figure=plt.figure(figsize=(3.5, 3.5), dpi=450)

for i in range (len(y0_orn_in)):

    ax=figure.add_subplot(2, 2, n_subp)
    legend_handles = []
    for j in [1, 0]:
        ax.fill_between(data_t_min_wT[i][j], data_wT[i][j]-stdev_wT[i][j], data_wT[i][j]+stdev_wT[i][j], color=colors[1+j], edgecolor=None, alpha=0.5)
        ax.plot(data_t_min_wT[i][j], data_wT[i][j], color=colors[1+j])
        legend_handles.append(Line2D([0], [0], color=colors[1+j], label=labels[1+j]))
    ax.fill_between(data_t_min_woT[i], data_woT[i]-stdev_woT[i], data_woT[i]+stdev_woT[i], color=colors[0], edgecolor=None, alpha=0.5)
    ax.plot(data_t_min_woT[i], data_woT[i], color=colors[0])
    legend_handles.append(Line2D([0], [0], color=colors[0], label=labels[0]))
    if i==0:
        plt.xlim(0, 15)
        plt.xticks(np.arange(0, 15+5, 5))
        plt.ylabel(r"$\mathrm{F}_{500} / \mathrm{F}_{430}$")
        plt.title(r"\textbf{b}")
        plt.text(7.5, 0.7, r"\SI{0.5}{\milli\Molar} Orn in", fontsize=7, ha="center", va="baseline")
        ax.legend(handles=legend_handles[::-1], alignment="left", ncols=3, loc="upper center", title=r"ADP out (\si{\milli\Molar})")
    if i==1:
        plt.xlim(0, 15)
        plt.xticks(np.arange(0, 15+5, 5))
        plt.title(r"\textbf{c}")
        plt.text(7.5, 0.7, r"\SI{10}{\milli\Molar} Orn in", fontsize=7, ha="center", va="baseline")
    plt.ylim(0.6, 1.8)
    plt.yticks(np.arange(0.6, 1.8+0.2, 0.2))

    ax=figure.add_subplot(2, 2, n_subp+2)
    for j in [1, 0]:
        ax.plot(data_t_min_wT[i][j], dF_dt_wT[i][j], color=colors[1+j])
    ax.plot(data_t_min_woT[i], dF_dt_woT[i], color=colors[0])
    if i==0:
        plt.xlim(0, 15)
        plt.xticks(np.arange(0, 15+5, 5))
        plt.ylabel(r"Rate of $\mathrm{F}_{500} / \mathrm{F}_{430}$ change (\si{\per\minute})")
        plt.title(r"\textbf{d}")
        plt.text(7.5, 0.125, r"\SI{0.5}{\milli\Molar} Orn in", fontsize=7, ha="center", va="center")
    if i==1:
        plt.xlim(0, 15)
        plt.xticks(np.arange(0, 15+5, 5))
        plt.title(r"\textbf{e}")
        plt.text(7.5, 0.125, r"\SI{10}{\milli\Molar} Orn in", fontsize=7, ha="center", va="center")
    plt.xlabel(r"Time (\si{\minute})")
    plt.ylim(-0.05, 0.15)
    plt.yticks(np.arange(-0.05, 0.15+0.05, 0.05))

    n_subp+=1

plt.savefig("Figure3b_e.pdf")
plt.savefig("Figure3b_e.svg")
