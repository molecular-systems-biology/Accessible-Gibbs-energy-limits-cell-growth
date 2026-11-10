"""
Figure 1e -- 1g: Cellular metabolism modeled as a series of lumped reactions.
================================================================================

This code produces panels e -- g of Figure 1 from the manuscript
"Accessible Gibbs energy at metabolic activation can constrain steady-state growth".
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

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

colors=sns.color_palette("deep")

lss=["dotted", "dashdot", "dashed", "solid"]
alphas=[0.4, 0.6, 0.8, 1]

figure = plt.figure(figsize=(3.5, 1), dpi=450)

####################
# Figure 1e
####################

ax = plt.subplot(131)

for i in range(4):
    data_e = np.load(f"e{i}.npz")
    ax.plot(data_e["x"], data_e["y"], color=colors[0], ls=lss[i], alpha=alphas[i])
ax.set_xlim(0, 500)
ax.set_xticks([0, 250, 500])
ax.set_xlabel(r"[C]$^0_{\mathrm{in}}$ (\si{\micro\Molar})")
ax.set_ylim(0, 1)
ax.set_yticks(np.arange(0, 1+0.5, 0.5))
ax.set_ylabel(r"$\mu / \mu_{\mathrm{max}}$")
#ax.set_title(r"\textbf{e}")

####################
# Figure 1f
####################

ax = plt.subplot(132)

for i in range(4):
    data_f = np.load(f"f{i}.npz")
    ax.plot(data_f["x"], data_f["y"], color=colors[0], ls=lss[i], alpha=alphas[i])
ax.set_xlim(0, 5)
ax.set_xticks([0, 2.5, 5])
ax.set_xlabel(r"[ATP]$_0$ (\si{\milli\Molar})")
ax.set_ylim(0, 1)
ax.set_yticks(np.arange(0, 1+0.5, 0.5))
#ax.set_title(r"\textbf{f}")

####################
# Figure 1g
####################

ax = plt.subplot(133)

for i in range(4):
    data_g = np.load(f"g{i}.npz")
    ax.plot(data_g["x"], data_g["y"], color=colors[0], ls=lss[i], alpha=alphas[i])
ax.set_xlim(0, 5)
ax.set_xticks([0, 2.5, 5])
ax.set_xlabel(r"[ATP]$_0$ (\si{\milli\Molar})")
ax.set_ylim(0, 1)
ax.set_yticks(np.arange(0, 1+0.5, 0.5))
#ax.set_title(r"\textbf{g}")

plt.savefig("efg.svg")
