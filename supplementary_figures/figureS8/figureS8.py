"""
Figure S8: Carbamoyl phosphate degradation characterization.
================================================================================

This code produces Figure S8 from the manuscript
"Accessible Gibbs energy at metabolic activation can constrain steady-state growth".
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

y0_CP=100e-3 # initial CP concentration [M]

fnam="100mMCP.csv" # file name

data_t_min=np.loadtxt(fnam, delimiter=",", usecols=0)

Data=np.loadtxt(fnam, delimiter=",", usecols=1)
data=Data*y0_CP

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
ax.plot(data_t_min, 1e3*data, color=colors[0])
plt.xlim(0, 800)
plt.xticks(np.arange(0, 800+200, 200))
plt.xlabel(r"Time (\si{\minute})")
plt.ylim(0, 100)
plt.yticks(np.arange(0, 100+20, 20))
plt.ylabel(r"CP concentration (\si{\milli\Molar})")

figure.tight_layout()
plt.savefig("CP_degradation_SI.pdf")
