# Accessible-Gibbs-energy-limits-cell-growth
Python codes to generate the main and supplementary figures of the manuscript "Accessible Gibbs energy at metabolic activation limits long-term cell growth".

Includes steady-state modeling of the coarse-grained cell models (antiport, uniport-ATP, and PTS), kinetic modeling of the arginine deiminase (ADI) pathway in vesicles, and parameter fitting for enzyme characterizations (ArcA, ArcB, ArcC, ArcD).

1. System Requirements
----------------------
- Python 3.8 or higher
- Required packages:
    numpy
    scipy
    matplotlib
    seaborn

2. Installation
---------------
It is recommended to create a dedicated environment:

    conda create -n figures python=3.9
    conda activate figures
    pip install numpy scipy matplotlib seaborn

Alternatively, using pip only:

    pip install numpy scipy matplotlib seaborn

3. Generating Figures
---------------------
All codes can be run independently from the repository root directory.

Example:

    python3 main_figures/figure2/figure2.py

Each code generates the corresponding PDF figure file(s) in its own folder.

4. Notes
--------
- Some codes require accompanying data files located in the same directory.
- Figures are exported as vector PDF files.
