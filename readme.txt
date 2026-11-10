This repository contains the Python codes used to generate the figures for the manuscript
"Accessible Gibbs energy at metabolic activation can constrain steady-state growth".

1. System Requirements
----------------------
- Python 3.8 or higher
- Python packages: numpy, scipy, matplotlib (3.6 or higher), seaborn
- Versions used in the study: Python 3.13.5, numpy 1.26.4, scipy 1.11.4, matplotlib 3.6.3, seaborn 0.13.2
- A LaTeX installation with the packages sfmath and siunitx, and the Helvetica font.
- The figure codes render all text with LaTeX (usetex=True). Without LaTeX they stop with an error.

2. Installation
---------------
    conda create -n figures python=3.9
    conda activate figures
    pip install numpy scipy matplotlib seaborn

3. Generating Figures
---------------------
Each code reads its data files by name from its own folder. Run it from that folder:

    cd main_figures/figure2
    python3 figure2.py

The figures are written to the same folder as PDF (Figure 1 and Figure 3 also as SVG).

Code-to-figure map:

    main_figures/figure1/         Figure 1e-g. Run figure1e.py, figure1f.py and figure1g.py first
                                  (they write e0-e3.npz, f0-f3.npz, g0-g3.npz), then figure1e_g.py.
    main_figures/figure2/         Figure 2
    main_figures/figure3/         Figure 3b-e (data: individual replicates; the code computes mean and SD)
    main_figures/figure4/ArcA/    Figure 4a and Supplementary Figure 6
    main_figures/figure4/ArcB/    Figure 4b-c and Supplementary Figure 7
    main_figures/figure4/ArcC/    Figure 4d
    main_figures/figure4/ArcD/    Figure 4e-f and Supplementary Figure 9
    main_figures/figure5/         Figure 5 and Supplementary Figure 13
    supplementary_figures/        figureS1, figureS2, figureS3, figureS4, figureS5, figureS8, figureS11
    supplementary_figures/ADI_pathway_model/
                                  ADI pathway model (without fitting), used to compute the model curves
                                  of Supplementary Figure 12.

4. Notes
--------
- The kinetic characterization codes (Figure 4) run a global optimization (differential evolution with
  seed 0) and write fit_report_*.txt with the fitted parameters and their uncertainties. The
  uncertainties are interquartile ranges of the parameter values visited by the optimizer, so they
  can differ slightly between machines.
- Supplementary Figure 2 and the Figure 4 fits take the longest to run (Supplementary Figure 2 about 30 minutes).
- The CSV files contain the full measured range. Some figures show a subset.
- The values plotted in every panel are provided in the Source Data file that accompanies the paper.
