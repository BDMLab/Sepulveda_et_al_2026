# Goals shape human confidence

Code and data for:

> Sepulveda, P., Fleming, S. M., Zurita, M., & De Martino, B. (2026). *Goals shape human confidence.* bioRxiv. https://doi.org/10.64898/2026.09.02.748915

Across four experiments we show that confidence is constructed relative to the decision-maker's goal: reversing the task goal (e.g. choosing the option with *least* rather than *most* evidence) reverses the relationship between evidence and confidence, including the positive evidence bias (PEB). We account for this with **GOAL** (Goal-Oriented Asymmetric Likelihood), a Bayesian signal-detection observer in which the variance of the internal belief distributions is asymmetrically allocated to the goal-relevant category, and compare it with an equal-variance SDT model (**EVM**) and a heuristic model using chosen-option evidence only (**HM**).

## Experiments

| Exp | Task | Goal frames | N | Data file |
|---|---|---|---|---|
| 1 | Value-based (food) choice | Like / Dislike | 31 | `Data/Sepulveda2020/DataFoodFramingNotebook_31.csv` |
| 2 | Perceptual (dot numerosity) choice | More / Less | 32 | `Data/Sepulveda2020/DataPerceptualFramingNotebook.csv` |
| 3 | Random-dot motion (online, Prolific) | Frame 1 / 2 | 29 | `Data/SepulvedaNew/RDMFrame_FullProlificData_CoherenceControl_Apr2021.csv` |
| 4 | Auditory (click) discrimination + pupillometry | Frame 1 / 2 | 32 | `Data/SepulvedaNew/Pupil2022_FullData_Dec2022.csv` |

Experiments 1–2 were first reported in Sepulveda et al. (2020, *eLife*). `Data/Sepulveda2020/GlamData*_NoBin.csv` are per-frame trial tables in GLAM format, kept for reference.

## Repository structure

```
Data/
  Sepulveda2020/            Exp 1–2 trial-level data
  SepulvedaNew/             Exp 3–4 trial-level behavioural data
  DataPupil_Exp4/           Exp 4 EyeLink sample report (.txt.xz) and pilot<N>/RESULTS_FILE.txt behaviour files
AnalysisBehavior/
  Figure_exp1-4_HumanBehavior.ipynb   Hierarchical linear models of confidence (Figure 1)
AnalysisModel/
  exp{1-4}_*_NormalPrior.ipynb        GOAL model fits per experiment and goal frame (main text)
  exp{1-4}_*_gammaPrior.ipynb         Same fits with Gamma priors on belief SDs (robustness)
  Figure_exp1-4_ModelFit_NormalPrior.ipynb
                                      Model-fit figures and held-out simulations (Figure 6)
  Figure_exp1-4_ModelFit_GammaPrior.ipynb
                                      Same with Gamma priors (Figure S3)
  functionsSims.py, functionsRegress.py   Shared helpers
  environment_pymc3.yml               Conda environment (PyMC 5)
ModelSimulations/
  MCMC_PEB_NullSymModel_pymc3_Sims.ipynb    EVM simulations (Figure 3)
  MCMC_PEB_AsymVarModel_pymc3_Sims.ipynb    GOAL simulations (Figure 4)
  MCMC_PEB_HeuristicModel_pymc3_Sims.ipynb  HM simulations (Figure 5)
  functionsSims.py
PupilAnalysis/
  PupilAnalysis.ipynb                 Exp 4 pupil preprocessing, Figure 7B–C and pupil statistics
  Permutation_GroupRegression_Results_50perms.csv   Saved output of the permutation analysis
  FIRDeconvolution-master/            Third-party FIR deconvolution package (T. Knapen, MIT licence)
```

## Methods implemented

**Behaviour (`AnalysisBehavior/`).** Hierarchical linear models fitted with R `lme4` through `pymer4`, separately for each goal frame. Confidence is regressed on |Δevidence|, RT, chosen evidence and unchosen evidence (all z-scored within participant), with random slopes for all predictors by participant. Participant-level coefficients are compared within and across frames with t-tests.

**Computational models (`AnalysisModel/`, `ModelSimulations/`).** Models are written in PyMC 5. Fits use NUTS, 4 chains × 4000 draws (1000 tuning steps), fitted separately per experiment and goal frame on even-numbered trials, with held-out simulations on odd-numbered trials. Posterior traces are saved as `.nc` files in `AnalysisModel/`. `ModelSimulations/` generates the qualitative predictions of GOAL, EVM and HM.

**Pupillometry (`PupilAnalysis/`).** Blink interpolation and blink/saccade nuisance regression via FIR deconvolution (Knapen et al., 2016), downsampling to 100 Hz and baseline correction. At each time point, per participant and frame, pupil size is regressed on chosen evidence, unchosen evidence, RT and gaze position. Group-level frame differences are tested by permutation, with Benjamini–Hochberg FDR correction across time points (α = 0.01) and a minimum cluster of 6 consecutive samples (60 ms).



## System requirements

**Operating systems tested:** macOS Tahoe, Version 26.6.2

Versions tested:
- Python 3.11.15
- PyMC 5.28.5 (PyTensor 2.38.3)
- ArviZ 0.23.4
- NumPy 2.4.6
- pandas 3.0.5
- SciPy 1.17.1
- statsmodels 0.14.6
- scikit-learn 1.9.0
- xarray 2026.7.0, netCDF4 1.7.4, h5netcdf 1.8.1 (reading and writing `.nc` posterior traces)
- matplotlib 3.11.1, seaborn 0.13.2
- python-graphviz 0.21 (model graph plots)
- JupyterLab 4.6.3, ipykernel 7.3.0

**Hardware:** No non-standard hardware is required. All analyses run on a standard desktop or laptop computer. The EyeLink eye tracker was used only to collect the data, not to run the code.



## Installation

Model fitting:

```bash
conda env create -f AnalysisModel/environment_pymc3.yml
conda activate peb-model-fit
```

Behavioural analyses also need R (≥ 4.x) with `lme4` and `lmerTest`, plus `pymer4` (0.9) and `polars` in Python. The pupil notebook also needs `lmfit`; it adds the bundled `FIRDeconvolution` package to the path itself.


Typical install time on a normal desktop computer: a couple of minutes (mostly creating the conda environment and downloading packages).



## Running

Run each notebook from its own folder; data are read via relative paths (`../Data/...`). The behaviour notebook imports `functionsSims.py` from `../AnalysisModel`. Run the `AnalysisModel/exp*_{Normal,gamma}Prior.ipynb` notebooks before the corresponding `Figure_exp1-4_ModelFit_*` notebook so the `.nc` posterior traces exist. Figures are saved to `figs/` and `figures/` subfolders.

### Pupil data

`PupilAnalysis/PupilAnalysis.ipynb` expects the EyeLink DataViewer sample report `eye_infoFull_report_fix_blink_100Hz.txt.xz` (xz-compressed, 38 MB; read directly by pandas) and per-participant behavioural files `pilot<N>/RESULTS_FILE.txt` in `Data/DataPupil_Exp4/`. The permutation test takes several hours; its saved output is `PupilAnalysis/Permutation_GroupRegression_Results_50perms.csv`.




**Expected run time:** a couple of minutes on a normal desktop computer for behavior analysis. A model-based demo is also available: `ModelSimulations/MCMC_PEB_AsymVarModel_pymc3_Sims.ipynb` simulates the GOAL model and reproduces Figure 4 in a couple of minutes. Model fitting and pupil analysis can take a couple of hours. 


### License.

The code in this repository is released under the MIT License (see the `LICENSE` file).
You are free to use or adapt it under those terms. If you use or build on these
analyses, we would appreciate it if you cite the paper (see Citation above).

The bundled `PupilAnalysis/FIRDeconvolution-master/` package is third-party code
by T. Knapen and is distributed under its own MIT License, included in that folder.

The data in `Data/` are released under the Creative Commons Attribution 4.0
International License (CC BY 4.0).



## Citation

```bibtex
@article{Sepulveda2026goals,
  title   = {Goals shape human confidence},
  author  = {Sepulveda, Pradyumna and Fleming, Stephen M. and Zurita, M. and De Martino, Benedetto},
  journal = {bioRxiv},
  year    = {2026},
  doi     = {10.64898/2026.09.02.748915}
}
```


