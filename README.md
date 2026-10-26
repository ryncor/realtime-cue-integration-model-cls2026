# Real-time phonetic cue integration model (CLS 62)

Code, trained model, analysis outputs, and figure-generating scripts for the computational model accompanying *A Computational Model of Real-Time Phonetic Cue Integration*.

The purpose of this project was to provide the groundwork for a computational architecture that operates incrementally, providing predictions analogous to those found in eye-tracking studies on cue-integration. The paper demonstrates this capability though the implementation here (which took inspiration from NAPP and C-CuRE as inspiration) is lacking in terms of accuracy and quantitative replicatability of the target Galle et al. (2019) experiments 5 and 6. Nevertheless, this project shows that this architecture can qualitatively reproduce the findings of Galle et al., thereby serving as a basis for investigating and evaluating real-time cue-integration.

## Requirements

- Python 3.10+ with packages from `requirements.txt` (`numpy`, `pandas`).
- R with the `ggplot2` package to regenerate figures. `grid` comes with R
- For building the paper PDF, a LaTeX installation providing `pdflatex` and
  `bibtex`

## Setup

This is all assumed to be from the repository directory. On Linux, macOS, and in my unique current case of FreeBSD:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Install R and `ggplot2` separately. The Python requirements file only lists Python packages. You will need a TeX distribution if you want to rebuild the paper.

## Run

Using the virtual environment's Python executable (typically `python` or `python3` on Unix-based operating systems with the environment active, or `.\.venv\Scripts\python.exe` on Windows):

```sh
python run.py figures
python run.py paper
python run.py reuse
python run.py reproduce
```

- `figures` regenerates the four PDF figures from existing results and model.
- `paper` builds the LaTeX manuscript without regenerating the model.
- `reuse` reruns the analysis using the existing trained model, regenerates
  figures, and rebuilds the paper without fitting a new model.
- `reproduce` retrains the model and regenerates the results, figures, and
  manuscript. This will overwrite the existing outputs.

`reuse` and `reproduce` can also take `--n-train`, `--n-test`, and `--seed`.
The default values are 6000, 1500, and 20260928 respectively. 

## Directory guide

- `realtime_model/` contains the model and its simulation/audit programs.
- `demos/output/learning/` contains the tracked model and analysis outputs.
- `scripts/` contains rebuild and figure scripts.
- `figures/` and `tables/` hold rendered outputs.
- `realtime_cue_integration_draft.tex` is the paper source.

