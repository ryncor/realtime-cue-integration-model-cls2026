# No package installation or retraining is implicit in paper/figures.
.PHONY: test verify figures paper reuse reproduce

test:
	python3 -m unittest discover -s realtime_model -p 'test_*.py' -v

verify: test
	python3 scripts/check_realtime_provenance.py

figures:
	Rscript scripts/make_realtime_visuals.R
	Rscript scripts/make_control_system_figures.R

paper:
	pdflatex -interaction=nonstopmode -halt-on-error realtime_cue_integration_draft.tex
	bibtex realtime_cue_integration_draft
	pdflatex -interaction=nonstopmode -halt-on-error realtime_cue_integration_draft.tex
	pdflatex -interaction=nonstopmode -halt-on-error realtime_cue_integration_draft.tex

reuse:
	python3 -u scripts/rebuild_realtime_results.py --reuse-model --n-test 1500 --seed 20260928 --skip-latex
	$(MAKE) paper

reproduce:
	python3 -u scripts/rebuild_realtime_results.py --n-train 6000 --n-test 1500 --seed 20260928 --skip-latex
	$(MAKE) paper
