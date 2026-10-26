# On Windows, use "python run.py <action>" instead of make.
PYTHON ?= python3
.PHONY: test verify figures paper reuse reproduce

test:
	$(PYTHON) -m unittest discover -s realtime_model -p 'test_*.py' -v

verify: test
	$(PYTHON) run.py verify

figures:
	$(PYTHON) run.py figures

paper:
	$(PYTHON) run.py paper

reuse:
	$(PYTHON) run.py reuse

reproduce:
	$(PYTHON) run.py reproduce
