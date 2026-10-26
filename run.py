#!/usr/bin/env python3
"""Cross-platform entry point for the CLS 2026 model and paper."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def run(*args: object) -> None:
    command = [str(arg) for arg in args]
    print('RUN:', ' '.join(command), flush=True)
    env = os.environ.copy()
    env['PYTHON'] = sys.executable
    try:
        subprocess.run(command, cwd=ROOT, env=env, check=True)
    except FileNotFoundError as exc:
        raise SystemExit(f'Missing program: {command[0]} ({exc})') from exc


def paper() -> None:
    tex = 'realtime_cue_integration_draft.tex'
    run('pdflatex', '-interaction=nonstopmode', '-halt-on-error', tex)
    run('bibtex', 'realtime_cue_integration_draft')
    run('pdflatex', '-interaction=nonstopmode', '-halt-on-error', tex)
    run('pdflatex', '-interaction=nonstopmode', '-halt-on-error', tex)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['verify', 'figures', 'paper', 'reuse', 'reproduce'])
    p.add_argument('--n-train', type=int, default=6000)
    p.add_argument('--n-test', type=int, default=1500)
    p.add_argument('--seed', type=int, default=20260928)
    args = p.parse_args()

    if args.action == 'verify':
        run(sys.executable, 'scripts/check_realtime_provenance.py')
    elif args.action == 'figures':
        run('Rscript', 'scripts/make_realtime_visuals.R')
        run('Rscript', 'scripts/make_control_system_figures.R')
    elif args.action == 'paper':
        paper()
    else:
        command = [sys.executable, '-u', 'scripts/rebuild_realtime_results.py',
                   '--n-train', args.n_train, '--n-test', args.n_test,
                   '--seed', args.seed, '--skip-latex']
        if args.action == 'reuse':
            command.append('--reuse-model')
        run(*command)
        paper()


if __name__ == '__main__':
    main()
