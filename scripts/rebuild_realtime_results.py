#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def run(*args):
    print('\nRUN:', ' '.join(map(str, args)), flush = True)
    subprocess.run([str(arg) for arg in args], cwd = ROOT, check = True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-train', type = int, default = 6000)
    parser.add_argument('--n-test', type = int, default = 1500)
    parser.add_argument('--seed', type = int, default = 20260928)
    parser.add_argument('--skip-latex', action = 'store_true')
    parser.add_argument('--reuse-model', action = 'store_true',
                        help = 'Regenerate simulations/figures from existing model; do not retrain')
    args = parser.parse_args()

    output = ROOT / 'demos' / 'output' / 'learning'
    model = output / 'phonetic_full_inventory_model.npz'
    demo_cmd = [sys.executable, '-u', 'realtime_model/phonetic_competition_learning_demo.py', '--n-train', args.n_train, '--n-test', args.n_test, '--seed', args.seed, '--outdir', output]
    if args.reuse_model:
        demo_cmd.append('--reuse-model')
    run(*demo_cmd)
    run(sys.executable, '-u', 'realtime_model/boundary_locking_audit.py', '--model', model, '--n-test', args.n_test, '--seed', args.seed, '--include-binary-manipulation', '--outdir', output / 'boundary_locking_audit')
    run('Rscript', 'scripts/make_realtime_visuals.R')
    run('Rscript', 'scripts/make_control_system_figures.R')
    if not args.skip_latex:
        run('latexmk', '-g', '-pdf', '-interaction=nonstopmode', '-halt-on-error', 'realtime_cue_integration_draft.tex')
    print('\nCOMPLETE: regenerated analyses and figures from the revised controller', flush = True)
    print('IMPORTANT: Previously stated numerical results in the manuscript may now be outdated', flush = True)
    print('Read demos/output/learning/phonetic_learning_summary.json and demos/output/learning/boundary_locking_audit/boundary_locking_summary.json before revising the manuscript', flush = True)


if __name__ == '__main__':
    main()
