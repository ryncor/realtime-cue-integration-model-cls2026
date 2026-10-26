#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / 'realtime_model'
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from phonetic_competition_model import (
    CUE_INDEX,
    LABELS,
    Scorer,
    Trial,
    make_speaker_offsets,
    run_controller,
)
from phonetic_competition_learning_demo import make_trials

S = 's|C_prevocalic'
SH = 'sh|C_prevocalic'
DT = 5

CUES = [
    ('frication_M1_kHz', 'COG'),
    ('frication_M2_kHz', 'Spread'),
    ('frication_M3_skewness', 'Skewness'),
    ('frication_duration_ms', 'Duration'),
    ('frication_voicing_proportion', 'Voicing'),
    ('F2_onset_kHz', 'F2 onset'),
]



def build_revision_trial() -> Trial:
    from phonetic_competition_learning_demo import make_trials

    seed = 20260928
    speakers = make_speaker_offsets(seed + 100, 8)
    return make_trials(750, seed + 1, speakers, 'fricative')[463]


def display_state(t: int, internal: str, result: dict) -> str:
    if internal == 'WAIT':
        return 'IDLE'
    if internal == 'EVALUATING':
        return 'EVAL'
    if internal == 'REPROCESSING':
        return 'REPROC'
    if internal != 'COMMITTED':
        return internal

    rp = result['reprocess_ms']
    rc = result['recommit_ms']
    lab = result['initial_commit_label']

    if rc is not None and t >= int(rc):
        lab = result['final_label']
    elif rp is not None and t >= int(rp):
        lab = result['initial_commit_label']

    if lab == SH:
        return 'COMMITTED(/ʃ/)'
    if lab == S:
        return 'COMMITTED(/s/)'
    return f'COMMITTED({lab})'


def _raw_z(base: Scorer, cue: str, value: float) -> float:
    """Put heterogeneous cue values on the learned raw z-score scale."""
    j = base.fi[cue]
    scale = max(float(base.raw_scale[j]), 1e-8)
    return (float(value) - float(base.raw_global[j])) / scale


def _online_cue_values(
    base: Scorer,
    call: dict,
    t: int,
    boundary: int,
) -> dict[str, float]:
    out = {short: np.nan for _, short in CUES}

    # exact frication spectral cues already observed by this frame
    if t <= boundary:
        for cue, short in (
            ('frication_M1_kHz', 'COG'),
            ('frication_M2_kHz', 'Spread'),
            ('frication_M3_skewness', 'Skewness'),
        ):
            if cue in call['obs']:
                out[short] = _raw_z(base, cue, call['obs'][cue])

        # duration is right-censored while the fricative is still unfolding
        # the online observable value is the elapsed lower bound
        if 'frication_duration_ms' in call['censored']:
            out['Duration'] = _raw_z(
                base,
                'frication_duration_ms',
                call['censored']['frication_duration_ms'],
            )
        elif 'frication_duration_ms' in call['obs']:
            out['Duration'] = _raw_z(
                base,
                'frication_duration_ms',
                call['obs']['frication_duration_ms'],
            )

        # running frication voicing proportion
        if call['partial_voicing'] is not None:
            k, n = call['partial_voicing']
            if int(n) > 0:
                out['Voicing'] = _raw_z(
                    base,
                    'frication_voicing_proportion',
                    float(k) / float(n),
                )
        elif 'frication_voicing_proportion' in call['obs']:
            out['Voicing'] = _raw_z(
                base,
                'frication_voicing_proportion',
                call['obs']['frication_voicing_proportion'],
            )

    # F2 belongs to the following vowel and starts only when measured
    if 'F2_onset_kHz' in call['obs']:
        out['F2 onset'] = _raw_z(
            base,
            'F2_onset_kHz',
            call['obs']['F2_onset_kHz'],
        )

    return out

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument(
        '--model',
        default=str(
            ROOT / 'demos' / 'output' / 'learning' / 'phonetic_full_inventory_model.npz'
        ),
    )
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    model = Path(args.model)
    if not model.is_file():
        raise SystemExit(f'ERROR: learned model not found: {model}')

    base = Scorer(model)
    tr = build_revision_trial()
    result = run_controller(
        base,
        tr,
        trace=True,
        labels=LABELS,
        future_robustness=True,
        organizational_release=True,
        stop_after_first_commit=False,
        dt=DT,
    )

    if not (
        result['genuine_revision']
        and result['initial_commit_label'] == S
        and result['final_label'] == SH
        and result['reprocess_ms'] is not None
        and result['recommit_ms'] is not None
    ):
        raise SystemExit(
            'ERROR: sampled fricative trial 463 no longer produces '
            '/s/ -> REPROCESSING -> /ʃ/ '
            f'commit={result["initial_commit_label"]}@{result["initial_commit_ms"]}, '
            f'reprocess={result["reprocess_ms"]}, '
            f'recommit={result["recommit_ms"]}, final={result["final_label"]}'
        )

    rows = result['rows']
    times = np.asarray([int(r['t_ms']) for r in rows], dtype=int)

    p_s = np.zeros(len(rows), dtype=float)
    p_sh = np.zeros(len(rows), dtype=float)
    cue_values = {
        short: np.full(len(rows), np.nan, dtype=float)
        for _, short in CUES
    }

    for i, row in enumerate(rows):
        if not str(row['top']):
            continue
        call = {
            'scores': row['trace_scores'],
            'obs': row['trace_obs'],
            'censored': row['trace_censored'],
            'partial_voicing': row['trace_partial_voicing'],
        }
        p_s[i] = float(call['scores'].get(S, 0.0))
        p_sh[i] = float(call['scores'].get(SH, 0.0))

        values = _online_cue_values(
            base,
            call,
            int(times[i]),
            int(tr.boundary_ms),
        )
        for short, value in values.items():
            cue_values[short][i] = value

    states = []
    for t, row in zip(times, rows):
        internal = (
            row['primary_state']
            if not row['handoff']
            else row['secondary_state']
        )
        states.append(display_state(int(t), str(internal), result))

    D = pd.DataFrame(
        {
            't_ms': times,
            'p_s': p_s,
            'p_sh': p_sh,
            'release_support': [r['release_support'] for r in rows],
            'duration_support': [r['duration_support'] for r in rows],
            'vowel_support': [r['vowel_support'] for r in rows],
            'state': states,
            'boundary_ms': int(tr.boundary_ms),
            'f2_observed_ms': int(tr.arrivals['F2_onset_kHz']),
            'speaker_available_ms': int(tr.right_context_reveal_ms),
            'initial_commit_ms': int(result['initial_commit_ms']),
            'reprocess_ms': int(result['reprocess_ms']),
            'recommit_ms': int(result['recommit_ms']),
        }
    )

    for _, short in CUES:
        safe = short.lower().replace(' ', '_')
        D[f'cue_{safe}'] = cue_values[short]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    D.to_csv(out, index=False)

    print(
        'PASS: sampled trial 463 revision trace: '
        f'/s/ commit {result["initial_commit_ms"]} ms -> '
        f'REPROCESSING {result["reprocess_ms"]} ms -> '
        f'/ʃ/ commit {result["recommit_ms"]} ms'
    )
    print(f'PASS: wrote one-tier online cue trace: {out}')

