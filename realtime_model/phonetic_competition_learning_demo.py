#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phonetic_competition_model import *
from run_provenance import digest, write_manifest


def make_train(n, seed):
    rng = np.random.default_rng(seed)
    speakers = make_speaker_offsets(seed + 100, 8)
    rows = []
    skeys = list(speakers)

    cells = [
        (lab, sp, left, right)
        for lab in LABELS
        for sp in skeys
        for left in LEFT_CONTEXTS
        for right in RIGHT_CONTEXTS
    ]

    order = []
    while len(order) < n:
        block = list(cells)
        rng.shuffle(block)
        order.extend(block)

    for lab, sp, left, right in order[:n]:
        x = sample_token(lab, rng, speakers[sp], left, right)
        row = {
            c: float(x[j]) if np.isfinite(x[j]) else np.nan
            for j, c in enumerate(CUES)
        }
        row.update(
            current_sound=lab.split('|')[0],
            rep_env='C_prevocalic',
            speaker=sp,
            prev_spoken_label=left,
            next_spoken_label=right,
        )
        rows.append(row)

    return pd.DataFrame(rows), speakers


def make_trials(n, seed, speakers, group):
    rng = np.random.default_rng(seed)
    labs = ('s|C_prevocalic', 'sh|C_prevocalic') if group == 'fricative' else ('d|C_prevocalic', 't|C_prevocalic')
    skeys = list(speakers)
    vowels = ('a', 'i')
    out = []

    for i in range(n):
        lab = labs[i % 2]
        sp = skeys[(i // 2) % len(skeys)]
        right = vowels[(i // (2 * len(skeys))) % 2]
        x = sample_token(lab, rng, speakers[sp], 'neutral_rate', right)
        arr, boundary, end = arrivals_for(lab, x)
        reveal = arr['F2_onset_kHz']
        out.append(Trial(lab, sp, x, arr, boundary, 'neutral_rate', right, reveal, end))

    return out


def summarize_times(vals):
    a = np.asarray([v for v in vals if v is not None], float)
    return dict(
        n=len(vals),
        committed=len(a),
        rate=len(a) / len(vals) if vals else 0,
        median=float(np.median(a)) if len(a) else None,
        p10=float(np.percentile(a, 10)) if len(a) else None,
        p90=float(np.percentile(a, 90)) if len(a) else None,
    )


def timing_audit(sc, trials, group):
    focal = ('s|C_prevocalic', 'sh|C_prevocalic') if group == 'fricative' else ('d|C_prevocalic', 't|C_prevocalic')
    fields = {'full_field': LABELS, 'focal_binary': focal}
    summary = {'group': group}
    rows = []

    for name, labs in fields.items():
        vals = []
        for i, tr in enumerate(trials):
            r = run_controller(sc, tr, labels=labs, stop_after_first_commit=True)
            vals.append(r['initial_commit_ms'])
            rows.append(dict(
                group=group,
                field=name,
                trial=i,
                truth=tr.truth,
                vowel=tr.right_context,
                speaker=tr.speaker,
                commit_ms=r['initial_commit_ms'],
                commit_label=r['initial_commit_label'],
                correct=int(r['initial_commit_label'] == tr.truth),
            ))

        summary[name] = summarize_times(vals)
        trial_rows = [r for r in rows if r['field'] == name]
        summary[name]['initial_accuracy'] = (
            sum(r['correct'] for r in trial_rows) / len(trial_rows) if trial_rows else 0.0
        )
        summary[name]['by_vowel_accuracy'] = {
            v: sum(r['correct'] for r in trial_rows if r['vowel'] == v)
            / max(1, sum(r['vowel'] == v for r in trial_rows))
            for v in sorted({r['vowel'] for r in trial_rows})
        }

    return summary, rows


def revision_trials(n, seed, speakers, group):
    rng = np.random.default_rng(seed)
    skeys = list(speakers)
    vowels = ('a', 'i')
    out = []
    labs = ('s|C_prevocalic', 'sh|C_prevocalic') if group == 'fricative' else ('d|C_prevocalic', 't|C_prevocalic')

    for i in range(n):
        lab = labs[i % 2]
        sp = skeys[(i // 2) % len(skeys)]
        right = vowels[(i // (2 * len(skeys))) % 2]
        x = sample_token(lab, rng, speakers[sp], 'neutral_rate', right)
        arr, b, end = arrivals_for(lab, x)
        reveal = arr['F2_onset_kHz']
        out.append(Trial(lab, sp, x, arr, b, 'neutral_rate', right, reveal, end))

    return out


def revision_audit(sc, trials, group):
    rows = []
    for i, tr in enumerate(trials):
        r = run_controller(sc, tr)
        init = r['initial_commit_label']
        truth = tr.truth
        final = r['final_label']
        trigger = (
            tr.right_context_reveal_ms if group == 'fricative'
            else tr.arrivals['following_vowel_duration_ms']
        )
        early = bool(init and r['initial_commit_ms'] is not None and r['initial_commit_ms'] < trigger)
        outcome = 'no_early_commit'

        if early:
            if init != truth and r['genuine_revision'] and final == truth:
                outcome = 'wrong_to_correct'
            elif init == truth and r['genuine_revision'] and final != truth:
                outcome = 'correct_to_wrong'
            elif init != truth and r['genuine_revision'] and final != truth:
                outcome = 'wrong_to_other_wrong'
            elif init == truth:
                outcome = 'correct_unchanged'
            else:
                outcome = 'wrong_unchanged'

        rows.append(dict(
            group=group,
            trial=i,
            truth=truth,
            trigger_ms=trigger,
            initial_commit_ms=r['initial_commit_ms'],
            initial_label=init,
            reprocess_ms=r['reprocess_ms'],
            recommit_ms=r['recommit_ms'],
            final_label=final,
            genuine_revision=int(r['genuine_revision']),
            early_committed=int(early),
            outcome=outcome,
        ))

    D = pd.DataFrame(rows)
    E = D[D.early_committed == 1]
    W = E[E.initial_label != E.truth]
    C = E[E.initial_label == E.truth]
    wc = int((E.outcome == 'wrong_to_correct').sum())
    cw = int((E.outcome == 'correct_to_wrong').sum())
    ww = int((E.outcome == 'wrong_to_other_wrong').sum())

    return dict(
        group=group,
        n=len(D),
        early_committed=len(E),
        wrong_early=len(W),
        correct_early=len(C),
        wrong_to_correct=wc,
        correct_to_wrong=cw,
        wrong_to_other_wrong=ww,
        wrong_to_correct_rate=wc / len(W) if len(W) else 0,
        correct_to_wrong_rate=cw / len(C) if len(C) else 0,
    ), D


def recovery(model):
    z = np.load(model, allow_pickle=False)
    labs = [str(x) for x in z['labels']]
    means = z['category_means']
    scale = z['resid_scale']
    rows = []
    details = []

    for a, b in [
        ('s|C_prevocalic', 'sh|C_prevocalic'),
        ('d|C_prevocalic', 't|C_prevocalic'),
    ]:
        ai, bi = labs.index(a), labs.index(b)
        ta, tb = LABELS.index(a), LABELS.index(b)
        idx = np.intersect1d(
            np.flatnonzero(np.isfinite(TRUE_MEANS[ta])),
            np.flatnonzero(np.isfinite(TRUE_MEANS[tb])),
        )
        learned = (means[ai, idx] - means[bi, idx]) * scale[idx]
        truth = TRUE_MEANS[ta, idx] - TRUE_MEANS[tb, idx]
        pooled_sd = np.sqrt((TRUE_SDS[ta, idx] ** 2 + TRUE_SDS[tb, idx] ** 2) / 2)
        std_err = (learned - truth) / pooled_sd
        pair = f'{a} vs {b}'

        rows.append(dict(
            pair=pair,
            standardized_separation_rmse=float(np.sqrt(np.mean(std_err ** 2))),
        ))
        for j, l, t, e in zip(idx, learned, truth, std_err):
            details.append(dict(
                pair=pair,
                cue=CUES[j],
                true_separation=float(t),
                learned_separation=float(l),
                standardized_error=float(e),
            ))

    return rows, details


def focal_context_audit(sc, trials):
    rows = []
    for tr in trials:
        obs = {
            cue: float(tr.raw[CUE_INDEX[cue]])
            for cue in tr.arrivals if np.isfinite(tr.raw[CUE_INDEX[cue]])
        }
        for condition, speaker, right in [
            ('full', tr.speaker, tr.right_context),
            ('no_vowel', tr.speaker, None),
            ('no_speaker', None, tr.right_context),
            ('neither', None, None),
        ]:
            scores = sc.score(obs, speaker=speaker, left=tr.left_context, right=right, labels=LABELS)
            pred = max(scores, key=scores.get)
            rows.append(dict(
                condition=condition,
                truth=tr.truth,
                vowel=tr.right_context,
                speaker=tr.speaker,
                pred=pred,
                correct=int(pred == tr.truth),
                truth_p=float(scores.get(tr.truth, 0.0)),
            ))

    D = pd.DataFrame(rows)
    summary = []
    for cond, g in D.groupby('condition', sort=False):
        summary.append(dict(
            condition=cond,
            n=len(g),
            accuracy=float(g.correct.mean()),
            mean_truth_p=float(g.truth_p.mean()),
        ))
    return summary, D


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', default=str(ROOT / 'demos' / 'output' / 'learning'))
    parser.add_argument('--n-train', type=int, default=6000)
    parser.add_argument('--n-test', type=int, default=1500)
    parser.add_argument('--seed', type=int, default=20260928)
    parser.add_argument(
        '--reuse-model', action='store_true',
        help='Regenerate test simulations from the existing trained model without refitting it',
    )
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    completion = outdir / 'completed_learning_provenance.csv'
    completion.unlink(missing_ok=True)
    model_path = outdir / 'phonetic_full_inventory_model.npz'
    if args.reuse_model:
        if not model_path.is_file():
            raise SystemExit(f'Missing trained model for --reuse-model: {model_path}')
        speakers = make_speaker_offsets(args.seed + 100, 8)
    else:
        training_data, speakers = make_train(args.n_train, args.seed)
        fit_bundle(training_data, model_path, min_label_n=20)
    scorer = Scorer(model_path)

    n_each = max(1, args.n_test // 2)
    fricative_trials = make_trials(n_each, args.seed + 1, speakers, 'fricative')
    stop_trials = make_trials(n_each, args.seed + 2, speakers, 'stop')

    fricative_timing, fricative_rows = timing_audit(scorer, fricative_trials, 'fricative')
    stop_timing, stop_rows = timing_audit(scorer, stop_trials, 'stop')

    fricative_revision_trials = revision_trials(args.n_test, args.seed + 3, speakers, 'fricative')
    stop_revision_trials = revision_trials(args.n_test, args.seed + 4, speakers, 'stop')
    fricative_revisions, fricative_revision_rows = revision_audit(scorer, fricative_revision_trials, 'fricative')
    stop_revisions, stop_revision_rows = revision_audit(scorer, stop_revision_trials, 'stop')

    recovery_summary, recovery_details = recovery(model_path)
    context_summary, context_rows = focal_context_audit(scorer, fricative_trials)

    summary = {
        'timing': [fricative_timing, stop_timing],
        'revisions': [fricative_revisions, stop_revisions],
        'fricative_context_dependence': context_summary,
        'recovery': recovery_summary,
    }
    (outdir / 'phonetic_learning_summary.json').write_text(json.dumps(summary, indent=2))

    pd.DataFrame(fricative_rows + stop_rows).to_csv(outdir / 'commitment_timing_trials.csv', index=False)
    fricative_revision_rows.to_csv(outdir / 'fricative_controller_revision_trials.csv', index=False)
    stop_revision_rows.to_csv(outdir / 'stop_controller_revision_trials.csv', index=False)
    context_rows.to_csv(outdir / 'fricative_context_ablation_trials.csv', index=False)
    pd.DataFrame(context_summary).to_csv(outdir / 'fricative_context_ablation_summary.csv', index=False)
    pd.DataFrame(recovery_summary).to_csv(outdir / 'parameter_recovery.csv', index=False)
    pd.DataFrame(recovery_details).to_csv(outdir / 'parameter_recovery_by_cue.csv', index=False)

    print(json.dumps(summary, indent=2))

    write_manifest(completion, {
        'controller_md5': digest(Path(__file__).with_name('phonetic_competition_model.py')),
        'model_md5': digest(model_path),
        'timing_csv_md5': digest(outdir / 'commitment_timing_trials.csv'),
        'seed': args.seed,
        'n_test': args.n_test,
        'model_reused': int(args.reuse_model),
    })
    print(f'completed learning replay: {completion}', flush=True)


if __name__ == '__main__':
    main()
