#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import sys
from dataclasses import fields, replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phonetic_competition_model import (
    CUE_INDEX,
    LABELS,
    Scorer,
    _frication_voicing_sequence,
    arrivals_for,
    make_speaker_offsets,
    run_controller,
)
from phonetic_competition_learning_demo import make_trials
from run_provenance import digest, write_manifest

FOCAL = ("s|C_prevocalic", "sh|C_prevocalic")
SHIFTS = (-50, 0, 50)


def _provenance(model, args):
    expected = Path(__file__).with_name('phonetic_competition_model.py').resolve()
    actual = Path(inspect.getsourcefile(run_controller) or '').resolve()
    if actual != expected:
        raise RuntimeError(f'wrong controller imported: {actual} (expected {expected})')
    return {
        'controller_source': str(actual),
        'controller_sha256': hashlib.sha256(actual.read_bytes()).hexdigest(),
        'audit_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'model_source': str(model.resolve()),
        'model_sha256': hashlib.sha256(model.read_bytes()).hexdigest(),
        'n_test': args.n_test,
        'seed': args.seed,
        'python_executable': sys.executable,
        'commitment_policy': 'online_partial_channel_predictive_robustness_and_provisional_exhaustion',
    }


def _summary(x):
    a = pd.to_numeric(pd.Series(x), errors="coerce").dropna().to_numpy(float)
    n = len(x)
    return {
        "n": int(n),
        "committed": int(len(a)),
        "commit_rate": float(len(a) / n) if n else 0.0,
        "median": float(np.median(a)) if len(a) else None,
        "p10": float(np.percentile(a, 10)) if len(a) else None,
        "p90": float(np.percentile(a, 90)) if len(a) else None,
    }


def _replace_duration(tr, target_boundary_ms):
    raw = np.array(tr.raw, copy=True)
    raw[CUE_INDEX["frication_duration_ms"]] = float(target_boundary_ms)
    arrivals, boundary, end_ms = arrivals_for(tr.truth, raw)

    if int(boundary) != int(target_boundary_ms):
        raise RuntimeError(
            f"requested boundary {target_boundary_ms}, got {boundary}"
        )

    kwargs = {
        "raw": raw,
        "arrivals": arrivals,
        "boundary_ms": int(boundary),
        "right_context_reveal_ms": int(arrivals["F2_onset_kHz"]),
        "end_ms": int(end_ms),
    }

    names = {f.name for f in fields(tr)}
    if "speaker_reveal_ms" in names:
        sr = tr.speaker_reveal_ms
        if sr is not None and sr > 0:
            delta = int(boundary) - int(tr.boundary_ms)
            kwargs["speaker_reveal_ms"] = int(sr) + delta

    return replace(tr, **kwargs)


def _run(sc, tr, condition):
    if condition == "full_model":
        return run_controller(
            sc, tr, labels=LABELS, stop_after_first_commit=True,
            future_robustness=True, organizational_release=True,
        )
    if condition == "full_robustness_only":
        return run_controller(
            sc, tr, labels=LABELS, stop_after_first_commit=True,
            future_robustness=True, organizational_release=False,
        )
    if condition == "full_no_guard":
        return run_controller(
            sc, tr, labels=LABELS, stop_after_first_commit=True,
            future_robustness=False, organizational_release=False,
        )
    if condition == "focal_binary":
        return run_controller(
            sc, tr, labels=FOCAL, stop_after_first_commit=True,
            future_robustness=False, organizational_release=False,
        )
    raise ValueError(condition)

def main():
    ap = argparse.ArgumentParser(
        description=(
            "Audit boundary-relative fricative commitment and matched "
            "-50/0/+50 ms frication-duration manipulations."
        )
    )
    ap.add_argument(
        "--model",
        default=str(ROOT / "demos" / "output" / "learning" / "phonetic_full_inventory_model.npz"),
    )
    ap.add_argument("--n-test", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=20260928)
    ap.add_argument(
        "--outdir",
        default="demos/output/learning/boundary_locking_audit",
    )
    ap.add_argument(
        "--progress-every",
        type=int,
        default=50,
        help="print progress every N base trials; 0 disables progress",
    )
    ap.add_argument(
        "--include-binary-manipulation",
        action="store_true",
        help=(
            "also replay the matched duration manipulation in the focal "
            "/s/-/sh/ binary field; slower but useful as a gating control"
        ),
    )
    args = ap.parse_args()

    model = Path(args.model)
    if not model.is_file():
        raise SystemExit(
            f"ERROR: learned model not found: {model}\n"
            "Run demos/run_all.py first."
        )

    sc = Scorer(model)
    speakers = make_speaker_offsets(args.seed + 100, 8)
    n = max(1, args.n_test // 2)
    trials = make_trials(n, args.seed + 1, speakers, "fricative")

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    # mark prior reports invalid until this specific replay fully completes
    manifest = out / 'completed_run_provenance.json'
    manifest.unlink(missing_ok=True)
    completion = out / 'completed_audit_provenance.csv'
    completion.unlink(missing_ok=True)
    provenance = _provenance(model, args)
    print('[boundary-lock] controller:', provenance['controller_source'], flush=True)
    print('[boundary-lock] controller sha256:', provenance['controller_sha256'], flush=True)
    print('[boundary-lock] learned-model sha256:', provenance['model_sha256'], flush=True)

    # A. Natural-token boundary-relative timing
    base_conditions = ("full_model", "full_robustness_only", "full_no_guard", "focal_binary")
    base_rows = []
    print(
        f"\n[boundary-lock] natural boundary-relative replay: "
        f"{len(trials)} trials x {len(base_conditions)} conditions",
        flush=True,
    )

    for i, tr in enumerate(trials, 1):
        by_condition = {}
        for condition in base_conditions:
            r = _run(sc, tr, condition)
            by_condition[condition] = (r['initial_commit_ms'], r['initial_commit_label'], r['initial_commit_reason'])
            cm = r["initial_commit_ms"]
            delta = (
                float(cm) - float(tr.boundary_ms)
                if cm is not None
                else np.nan
            )
            base_rows.append(
                {
                    "trial": i - 1,
                    "truth": tr.truth,
                    "vowel": tr.right_context,
                    "speaker": tr.speaker,
                    "condition": condition,
                    "boundary_ms": int(tr.boundary_ms),
                    "commit_ms": cm,
                    "commit_label": r["initial_commit_label"],
                    "correct_initial": int(r["initial_commit_label"] == tr.truth),
                    "commit_reason": r["initial_commit_reason"],
                    "commit_minus_boundary_ms": delta,
                    "committed_before_boundary": int(
                        cm is not None and cm < tr.boundary_ms
                    ),
                    "committed_at_or_after_boundary": int(
                        cm is not None and cm >= tr.boundary_ms
                    ),
                }
            )
        if by_condition['full_model'] != by_condition['full_robustness_only']:
            raise AssertionError(f'trial {i}: category commitment still depends on organizational release')
        if args.progress_every and (
            i % args.progress_every == 0 or i == len(trials)
        ):
            print(
                f"[boundary-lock] natural {i}/{len(trials)}",
                flush=True,
            )

    B = pd.DataFrame(base_rows)
    B.to_csv(out / "boundary_relative_trials.csv", index=False)

    bsum = []
    for (condition, truth), g in B.groupby(
        ["condition", "truth"], sort=True
    ):
        q = _summary(g["commit_ms"])
        committed = g[g["commit_ms"].notna()]
        d = committed["commit_minus_boundary_ms"].to_numpy(float)
        q.update(
            condition=condition,
            truth=truth,
            median_boundary_ms=float(g["boundary_ms"].median()),
            median_commit_minus_boundary_ms=(
                float(np.median(d)) if len(d) else None
            ),
            p10_commit_minus_boundary_ms=(
                float(np.percentile(d, 10)) if len(d) else None
            ),
            p90_commit_minus_boundary_ms=(
                float(np.percentile(d, 90)) if len(d) else None
            ),
            before_boundary_rate_among_committed=(
                float(committed["committed_before_boundary"].mean())
                if len(committed)
                else None
            ),
            before_boundary_n=int(committed["committed_before_boundary"].sum()),
            initial_accuracy=float(g['correct_initial'].mean()),
            initial_accuracy_before_a=float(g.loc[g['vowel']=='a', 'correct_initial'].mean()),
            initial_accuracy_before_i=float(g.loc[g['vowel']=='i', 'correct_initial'].mean()),
            high_confidence_n=int((committed["commit_reason"] == 'high_confidence').sum()),
            evidence_exhaustion_n=int((committed["commit_reason"] == 'evidence_exhaustion').sum()),
            within_10ms_of_boundary_rate=(
                float((np.abs(d) <= 10).mean()) if len(d) else None
            ),
            within_20ms_of_boundary_rate=(
                float((np.abs(d) <= 20).mean()) if len(d) else None
            ),
        )
        bsum.append(q)

    BS = pd.DataFrame(bsum)
    BS.to_csv(out / "boundary_relative_summary.csv", index=False)

    # B. Matched duration manipulation.
    eligible = list(enumerate(trials))
    manip_conditions = ["full_model", "full_robustness_only", "full_no_guard"]
    if args.include_binary_manipulation:
        manip_conditions.append("focal_binary")

    print(
        f"\n[boundary-lock] matched duration manipulation: "
        f"{len(eligible)}/{len(trials)} eligible trials; "
        f"{len(SHIFTS)} durations x {len(manip_conditions)} conditions",
        flush=True,
    )

    mrows = []
    for k, (trial_id, tr) in enumerate(eligible, 1):
        base_boundary = int(tr.boundary_ms)

        variants = {
            shift: _replace_duration(tr, base_boundary + shift)
            for shift in SHIFTS
        }

        # verify matched variants have the same early acoustic history
        # re-randomizing the prefix when duration changes is a confound
        prefix = _frication_voicing_sequence(variants[-50], 5)
        for shift, tt in variants.items():
            if not np.array_equal(_frication_voicing_sequence(tt, 5)[:len(prefix)], prefix):
                raise AssertionError(f'trial {trial_id}: duration shift {shift} changed early voicing frames')

        for shift, tt in variants.items():
            expected = base_boundary + shift
            if int(tt.boundary_ms) != expected:
                raise RuntimeError(
                    f"trial {trial_id}: shift {shift}: "
                    f"expected boundary {expected}, got {tt.boundary_ms}"
                )

        by_condition = {}
        for condition in manip_conditions:
            for shift in SHIFTS:
                tt = variants[shift]
                r = _run(sc, tt, condition)
                by_condition[(condition, shift)] = (r['initial_commit_ms'], r['initial_commit_label'], r['initial_commit_reason'])
                cm = r["initial_commit_ms"]
                mrows.append(
                    {
                        "trial": trial_id,
                        "truth": tr.truth,
                        "vowel": tr.right_context,
                        "speaker": tr.speaker,
                        "condition": condition,
                        "duration_shift_ms": shift,
                        "boundary_ms": int(tt.boundary_ms),
                        "commit_ms": cm,
                        "commit_label": r["initial_commit_label"],
                    "correct_initial": int(r["initial_commit_label"] == tr.truth),
                        "commit_reason": r["initial_commit_reason"],
                        "commit_minus_boundary_ms": (
                            float(cm) - float(tt.boundary_ms)
                            if cm is not None
                            else np.nan
                        ),
                    }
                )

        for shift in SHIFTS:
            if by_condition['full_model', shift] != by_condition['full_robustness_only', shift]:
                raise AssertionError(f'trial {trial_id} shift {shift}: release gate still affects commitment')
        if args.progress_every and (
            k % args.progress_every == 0 or k == len(eligible)
        ):
            print(
                f"[boundary-lock] matched {k}/{len(eligible)}",
                flush=True,
            )

    M = pd.DataFrame(mrows)
    M.to_csv(out / "matched_duration_trials.csv", index=False)

    msum = []
    for (condition, truth, shift), g in M.groupby(
        ["condition", "truth", "duration_shift_ms"], sort=True
    ):
        q = _summary(g["commit_ms"])
        committed = g[g["commit_ms"].notna()]
        d = committed["commit_minus_boundary_ms"].to_numpy(float)
        if len(d):
            vals, counts = np.unique(d, return_counts=True)
            mode_share = float(np.max(counts) / len(d))
        else:
            mode_share = None
        q.update(
            condition=condition,
            truth=truth,
            duration_shift_ms=int(shift),
            median_boundary_ms=float(g["boundary_ms"].median()),
            median_commit_minus_boundary_ms=(
                float(np.median(d)) if len(d) else None
            ),
            p10_commit_minus_boundary_ms=(
                float(np.percentile(d, 10)) if len(d) else None
            ),
            p25_commit_minus_boundary_ms=(
                float(np.percentile(d, 25)) if len(d) else None
            ),
            p75_commit_minus_boundary_ms=(
                float(np.percentile(d, 75)) if len(d) else None
            ),
            p90_commit_minus_boundary_ms=(
                float(np.percentile(d, 90)) if len(d) else None
            ),
            sd_commit_minus_boundary_ms=(
                float(np.std(d, ddof=1)) if len(d) > 1 else 0.0
            ),
            distinct_commit_minus_boundary_times=(
                int(len(np.unique(d))) if len(d) else 0
            ),
            modal_commit_minus_boundary_share=mode_share,
        )
        msum.append(q)

    MS = pd.DataFrame(msum)
    MS.to_csv(out / "matched_duration_summary.csv", index=False)

    # paired within-token short-vs-normal-vs-long effects
    paired = []
    for (condition, truth), g in M.groupby(
        ["condition", "truth"], sort=True
    ):
        P = g.pivot(
            index="trial",
            columns="duration_shift_ms",
            values="commit_ms",
        )
        for col in SHIFTS:
            if col not in P:
                P[col] = np.nan

        both_sl = P[-50].notna() & P[50].notna()
        all_three = P[-50].notna() & P[0].notna() & P[50].notna()

        d_sl = (P.loc[both_sl, 50] - P.loc[both_sl, -50]).to_numpy(float)
        d_sn = (P.loc[all_three, 0] - P.loc[all_three, -50]).to_numpy(float)
        d_nl = (P.loc[all_three, 50] - P.loc[all_three, 0]).to_numpy(float)
        exact_boundary_track = np.isclose(d_sl, 100.0, atol=1e-9)
        unchanged_commit_time = np.isclose(d_sl, 0.0, atol=1e-9)

        paired.append(
            {
                "condition": condition,
                "truth": truth,
                "eligible_n": int(len(P)),
                "short_and_long_commit_n": int(both_sl.sum()),
                "all_three_commit_n": int(all_three.sum()),
                "median_long_minus_short_commit_ms": (
                    float(np.median(d_sl)) if len(d_sl) else None
                ),
                "mean_long_minus_short_commit_ms": (
                    float(np.mean(d_sl)) if len(d_sl) else None
                ),
                "p10_long_minus_short_commit_ms": (
                    float(np.percentile(d_sl, 10)) if len(d_sl) else None
                ),
                "p90_long_minus_short_commit_ms": (
                    float(np.percentile(d_sl, 90)) if len(d_sl) else None
                ),
                "median_normal_minus_short_commit_ms": (
                    float(np.median(d_sn)) if len(d_sn) else None
                ),
                "median_long_minus_normal_commit_ms": (
                    float(np.median(d_nl)) if len(d_nl) else None
                ),
                # measure how closely commitment timing follows frication offset
                # a ratio near 1 indicates offset-aligned commitment
                # a ratio near 0 indicates little change in commitment time
                "boundary_tracking_ratio": (
                    float(np.median(d_sl) / 100.0)
                    if len(d_sl)
                    else None
                ),
                "exact_100ms_tracking_n": int(exact_boundary_track.sum()),
                "exact_100ms_tracking_rate": (
                    float(exact_boundary_track.mean()) if len(d_sl) else None
                ),
                "unchanged_commit_time_n": int(unchanged_commit_time.sum()),
                "unchanged_commit_time_rate": (
                    float(unchanged_commit_time.mean()) if len(d_sl) else None
                ),
                "nonexact_tracking_n": int((~exact_boundary_track).sum()),
                "short_only_commit_n": int(
                    (P[-50].notna() & P[50].isna()).sum()
                ),
                "long_only_commit_n": int(
                    (P[-50].isna() & P[50].notna()).sum()
                ),
            }
        )

    PS = pd.DataFrame(paired)
    PS.to_csv(out / "matched_duration_paired.csv", index=False)

    print("\nBOUNDARY-RELATIVE TIMING")
    print(BS.to_string(index=False))

    print("\nMATCHED DURATION CONDITIONS")
    print(MS.to_string(index=False))

    print("\nPAIRED SHORT (-50) vs LONG (+50)")
    print(PS.to_string(index=False))

    result = {
        "n_natural_trials": int(len(trials)),
        "n_duration_manipulation_eligible": int(len(eligible)),
        "eligibility_rule": "all sampled fricative trials",
        "duration_shifts_ms": list(SHIFTS),
        "exact_long_minus_short_boundary_difference_ms": 100,
        "boundary_relative": bsum,
        "matched_duration": msum,
        "paired": paired,
        "provenance": provenance,
    }
    (out / "boundary_locking_summary.json").write_text(
        json.dumps(result, indent=2)
    )
    temp_manifest = out / 'completed_run_provenance.json.tmp'
    temp_manifest.write_text(json.dumps(provenance, indent=2) + '\n')
    temp_manifest.replace(manifest)
    write_manifest(completion, {
        'controller_md5': digest(Path(__file__).with_name('phonetic_competition_model.py')),
        'model_md5': digest(model),
        'matched_csv_md5': digest(out / 'matched_duration_trials.csv'),
        'seed': args.seed,
        'n_test': args.n_test,
    })

    print(f"\nwrote {out} (verified fresh controller execution)")


if __name__ == "__main__":
    main()
