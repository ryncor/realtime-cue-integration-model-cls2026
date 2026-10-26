#!/usr/bin/env python3
from __future__ import annotations
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence
import numpy as np
import pandas as pd
LABELS = (
    'f|C_prevocalic',
    'v|C_prevocalic',
    'th|C_prevocalic',
    'dh|C_prevocalic',
    's|C_prevocalic',
    'z|C_prevocalic',
    'sh|C_prevocalic',
    'zh|C_prevocalic',
    'p|C_prevocalic',
    'b|C_prevocalic',
    't|C_prevocalic',
    'd|C_prevocalic',
    'k|C_prevocalic',
    'g|C_prevocalic',
)
FRIC = tuple(x for x in LABELS if x.split('|')[0] in {'f', 'v', 'th', 'dh', 's', 'z', 'sh', 'zh'})
STOP = tuple(x for x in LABELS if x not in FRIC)
DISPLAY = {
    lab: '/' + lab.split('|', 1)[0].replace('sh', 'ʃ').replace('zh', 'ʒ').replace('th', 'θ').replace('dh', 'ð') + '/'
    for lab in LABELS
}
CUES = (
    'frication_M1_kHz',
    'frication_M2_kHz',
    'frication_M3_skewness',
    'frication_duration_ms',
    'frication_voicing_proportion',
    'VOT_ms',
    'vowel_onset_f0_ST',
    'following_vowel_duration_ms',
    'burst_M1_kHz',
    'F2_onset_kHz',
)
CUE_INDEX = {c: i for i, c in enumerate(CUES)}
FRIC_CUES = (
    'frication_M1_kHz',
    'frication_M2_kHz',
    'frication_M3_skewness',
    'frication_duration_ms',
    'frication_voicing_proportion',
    'F2_onset_kHz',
)
STOP_CUES = (
    'VOT_ms',
    'vowel_onset_f0_ST',
    'following_vowel_duration_ms',
    'burst_M1_kHz',
    'F2_onset_kHz',
)
# hand-specified synthetic cue distributions with phonetic directions and overlap; each tuple is (mean, SD)
FRIC_DISTS = {
    # M1, M2, M3 skewness, duration, voicing proportion, raw F2 onset (kHz)
    'f': ((4.5, 1.3), (2.2, 0.6), (0.0, 0.7), (120.0, 22.0), (0.06, 0.08), (1.55, 0.18)),
    'v': ((4.4, 1.3), (2.2, 0.6), (0.0, 0.7), (90.0, 20.0), (0.55, 0.2), (1.55, 0.18)),
    'th': ((5.0, 1.3), (2.35, 0.65), (0.05, 0.7), (115.0, 22.0), (0.06, 0.08), (1.65, 0.18)),
    'dh': ((4.9, 1.3), (2.35, 0.65), (0.05, 0.7), (85.0, 20.0), (0.55, 0.2), (1.65, 0.18)),
    's': ((7.0, 1.05), (1.3, 0.45), (-0.2, 0.7), (150.0, 22.0), (0.05, 0.07), (1.78, 0.18)),
    'z': ((6.8, 1.05), (1.35, 0.45), (-0.15, 0.75), (110.0, 20.0), (0.5, 0.2), (1.78, 0.18)),
    'sh': ((5.4, 1.0), (1.65, 0.5), (1.25, 0.55), (155.0, 22.0), (0.05, 0.07), (1.7, 0.18)),
    'zh': ((5.2, 1.0), (1.7, 0.5), (1.1, 0.6), (115.0, 20.0), (0.5, 0.2), (1.7, 0.18)),
}
STOP_DISTS = {
    # VOT, onset f0 (speaker-normalized semitones), vowel duration, burst M1, raw F2 onset (kHz)
    'p': ((55.0, 12.0), (1.2, 0.9), (170.0, 44.0), (1.7, 0.8), (1.25, 0.18)),
    'b': ((8.0, 7.0), (-0.8, 0.9), (188.0, 45.0), (1.7, 0.8), (1.25, 0.18)),
    't': ((65.0, 12.0), (1.2, 0.9), (170.0, 44.0), (4.5, 0.9), (1.5, 0.18)),
    'd': ((10.0, 7.0), (-0.8, 0.9), (188.0, 45.0), (4.5, 0.9), (1.5, 0.18)),
    'k': ((75.0, 14.0), (1.2, 0.9), (170.0, 44.0), (3.0, 0.85), (1.72, 0.2)),
    'g': ((18.0, 9.0), (-0.8, 0.9), (188.0, 45.0), (3.0, 0.85), (1.72, 0.2)),
}
SUPPORT = {}
for lab in LABELS:
    names = FRIC_CUES if lab in FRIC else STOP_CUES
    SUPPORT[lab] = tuple(CUE_INDEX[c] for c in names)


def true_mean_sd(label: str) -> tuple[np.ndarray, np.ndarray]:
    snd = label.split('|')[0]
    mu = np.full(len(CUES), np.nan, float)
    sd = np.full(len(CUES), np.nan, float)
    spec = FRIC_DISTS[snd] if label in FRIC else STOP_DISTS[snd]
    names = FRIC_CUES if label in FRIC else STOP_CUES
    for cue, (m, sig) in zip(names, spec):
        j = CUE_INDEX[cue]
        mu[j] = m
        sd[j] = sig
    return (mu, sd)
TRUE_MEANS = np.vstack([true_mean_sd(l)[0] for l in LABELS])
TRUE_SDS = np.vstack([true_mean_sd(l)[1] for l in LABELS])
# category-blind contextual shifts in raw cue units
LEFT_CONTEXTS = ('neutral_rate', 'fast_rate', 'slow_rate')
RIGHT_CONTEXTS = ('a', 'i')
LEFT_OFFSETS = {
    'neutral_rate': np.zeros(len(CUES)),
    'fast_rate': np.array([0, 0, 0, -12, 0, -7, 0, -15, 0, 0], float),
    'slow_rate': np.array([0, 0, 0, 12, 0, 7, 0, 15, 0, 0], float),
}
# category-blind vowel effects for C-CuRE: M1/M2 have almost no shared effect, while F2 onset differs substantially between /a/ and /i/
RIGHT_OFFSETS = {
    'a': np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, -0.52], float),
    'i': np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0.52], float),
}
# category-specific vowel interactions omitted by the category-blind normalizer; symmetric /a/ and /i/ shifts preserve the marginal category means; values are (M1, M2, F2) shifts in kHz
FRICATIVE_VOWEL_INTERACTIONS = {
    'f': {'a': (-0.05, 0.0, 0.0), 'i': (0.05, 0.0, 0.0)},
    'v': {'a': (-0.05, 0.0, 0.0), 'i': (0.05, 0.0, 0.0)},
    'th': {'a': (-0.04, 0.01, 0.0), 'i': (0.04, -0.01, 0.0)},
    'dh': {'a': (-0.04, 0.01, 0.0), 'i': (0.04, -0.01, 0.0)},
    # adult English sibilant production has a vowel influence on /s/, with higher spectral center of gravity before /i/ than /a/
    's': {'a': (-0.3, 0.03, -0.03), 'i': (0.3, -0.03, 0.03)},
    'z': {'a': (-0.24, 0.02, -0.03), 'i': (0.24, -0.02, 0.03)},
    # adult /sh/ is more resistant to vowel coarticulation
    'sh': {'a': (-0.04, 0.01, 0.01), 'i': (0.04, -0.01, -0.01)},
    'zh': {'a': (-0.03, 0.01, 0.01), 'i': (0.03, -0.01, -0.01)},
}


@dataclass(frozen = True)
class SpeakerProfile:
    spectral_scale: float
    additive: np.ndarray

    def __getitem__(self, key):
        return self.additive[key]


def make_speaker_offsets(seed: int, n: int = 8) -> dict[str, SpeakerProfile]:
    rng = np.random.default_rng(seed)
    # a balanced +/-10% spectral-scale range shifts high-frequency /s/ more in Hz than /sh/, creating category-by-speaker structure without category-specific normalization
    if n == 8:
        scales = np.asarray([0.9, 0.93, 0.96, 0.98, 1.02, 1.04, 1.07, 1.1], float)
    else:
        scales = np.linspace(0.9, 1.1, n, dtype = float)
    rng.shuffle(scales)
    out = {}
    for i, scale in enumerate(scales):
        tempo, _level, voice = rng.normal(size = 3)
        add = np.zeros(len(CUES), float)
        # speaker/rate tendencies on temporal cues
        add[CUE_INDEX['frication_duration_ms']] = 6.0 * tempo
        add[CUE_INDEX['frication_voicing_proportion']] = 0.02 * voice
        add[CUE_INDEX['VOT_ms']] = 3.0 * tempo
        add[CUE_INDEX['following_vowel_duration_ms']] = 8.0 * tempo
        # onset f0 is already normalized by speaker in semitones
        out[f'spk{i}'] = SpeakerProfile(float(scale), add)
    return out


def _category_vowel_offset(label: str, right: str | None) -> np.ndarray:
    out = np.zeros(len(CUES), float)
    if right not in RIGHT_CONTEXTS or label not in FRIC:
        return out
    snd = label.split('|', 1)[0]
    vals = FRICATIVE_VOWEL_INTERACTIONS.get(snd, {}).get(str(right))
    if vals is None:
        return out
    dm1, dm2, df2 = vals
    out[CUE_INDEX['frication_M1_kHz']] = dm1
    out[CUE_INDEX['frication_M2_kHz']] = dm2
    out[CUE_INDEX['F2_onset_kHz']] = df2
    return out


def _apply_speaker_profile(x: np.ndarray, profile: SpeakerProfile) -> np.ndarray:
    y = x.copy()
    for cue in ('frication_M1_kHz', 'frication_M2_kHz', 'burst_M1_kHz', 'F2_onset_kHz'):
        index = CUE_INDEX[cue]
        if np.isfinite(y[index]):
            y[index] *= profile.spectral_scale
    finite = np.isfinite(y)
    y[finite] += profile.additive[finite]
    return y


def production_center(label: str, speaker_profile, left: str = 'neutral_rate', right: str = 'a') -> np.ndarray:
    # context-conditioned raw acoustic center before token noise
    mu, _ = true_mean_sd(label)
    x = mu.copy()
    support = np.asarray(SUPPORT[label], int)
    x[support] += LEFT_OFFSETS[left][support]
    x[support] += RIGHT_OFFSETS[right][support]
    x[support] += _category_vowel_offset(label, right)[support]
    x = _apply_speaker_profile(x, speaker_profile)
    return _clip_cues(x)


def _clip_cues(x: np.ndarray) -> np.ndarray:
    y = x.copy()
    # physically bounded proportion
    j = CUE_INDEX['frication_voicing_proportion']
    if np.isfinite(y[j]):
        y[j] = float(np.clip(y[j], 0, 1))
    # clip applicable durations and spectral values to positive values
    for cue in [
        'frication_M1_kHz',
        'frication_M2_kHz',
        'frication_duration_ms',
        'following_vowel_duration_ms',
        'burst_M1_kHz',
    ]:
        j = CUE_INDEX[cue]
        if np.isfinite(y[j]):
            y[j] = max(0.001, float(y[j]))
    return y


def sample_token(
    label: str,
    rng: np.random.Generator,
    speaker_offset,
    left: str = 'neutral_rate',
    right: str = 'a',
) -> np.ndarray:
    # sample from the context-conditioned production distribution
    mu, sd = true_mean_sd(label)
    center = production_center(label, speaker_offset, left, right)
    x = np.full(len(CUES), np.nan, float)
    support = np.asarray(SUPPORT[label], int)
    local_sd = sd.copy()
    if isinstance(speaker_offset, SpeakerProfile):
        for cue in ('frication_M1_kHz', 'frication_M2_kHz', 'burst_M1_kHz', 'F2_onset_kHz'):
            j = CUE_INDEX[cue]
            if np.isfinite(local_sd[j]):
                local_sd[j] *= speaker_offset.spectral_scale
    x[support] = rng.normal(center[support], local_sd[support])
    return _clip_cues(x)


def _nanmean(X: np.ndarray) -> np.ndarray:
    finite = np.isfinite(X)
    n = finite.sum(0)
    return np.divide(np.where(finite, X, 0).sum(0), n, out = np.zeros(X.shape[1]), where = n > 0)


def _group_offsets(keys: np.ndarray, residual: np.ndarray, shrink: float) -> tuple[np.ndarray, np.ndarray]:
    uniq = sorted({str(k) for k in keys if str(k) not in {'', 'nan', 'None'}})
    offs = np.zeros((len(uniq), residual.shape[1]))
    for i, k in enumerate(uniq):
        R = residual[keys == k]
        finite = np.isfinite(R)
        n = finite.sum(0).astype(float)
        mean = np.divide(np.where(finite, R, 0).sum(0), n, out = np.zeros(R.shape[1]), where = n > 0)
        offs[i] = mean * (n / (n + shrink))
    return (np.asarray(uniq, str), offs)


def _apply_offsets(keys: np.ndarray, uniq: np.ndarray, offs: np.ndarray, d: int) -> np.ndarray:
    out = np.zeros((len(keys), d), float)
    lut = {str(k): i for i, k in enumerate(uniq)}
    for r, k in enumerate(keys):
        i = lut.get(str(k))
        if i is not None:
            out[r] = offs[i]
    return out


def _cov(X: np.ndarray, ridge: float = 1e-05) -> np.ndarray:
    if len(X) <= 1:
        return np.eye(X.shape[1])
    C = np.cov(X, rowvar = False, ddof = 1)
    if np.ndim(C) == 0:
        C = np.asarray([[float(C)]])
    C = np.asarray(C, float)
    C[~np.isfinite(C)] = 0
    C = (C + C.T) * 0.5
    C += np.eye(C.shape[0]) * ridge
    return C


def fit_category_gaussians(
    standardized,
    cue_observations,
    *,
    labels,
    right_contexts,
    category_index,
    num_categories,
    num_cues,
    context_shrinkage,
    class_cov_prior,
    cov_shrink_diag,
    ridge,
    right_interactions = True,
):
    supports = np.zeros((num_categories, num_cues), dtype = bool)
    preliminary_means = np.zeros((num_categories, num_cues))
    for label, index in category_index.items():
        rows = labels == label
        availability = np.isfinite(cue_observations[rows]).mean(axis = 0)
        cue_indices = np.flatnonzero(availability > 0.5)
        supports[index, cue_indices] = True
        values = standardized[rows][:, cue_indices]
        values = values[np.isfinite(values).all(axis = 1)]
        preliminary_means[index, cue_indices] = values.mean(axis = 0)
    # estimate category-specific effects of the following vowel
    right_keys = np.asarray(sorted({str(value) for value in right_contexts if str(value) not in {'', 'nan', 'None'}}), dtype = str)
    right_offsets = np.zeros((num_categories, len(right_keys), num_cues), dtype = float)
    if right_interactions:
        for label, index in category_index.items():
            cue_indices = np.flatnonzero(supports[index])
            for right_index, right_value in enumerate(right_keys):
                rows = (labels == label) & (right_contexts == right_value)
                if not rows.any():
                    continue
                values = standardized[rows][:, cue_indices]
                finite = np.isfinite(values)
                counts = finite.sum(axis = 0).astype(float)
                context_mean = np.divide(
                    np.where(finite, values, 0).sum(axis = 0),
                    counts,
                    out = np.zeros(len(cue_indices)),
                    where = counts > 0,
                )
                difference = context_mean - preliminary_means[index, cue_indices]
                shrinkage = counts / (counts + context_shrinkage)
                right_offsets[index, right_index, cue_indices] = difference * shrinkage
    # separate category-specific vowel effects from intrinsic category variability
    context_adjustments = np.zeros_like(standardized)
    right_index = {str(value): index for index, value in enumerate(right_keys)}
    for row, (label, right_value) in enumerate(zip(labels, right_contexts)):
        index = right_index.get(str(right_value))
        if index is not None:
            context_adjustments[row] = right_offsets[category_index[str(label)], index]
    adjusted = standardized - context_adjustments
    # estimate a pooled covariance for each manner of articulation
    pooled_covariances = {}
    for manner, manner_labels in (('fric', set(FRIC)), ('stop', set(STOP))):
        rows = np.asarray([label in manner_labels for label in labels])
        if not rows.any():
            continue
        availability = np.isfinite(cue_observations[rows]).mean(axis = 0)
        cue_indices = np.flatnonzero(availability > 0.5)
        values = adjusted[rows][:, cue_indices]
        values = values[np.isfinite(values).all(axis = 1)]
        pooled_covariances[manner] = (cue_indices, _cov(values, ridge))
    presence = np.zeros((num_categories, num_cues))
    means = np.zeros((num_categories, num_cues))
    covariances = np.zeros((num_categories, num_cues, num_cues))
    category_counts = np.zeros(num_categories, dtype = int)
    for label, index in category_index.items():
        rows = labels == label
        category_counts[index] = rows.sum()
        availability = np.isfinite(cue_observations[rows]).mean(axis = 0)
        presence[index] = (availability * category_counts[index] + 1) / (category_counts[index] + 2)
        cue_indices = np.flatnonzero(availability > 0.5)
        supports[index, cue_indices] = True
        values = adjusted[rows][:, cue_indices]
        values = values[np.isfinite(values).all(axis = 1)]
        category_mean = values.mean(axis = 0)
        sample_covariance = _cov(values, ridge)
        manner = 'fric' if label in FRIC else 'stop'
        pooled_indices, pooled_covariance = pooled_covariances[manner]
        if not np.array_equal(cue_indices, pooled_indices):
            raise RuntimeError('unexpected support mismatch')
        weight = len(values) / (len(values) + class_cov_prior)
        covariance = weight * sample_covariance + (1 - weight) * pooled_covariance
        diagonal = np.diag(np.diag(covariance))
        covariance = (1 - cov_shrink_diag) * covariance + cov_shrink_diag * diagonal
        covariance += np.eye(len(cue_indices)) * ridge
        means[index, cue_indices] = category_mean
        covariances[index][np.ix_(cue_indices, cue_indices)] = covariance
    return (presence, supports, means, covariances, category_counts, right_keys, right_offsets)


def fit_bundle(
    df: pd.DataFrame,
    out_path: str | Path,
    *,
    context_shrinkage: float = 20,
    class_cov_prior: float = 32,
    cov_shrink_diag: float = 0.1,
    ridge: float = 1e-05,
    min_label_n: int = 8,
) -> None:
    data = df.copy().reset_index(drop = True)
    observations = data[list(CUES)].apply(pd.to_numeric, errors = 'coerce').to_numpy(float)
    labels = (data['current_sound'].astype(str).str.strip() + '|' + data['rep_env'].astype(str).str.strip()).to_numpy(str)
    label_counts = pd.Series(labels).value_counts()
    kept_labels = sorted(label_counts[label_counts >= min_label_n].index.astype(str))
    keep_rows = np.asarray([label in kept_labels for label in labels])
    data = data.loc[keep_rows].reset_index(drop = True)
    observations = observations[keep_rows]
    labels = labels[keep_rows]
    speakers = data['speaker'].fillna('').astype(str).to_numpy()
    left_contexts = data['prev_spoken_label'].fillna('').astype(str).to_numpy()
    right_contexts = data['next_spoken_label'].fillna('').astype(str).to_numpy()
    num_cues = len(CUES)
    num_categories = len(kept_labels)
    category_index = {label: index for index, label in enumerate(kept_labels)}
    # fit category distributions before speaker identity is available
    raw_global = _nanmean(observations)
    raw_scale = np.sqrt(np.maximum(_nanmean((observations - raw_global) ** 2), 1e-08))
    raw_scale[~np.isfinite(raw_scale) | (raw_scale < 0.0001)] = 1
    raw_standardized = (observations - raw_global) / raw_scale
    raw_present, raw_support, raw_means, raw_covs, raw_ns, raw_crk, raw_cro = fit_category_gaussians(
        raw_standardized,
        observations,
        labels = labels,
        right_contexts = right_contexts,
        category_index = category_index,
        num_categories = num_categories,
        num_cues = num_cues,
        context_shrinkage = context_shrinkage,
        class_cov_prior = class_cov_prior,
        cov_shrink_diag = cov_shrink_diag,
        ridge = ridge,
        right_interactions = True,
    )
    # estimate multiplicative spectral scaling for each speaker
    spectral_cues = ('frication_M1_kHz', 'frication_M2_kHz', 'burst_M1_kHz', 'F2_onset_kHz')
    spectral_indices = np.asarray([CUE_INDEX[cue] for cue in spectral_cues if cue in CUE_INDEX], dtype = int)
    speaker_scale_keys = np.asarray(sorted({str(speaker) for speaker in speakers if str(speaker) not in {'', 'nan', 'None'}}), dtype = str)
    speaker_scales = np.ones(len(speaker_scale_keys), dtype = float)
    raw_cue_mean = _nanmean(observations)
    for speaker_index, speaker in enumerate(speaker_scale_keys):
        rows = speakers == speaker
        log_scales = []
        weights = []
        for cue_index in spectral_indices:
            values = observations[rows, cue_index]
            finite = np.isfinite(values)
            if finite.sum() < 2 or not np.isfinite(raw_cue_mean[cue_index]) or raw_cue_mean[cue_index] <= 0:
                continue
            speaker_mean = float(np.nanmean(values[finite]))
            population_mean = float(raw_cue_mean[cue_index])
            if speaker_mean <= 0 or population_mean <= 0:
                continue
            log_scales.append(math.log(speaker_mean / population_mean))
            weights.append(float(finite.sum()))
        if log_scales:
            weights = np.asarray(weights, dtype = float)
            estimated_log_scale = float(np.average(np.asarray(log_scales), weights = weights))
            count = float(np.max(weights))
            estimated_log_scale *= count / (count + context_shrinkage)
            speaker_scales[speaker_index] = math.exp(estimated_log_scale)
    scale_by_speaker = {str(speaker): speaker_scales[index] for index, speaker in enumerate(speaker_scale_keys)}
    normalized_observations = observations.copy()
    for row, speaker in enumerate(speakers):
        speaker_scale = float(scale_by_speaker.get(str(speaker), 1.0))
        for cue_index in spectral_indices:
            if np.isfinite(normalized_observations[row, cue_index]):
                normalized_observations[row, cue_index] /= speaker_scale
    # estimate additive speaker and context offsets
    global_mean = _nanmean(normalized_observations)
    centered = normalized_observations - global_mean
    speaker_keys = np.asarray([], dtype = str)
    speaker_offsets = np.zeros((0, num_cues))
    left_keys = np.asarray([], dtype = str)
    left_offsets = np.zeros((0, num_cues))
    right_keys = np.asarray([], dtype = str)
    right_offsets = np.zeros((0, num_cues))
    for _ in range(5):
        speaker_effect = _apply_offsets(speakers, speaker_keys, speaker_offsets, num_cues)
        left_effect = _apply_offsets(left_contexts, left_keys, left_offsets, num_cues)
        right_effect = _apply_offsets(right_contexts, right_keys, right_offsets, num_cues)
        speaker_keys, speaker_offsets = _group_offsets(speakers, centered - left_effect - right_effect, context_shrinkage)
        speaker_effect = _apply_offsets(speakers, speaker_keys, speaker_offsets, num_cues)
        left_keys, left_offsets = _group_offsets(
            left_contexts,
            centered - speaker_effect - right_effect,
            context_shrinkage,
        )
        left_effect = _apply_offsets(left_contexts, left_keys, left_offsets, num_cues)
        right_keys, right_offsets = _group_offsets(
            right_contexts,
            centered - speaker_effect - left_effect,
            context_shrinkage,
        )
    speaker_effect = _apply_offsets(speakers, speaker_keys, speaker_offsets, num_cues)
    left_effect = _apply_offsets(left_contexts, left_keys, left_offsets, num_cues)
    right_effect = _apply_offsets(right_contexts, right_keys, right_offsets, num_cues)
    residuals = normalized_observations - (global_mean + speaker_effect + left_effect + right_effect)
    residual_mean = _nanmean(residuals)
    residual_scale = np.sqrt(np.maximum(_nanmean((residuals - residual_mean) ** 2), 1e-08))
    residual_scale[~np.isfinite(residual_scale) | (residual_scale < 0.0001)] = 1
    standardized_residuals = residuals / residual_scale
    present, supports, means, covariances, category_counts, category_right_keys, category_right_offsets = fit_category_gaussians(
        standardized_residuals,
        normalized_observations,
        labels = labels,
        right_contexts = right_contexts,
        category_index = category_index,
        num_categories = num_categories,
        num_cues = num_cues,
        context_shrinkage = context_shrinkage,
        class_cov_prior = class_cov_prior,
        cov_shrink_diag = cov_shrink_diag,
        ridge = ridge,
        right_interactions = True,
    )
    priors = np.full(num_categories, 1 / num_categories)
    # estimate following-vowel probabilities from F2 onset
    vowel_keys = np.asarray([vowel for vowel in ('a', 'i') if np.any(right_contexts == vowel)], dtype = str)
    f2_index = CUE_INDEX['F2_onset_kHz']
    vowel_f2_raw_mean = np.zeros(len(vowel_keys), dtype = float)
    vowel_f2_raw_var = np.ones(len(vowel_keys), dtype = float)
    vowel_f2_norm_mean = np.zeros(len(vowel_keys), dtype = float)
    vowel_f2_norm_var = np.ones(len(vowel_keys), dtype = float)
    vowel_priors = np.zeros(len(vowel_keys), dtype = float)
    for vowel_index, vowel in enumerate(vowel_keys):
        rows = right_contexts == vowel
        values = observations[rows, f2_index]
        finite = np.isfinite(values)
        vowel_priors[vowel_index] = float(finite.sum())
        if finite.any():
            valid_values = np.asarray(values[finite], dtype = float)
            vowel_f2_raw_mean[vowel_index] = float(valid_values.mean())
            vowel_f2_raw_var[vowel_index] = float(max(valid_values.var(ddof = 1) if len(valid_values) > 1 else 0.04, 0.0001))
        normalized_values = []
        for row in np.flatnonzero(rows):
            value = observations[row, f2_index]
            if not np.isfinite(value):
                continue
            speaker_scale = float(scale_by_speaker.get(str(speakers[row]), 1.0))
            normalized_values.append(float(value) / speaker_scale)
        if normalized_values:
            normalized_values = np.asarray(normalized_values, dtype = float)
            vowel_f2_norm_mean[vowel_index] = float(normalized_values.mean())
            vowel_f2_norm_var[vowel_index] = float(max(normalized_values.var(ddof = 1) if len(normalized_values) > 1 else 0.04, 0.0001))
    if vowel_priors.sum() > 0:
        vowel_priors /= vowel_priors.sum()
    else:
        vowel_priors[:] = 1 / max(len(vowel_priors), 1)

    np.savez_compressed(
        out_path,
        version = np.asarray([4]),
        features = np.asarray(CUES),
        labels = np.asarray(kept_labels),
        raw_global_mean = raw_global,
        raw_resid_scale = raw_scale,
        raw_category_means = raw_means,
        raw_category_covs = raw_covs,
        raw_category_support = raw_support.astype(np.uint8),
        raw_cue_presence = raw_present,
        raw_category_right_keys = raw_crk,
        raw_category_right_offsets = raw_cro,
        global_mean = global_mean,
        resid_scale = residual_scale,
        speaker_keys = speaker_keys,
        speaker_offsets = speaker_offsets,
        left_keys = left_keys,
        left_offsets = left_offsets,
        right_keys = right_keys,
        right_offsets = right_offsets,
        speaker_scale_keys = speaker_scale_keys,
        speaker_spectral_scales = speaker_scales,
        spectral_scale_indices = spectral_indices,
        category_right_keys = category_right_keys,
        category_right_offsets = category_right_offsets,
        category_means = means,
        category_covs = covariances,
        category_support = supports.astype(np.uint8),
        cue_presence = present,
        priors = priors,
        category_n = category_counts,
        vowel_keys = vowel_keys,
        vowel_priors = vowel_priors,
        vowel_f2_raw_mean = vowel_f2_raw_mean,
        vowel_f2_raw_var = vowel_f2_raw_var,
        vowel_f2_norm_mean = vowel_f2_norm_mean,
        vowel_f2_norm_var = vowel_f2_norm_var,
    )


class Scorer:

    def __init__(self, path: str | Path):
        z = np.load(path, allow_pickle = False)
        self.features = [str(x) for x in z['features']]
        self.labels = [str(x) for x in z['labels']]
        self.priors = z['priors']
        self.fi = {f: i for i, f in enumerate(self.features)}
        self.global_mean = z['global_mean']
        self.scale = z['resid_scale']
        self.means = z['category_means']
        self.covs = z['category_covs']
        self.support = z['category_support'].astype(bool)
        self.presence = z['cue_presence']
        self.raw_global = z['raw_global_mean']
        self.raw_scale = z['raw_resid_scale']
        self.raw_means = z['raw_category_means']
        self.raw_covs = z['raw_category_covs']
        self.raw_support = z['raw_category_support'].astype(bool)
        self.raw_presence = z['raw_cue_presence']
        self.sk = [str(x) for x in z['speaker_keys']]
        self.so = z['speaker_offsets']
        self.lk = [str(x) for x in z['left_keys']]
        self.lo = z['left_offsets']
        self.rk = [str(x) for x in z['right_keys']]
        self.ro = z['right_offsets']
        self.ssk = [str(x) for x in z['speaker_scale_keys']]
        self.ssc = z['speaker_spectral_scales']
        self.spectral_idx = set((int(x) for x in z['spectral_scale_indices']))
        self.crk = [str(x) for x in z['category_right_keys']]
        self.cro = z['category_right_offsets']
        self.raw_crk = [str(x) for x in z['raw_category_right_keys']]
        self.raw_cro = z['raw_category_right_offsets']
        self.vowel_keys = [str(x) for x in z['vowel_keys']]
        self.vowel_priors = np.asarray(z['vowel_priors'], float)
        self.vowel_f2_raw_mean = np.asarray(z['vowel_f2_raw_mean'], float)
        self.vowel_f2_raw_var = np.asarray(z['vowel_f2_raw_var'], float)
        self.vowel_f2_norm_mean = np.asarray(z['vowel_f2_norm_mean'], float)
        self.vowel_f2_norm_var = np.asarray(z['vowel_f2_norm_var'], float)
        self._gauss_cache = {}
        self._conditional_cache = {}

    def _off(self, key, keys, offs):
        if key is None:
            return np.zeros(len(self.features))
        try:
            return offs[keys.index(str(key))]
        except ValueError:
            return np.zeros(len(self.features))

    def _speaker_scale(self, speaker):
        if speaker is None:
            return 1.0
        try:
            return float(self.ssc[self.ssk.index(str(speaker))])
        except ValueError:
            return 1.0

    def expectation(self, speaker = None, left = None, right = None):
        return self.global_mean + self._off(speaker, self.sk, self.so) + self._off(left, self.lk, self.lo) + self._off(right, self.rk, self.ro)

    def _params(self, raw):
        if raw:
            return (
                self.raw_global,
                self.raw_scale,
                self.raw_means,
                self.raw_covs,
                self.raw_support,
                self.raw_presence,
                self.raw_crk,
                self.raw_cro,
            )
        return (self.global_mean, self.scale, self.means, self.covs, self.support, self.presence, self.crk, self.cro)

    def _category_mean(self, i, idx, right = None, raw = False):
        _, _, means, _, _, _, rkeys, roffs = self._params(raw)
        mu = np.asarray(means[i, idx], float).copy()
        if right is not None:
            try:
                ri = rkeys.index(str(right))
            except ValueError:
                ri = None
            if ri is not None:
                mu += roffs[i, ri, idx]
        return mu

    @staticmethod
    def _logsumexp(values):
        a = np.asarray(values, float)
        m = float(np.max(a))
        return m + math.log(float(np.exp(a - m).sum()))

    @staticmethod
    def _normal_log_survival(z):
        pp = 0.5 * math.erfc(float(z) / math.sqrt(2.0))
        return math.log(max(pp, 1e-300))

    def _conditional_mean_var(
        self,
        i,
        target_j,
        obs_idx,
        zvals,
        right = None,
        raw = False,
    ):
        _, _, _, covs, support, _, _, _ = self._params(raw)
        mu_t = float(self._category_mean(i, np.asarray([target_j]), right, raw)[0])
        var_t = float(covs[i, target_j, target_j])
        if len(obs_idx) == 0:
            return (mu_t, max(var_t, 1e-08))
        mask = np.asarray([support[i, j] and j != target_j for j in obs_idx], bool)
        if not mask.any():
            return (mu_t, max(var_t, 1e-08))
        eidx = obs_idx[mask]
        x = zvals[mask]
        key = (raw, i, int(target_j), tuple((int(q) for q in eidx)))
        cached = self._conditional_cache.get(key)
        if cached is None:
            Cee = covs[i][np.ix_(eidx, eidx)]
            Cte = covs[i][np.ix_([target_j], eidx)].reshape(-1)
            try:
                inv = np.linalg.inv(Cee)
            except np.linalg.LinAlgError:
                inv = np.linalg.inv(Cee + np.eye(len(eidx)) * 1e-05)
            beta = Cte @ inv
            cv = max(var_t - float(beta @ Cte), 1e-08)
            cached = (beta, cv)
            self._conditional_cache[key] = cached
        beta, cv = cached
        mu_e = self._category_mean(i, eidx, right, raw)
        return (mu_t + float(beta @ (x - mu_e)), cv)

    def _beta_binomial_log_kernel(self, k, n, mean, var):
        mean = float(np.clip(mean, 0.0001, 1 - 0.0001))
        maxvar = mean * (1 - mean) * 0.98
        var = float(np.clip(var, 1e-06, max(maxvar, 1e-06)))
        conc = max(mean * (1 - mean) / var - 1, 0.001)
        a = mean * conc
        b = (1 - mean) * conc
        return math.lgamma(k + a) + math.lgamma(n - k + b) - math.lgamma(n + a + b) - math.lgamma(a) - math.lgamma(b) + math.lgamma(a + b)

    def _context_logits(
        self,
        obs: dict[str, float],
        *,
        speaker = None,
        left = None,
        right = None,
        chosen: Sequence[str],
        censored: Optional[dict[str, float]] = None,
        partial_voicing: Optional[tuple[int, int]] = None,
    ) -> np.ndarray:
        raw = speaker is None
        global_mean, scale, means, covs, support, presence, _, _ = self._params(raw)
        if raw:
            exp = global_mean
        else:
            exp = self.expectation(speaker, left, right)
        # frication duration is observed as temporal evidence (endpoint/ongoing), not as an independent identity measurement
        # duration perturbations therefore affect release timing without overriding spectral identity
        names = [c for c, v in obs.items() if c in self.fi and c != 'frication_duration_ms' and np.isfinite(v)]
        obs_idx = np.asarray([self.fi[c] for c in names], int)
        obs_vals = np.asarray([float(obs[c]) for c in names], float)
        if not raw:
            ss = self._speaker_scale(speaker)
            if ss != 1:
                for q, j in enumerate(obs_idx):
                    if int(j) in self.spectral_idx:
                        obs_vals[q] /= ss
        zvals = (obs_vals - exp[obs_idx]) / scale[obs_idx] if len(obs_idx) else np.asarray([], float)
        censored = censored or {}
        logits = []
        for lab in chosen:
            i = self.labels.index(lab)
            lp = math.log(max(float(self.priors[i]), 1e-12))
            for j in obs_idx:
                lp += math.log(float(np.clip(presence[i, j], 1e-06, 1)))
            relmask = support[i, obs_idx]
            if relmask.any():
                idx = obs_idx[relmask]
                x = zvals[relmask]
                mu = self._category_mean(i, idx, right, raw)
                C = covs[i][np.ix_(idx, idx)]
                key = (raw, i, tuple((int(q) for q in idx)))
                cached = self._gauss_cache.get(key)
                if cached is None:
                    try:
                        sign, ld = np.linalg.slogdet(C)
                        inv = np.linalg.inv(C)
                        if sign <= 0:
                            raise np.linalg.LinAlgError
                    except np.linalg.LinAlgError:
                        C = C + np.eye(len(idx)) * 1e-05
                        sign, ld = np.linalg.slogdet(C)
                        inv = np.linalg.inv(C)
                    cached = (inv, float(ld))
                    self._gauss_cache[key] = cached
                inv, ld = cached
                delta = x - mu
                lp += -0.5 * (float(delta @ inv @ delta) + ld + len(idx) * math.log(2 * math.pi))
            for cue, lower_raw in censored.items():
                # the elapsed-time survival function belongs to the temporal release estimator
                if cue == 'frication_duration_ms':
                    continue
                if cue not in self.fi:
                    continue
                j = self.fi[cue]
                lp += math.log(float(np.clip(presence[i, j], 1e-06, 1)))
                if support[i, j]:
                    cm, cv = self._conditional_mean_var(
                        i, j, obs_idx, zvals, right, raw,
                    )
                    lower_z = (float(lower_raw) - exp[j]) / scale[j]
                    zz = (lower_z - cm) / math.sqrt(cv)
                    lp += self._normal_log_survival(zz)
            if partial_voicing is not None:
                kk, n = partial_voicing
                j = self.fi['frication_voicing_proportion']
                lp += math.log(float(np.clip(presence[i, j], 1e-06, 1)))
                if support[i, j] and n > 0:
                    cm, cv = self._conditional_mean_var(
                        i, j, obs_idx, zvals, right, raw,
                    )
                    mean_raw = exp[j] + scale[j] * cm
                    var_raw = scale[j] ** 2 * cv
                    lp += self._beta_binomial_log_kernel(int(kk), int(n), mean_raw, var_raw)
            logits.append(lp)
        return np.asarray(logits, float)

    def vowel_posterior(
        self,
        f2_value: float,
        *,
        speaker = None,
        frames: int = 1,
        prior: Optional[dict[str, float]] = None,
    ):
        if not np.isfinite(f2_value) or not self.vowel_keys:
            return {v: float(self.vowel_priors[i]) for i, v in enumerate(self.vowel_keys)}
        frames = max(1, int(frames))
        if speaker is None:
            value = float(f2_value)
            means = self.vowel_f2_raw_mean
            vars_ = self.vowel_f2_raw_var
        else:
            value = float(f2_value) / self._speaker_scale(speaker)
            means = self.vowel_f2_norm_mean
            vars_ = self.vowel_f2_norm_var
        if prior is None:
            pp = np.asarray(self.vowel_priors, float).copy()
        else:
            pp = np.asarray([max(float(prior.get(v, 0.0)), 1e-12) for v in self.vowel_keys], float)
            pp /= pp.sum()
        # only measurement uncertainty shrinks as the first few vowel frames accumulate
        eff_var = vars_ * (1.0 + 1.5 / frames)
        logits = np.log(np.clip(pp, 1e-12, 1.0)) - 0.5 * (np.log(2 * math.pi * eff_var) + (value - means) ** 2 / eff_var)
        logits -= float(np.max(logits))
        q = np.exp(logits)
        q /= q.sum()
        return {v: float(q[i]) for i, v in enumerate(self.vowel_keys)}

    def vowel_presence_strength(self, f2_value: float, *, speaker = None, frames: int = 1):
        """how vowel-like the observed onset F2 is, independent of vowel identity

        this uses a broad peak-normalized envelope over the learned /a/ and /i/ F2 distributions
        the envelope is intentionally broader than either identity distribution because organizational release asks whether a vowel is present
        """
        if not np.isfinite(f2_value) or not self.vowel_keys:
            return 0.0
        frames = max(1, int(frames))
        if speaker is None:
            value = float(f2_value)
            means = self.vowel_f2_raw_mean
            vars_ = self.vowel_f2_raw_var
        else:
            value = float(f2_value) / self._speaker_scale(speaker)
            means = self.vowel_f2_norm_mean
            vars_ = self.vowel_f2_norm_var
        eff_var = np.maximum(vars_ * (1.0 + 1.5 / frames), 1e-08)
        z2 = (value - means) ** 2 / eff_var
        # convert distance from the nearest vowel distribution into a two-sided Gaussian typicality probability
        # F2 onset supplies a small baseline for vowel presence, while acoustic typicality determines how quickly evidence accumulates
        nearest_z = math.sqrt(max(float(np.min(z2)), 0.0))
        calibrated_tail = math.erfc(nearest_z / math.sqrt(2.0))
        return float(np.clip(0.05 + 0.95 * calibrated_tail, 0.0, 1.0))

    def _future_expected_raw(
        self,
        label: str,
        cue: str,
        *,
        speaker = None,
        left = None,
        right = None,
        sd_shift: float = 0.0,
    ):
        if cue not in self.fi or label not in self.labels:
            return None
        j = self.fi[cue]
        i = self.labels.index(label)
        raw = speaker is None
        global_mean, scale, _, covs, support, _, _, _ = self._params(raw)
        if not support[i, j]:
            return None
        mu = float(self._category_mean(i, np.asarray([j]), right, raw)[0])
        sd = math.sqrt(max(float(covs[i, j, j]), 1e-08))
        z = mu + float(sd_shift) * sd
        if raw:
            return float(global_mean[j] + scale[j] * z)
        val = float(self.expectation(speaker, left, right)[j] + scale[j] * z)
        if j in self.spectral_idx:
            val *= self._speaker_scale(speaker)
        return val

    def temporal_completion_support(
        self,
        elapsed_ms: float,
        scores: dict[str, float],
        *,
        continuing: bool = False,
        horizon_ms: float = 5.0,
    ) -> float:
        """evidence of an imminent endpoint from learned duration distributions

        while a duration is right-censored, the stream establishes that the event has NOT occurred
        only the conditional probability of a future endpoint over the next horizon is relevant
        its marginal CDF is not evidence that an endpoint has already occurred
        """
        if not scores:
            return 0.0
        _, scale, _, covs, support, _, _, _ = self._params(True)
        weighted = 0.0
        total = 0.0
        for label, weight in scores.items():
            if label not in self.labels or weight <= 0:
                continue
            i = self.labels.index(label)
            for cue in ('frication_duration_ms', 'VOT_ms'):
                j = self.fi[cue]
                if not support[i, j]:
                    continue
                z = (float(elapsed_ms) - float(self.raw_global[j])) / max(float(scale[j]), 1e-08)
                mu = float(self._category_mean(i, np.asarray([j]), None, True)[0])
                sd = math.sqrt(max(float(covs[i, j, j]), 1e-08))
                cdf = 0.5 * (1.0 + math.erf((z - mu) / (sd * math.sqrt(2.0))))
                if continuing:
                    next_z = (float(elapsed_ms + horizon_ms) - float(self.raw_global[j])) / max(float(scale[j]), 1e-08)
                    next_cdf = 0.5 * (1.0 + math.erf((next_z - mu) / (sd * math.sqrt(2.0))))
                    # P(endpoint within horizon | no endpoint at elapsed_ms)
                    cdf = (next_cdf - cdf) / max(1.0 - cdf, 1e-12)
                weighted += float(weight) * float(np.clip(cdf, 0.0, 1.0))
                total += float(weight)
                break
        return weighted / total if total > 0 else 0.0

    def evidence_exhausted(self, obs: dict[str, float], winner: str) -> bool:
        """whether every learned acoustic feature for the current winner is observed

        this is an online check on actual observations
        an uncertain but complete percept can make a provisional MAP commitment and later be revised
        """
        if winner not in self.labels:
            return False
        i = self.labels.index(winner)
        required = np.flatnonzero(self.support[i])
        return bool(len(required)) and all(
            self.features[j] in obs and np.isfinite(obs[self.features[j]])
            for j in required
        )

    def future_robust(
        self,
        obs: dict[str, float],
        scores: dict[str, float],
        *,
        speaker = None,
        left = None,
        right_probs = None,
        labels: Optional[Sequence[str]] = None,
        censored = None,
        partial_voicing = None,
        robustness_horizon_frames: int = 10,
    ):
        if len(scores) < 2:
            return True
        order = sorted(scores, key = lambda lab: (-scores[lab], lab))
        winner, runner = order[:2]
        chosen = self.labels if labels is None else [l for l in labels if l in self.labels]
        if len(chosen) <= 2:
            return True
        raw = speaker is None
        _, _, _, covs, support, _, _, _ = self._params(raw)
        wi = self.labels.index(winner)
        ri = self.labels.index(runner)
        known = set(obs)
        future = []
        for cue in self.features:
            if cue in known:
                continue
            # whole-frication duration is used by the temporal release mechanism rather than projected as future place evidence
            if cue == 'frication_duration_ms':
                continue
            # observed voicing frames cannot be replaced with an imagined completed value
            # instead, test possible continuations below
            if cue == 'frication_voicing_proportion' and partial_voicing is not None:
                continue
            j = self.fi[cue]
            if not (support[wi, j] and support[ri, j]):
                continue
            # only unresolved cues capable of materially discriminating the current top two
            wm = float(self._category_mean(wi, np.asarray([j]), None, raw)[0])
            rm = float(self._category_mean(ri, np.asarray([j]), None, raw)[0])
            sd = math.sqrt(max(0.5 * (float(covs[wi, j, j]) + float(covs[ri, j, j])), 1e-08))
            if abs(wm - rm) / sd >= 0.5:
                future.append(cue)
        if not future and partial_voicing is None:
            return True
        if right_probs:
            avail = self.raw_crk if raw else self.crk
            ctx = [v for v, w in right_probs.items() if w >= 0.1 and v in avail]
        else:
            ctx = []
        if not ctx:
            ctx = [None]
        for right in ctx:
            aug = dict(obs)
            for cue in future:
                v = self._future_expected_raw(runner, cue, speaker = speaker, left = left, right = right)
                if v is not None and np.isfinite(v):
                    aug[cue] = v
            # treat a partial measurement as progressively accumulated evidence
            # test both extremes of a fixed short future horizon without looking ahead to the actual token endpoint or frames
            # the same horizon applies to any partial channel, with no stop/fricative switch or release requirement
            scenarios = [None]
            if partial_voicing is not None:
                k, n = partial_voicing
                h = max(1, int(robustness_horizon_frames))
                # use a Beta(1,1) predictive continuation
                a, b = int(k) + 1, int(n) - int(k) + 1
                mean = h * a / (a + b)
                variance = h * a * b * (a + b + h) / (
                    (a + b) ** 2 * (a + b + 1)
                )
                spread = 1.645 * math.sqrt(variance)
                possibles = {int(np.clip(round(x), 0, h)) for x in
                             (mean - spread, mean, mean + spread)}
                scenarios = [(int(k), int(n))] + [
                    (int(k) + extra, int(n) + h)
                    for extra in sorted(possibles)
                ]
            for scenario in scenarios:
                sc = self.score(
                    aug,
                    speaker = speaker,
                    left = left,
                    right = right,
                    labels = chosen,
                    censored = None,
                    partial_voicing = scenario,
                )
                if not sc or max(sc, key = sc.get) != winner:
                    return False
        return True

    def score(
        self,
        obs: dict[str, float],
        *,
        speaker = None,
        left = None,
        right = None,
        right_probs: Optional[dict[str, float]] = None,
        labels: Optional[Sequence[str]] = None,
        censored: Optional[dict[str, float]] = None,
        partial_voicing: Optional[tuple[int, int]] = None,
    ):
        chosen = self.labels if labels is None else [l for l in labels if l in self.labels]
        if not chosen:
            return {}
        available = self.raw_crk if speaker is None else self.crk
        if right is not None:
            weights = {str(right): 1.0}
        elif right_probs is not None:
            weights = {str(v): max(float(w), 0.0) for v, w in right_probs.items() if str(v) in available and float(w) > 0}
            z = sum(weights.values())
            if z > 0:
                weights = {v: w / z for v, w in weights.items()}
        else:
            contexts = [v for v in ('a', 'i') if v in available]
            weights = {v: 1 / len(contexts) for v in contexts} if contexts else {}
        kwargs = dict(
            obs = obs,
            speaker = speaker,
            left = left,
            chosen = chosen,
            censored = censored,
            partial_voicing = partial_voicing,
        )
        if len(weights) == 1:
            v = next(iter(weights))
            logits = self._context_logits(right = v, **kwargs)
        elif weights:
            arrs = [(self._context_logits(right = v, **kwargs), math.log(max(w, 1e-300))) for v, w in weights.items()]
            logits = np.asarray([self._logsumexp([a[j] + lw for a, lw in arrs]) for j in range(len(chosen))], float)
        else:
            logits = self._context_logits(right = None, **kwargs)
        logits -= float(np.max(logits))
        pp = np.exp(logits)
        pp /= pp.sum()
        return dict(zip(chosen, pp))


def make_hand_bundle(path: str | Path) -> None:
    rng = np.random.default_rng(101)
    speakers = make_speaker_offsets(2026, 8)
    rows = []
    skeys = list(speakers)
    cells = [(lab, sp, left, right) for lab in LABELS for sp in skeys for left in LEFT_CONTEXTS for right in RIGHT_CONTEXTS]
    order = []
    while len(order) < 7000:
        block = list(cells)
        rng.shuffle(block)
        order.extend(block)
    for lab, sp, left, right in order[:7000]:
        x = sample_token(lab, rng, speakers[sp], left, right)
        row = {c: float(x[j]) if np.isfinite(x[j]) else np.nan for j, c in enumerate(CUES)}
        row.update(
            current_sound = lab.split('|')[0],
            rep_env = 'C_prevocalic',
            speaker = sp,
            prev_spoken_label = left,
            next_spoken_label = right,
        )
        rows.append(row)
    fit_bundle(pd.DataFrame(rows), path, min_label_n = 20)


@dataclass
class Trial:
    truth: str
    speaker: str
    raw: np.ndarray
    arrivals: dict[str, int]
    boundary_ms: int
    left_context: Optional[str] = 'neutral_rate'
    right_context: Optional[str] = None
    right_context_reveal_ms: Optional[int] = None
    end_ms: int = 350
    # right_context_reveal_ms marks the onset of usable vowel-context evidence, with None for unavailable and 0 for known from onset
    # natural trials infer a graded /a/-/i/ posterior across vowel frames rather than receiving a categorical vowel label
    # speaker_reveal_ms is independent: negative means unavailable and 0 means known from onset
    speaker_reveal_ms: Optional[int] = None


@dataclass
class Tracker:
    state: str = 'WAIT'
    obs: dict[str, float] = field(default_factory = dict)
    stable: int = 0
    candidate_label: Optional[str] = None
    committed_label: Optional[str] = None
    committed_prob: float = 0.0
    last_scores: dict[str, float] = field(default_factory = dict)
    reprocess_origin: Optional[str] = None
    context_probs: dict[str, float] = field(default_factory = dict)


# the acoustic feature families represent independently acquired observation channels
# multiple statistics of the same frication spectrum count as one source
# right-censored durations and partial frame counts are not completed cues
ACOUSTIC_SOURCES = {
    'frication_M1_kHz': 'spectrum',
    'frication_M2_kHz': 'spectrum',
    'frication_M3_skewness': 'spectrum',
    'burst_M1_kHz': 'spectrum',
    'VOT_ms': 'duration',
    'following_vowel_duration_ms': 'duration',
    'frication_voicing_proportion': 'voicing',
    'vowel_onset_f0_ST': 'pitch',
    'F2_onset_kHz': 'formant',
}


def independent_observation_sources(obs: dict[str, float]) -> frozenset[str]:
    """count distinct completed measurement channels seen so far

    this never consults the trial's future arrivals or ground truth
    one spectral frame with three correlated moments is still only one source
    """
    return frozenset(
        ACOUSTIC_SOURCES[cue] for cue, value in obs.items()
        if cue in ACOUSTIC_SOURCES and np.isfinite(value)
    )


def _advance_persistence(tracker: Tracker, winner: str, ready: bool) -> int:
    """persistence is consecutive readiness for ONE categorical hypothesis

    changes of winner break the streak, including while reprocessing
    """
    if not ready:
        tracker.stable = 0
        tracker.candidate_label = None
    elif tracker.candidate_label != winner:
        tracker.candidate_label = winner
        tracker.stable = 1
    else:
        tracker.stable += 1
    return tracker.stable


def _winner(scores: dict[str, float]):
    o = sorted(scores.items(), key = lambda kv: (-kv[1], kv[0]))
    return (o[0][0], o[0][1], o[0][1] - o[1][1])


def _frication_voicing_sequence(tr: Trial, dt: int) -> np.ndarray:
    """a duration-invariant prefix of 5-ms voicing observations

    matched duration manipulations change the endpoint of a token, not the acoustic observations preceding the shorter endpoint
    the RNG seed and per-frame realizations must NOT depend on duration or on the number of frames generated
    balanced stochastic rounding maintains the observed voicing fraction to within one frame at every prefix length
    """
    dur = max(int(tr.boundary_ms), dt)
    n = max(1, int(math.ceil(dur / dt)))
    p = float(np.clip(tr.raw[CUE_INDEX['frication_voicing_proportion']], 0, 1))
    finite = np.nan_to_num(tr.raw, nan = 0.0, posinf = 0.0, neginf = 0.0).copy()
    finite[CUE_INDEX['frication_duration_ms']] = 0.0
    weights = np.arange(1, len(finite) + 1, dtype = float)
    numeric = int(abs(float(finite @ weights)) * 1000000) & 4294967295
    text_seed = sum(((i + 1) * ord(ch) for i, ch in enumerate(tr.truth + tr.speaker)))
    rng = np.random.default_rng((numeric + text_seed) & 4294967295)
    phase = rng.random()
    running = np.floor(phase + np.arange(n + 1, dtype = float) * p)
    return np.diff(running).astype(np.int8)


def run_controller(
    scorer: Scorer,
    tr: Trial,
    *,
    labels: Sequence[str] = LABELS,
    theta: float = 0.55,
    margin: float = 0.20,
    persist: int = 2,
    theta_drop: float = 0.08,
    dt: int = 5,
    stop_after_first_commit: bool = False,
    future_robustness: bool = True,
    organizational_release: bool = True,
    high_confidence: float = 0.95,
    partial_rate_se_max: float = 0.04,
    trace: bool = False,
):
    # replay the two trackers at each frame, with primary initially evaluating the consonant
    # observed vowel-onset evidence transfers the earlier candidate to the secondary tracker
    # primary supplies its graded /a/-/i/ posterior to secondary without changing the four-state automaton
    P = Tracker()
    S = Tracker()
    handed = False
    current_speaker = None
    rows = []
    revisions = []
    first_commit_ms = None
    first_commit_label = None
    first_commit_reason = None
    reprocess_ms = None
    recommit_ms = None
    prior = {v: float(scorer.vowel_priors[i]) for i, v in enumerate(scorer.vowel_keys)}
    if sum(prior.values()) <= 0:
        prior = {'a': 0.5, 'i': 0.5}
    vowel_probs = dict(prior)
    vowel_known_from_onset = tr.right_context_reveal_ms == 0 and tr.right_context in prior
    vowel_context_disabled = tr.right_context_reveal_ms is None
    if vowel_known_from_onset:
        vowel_probs = {v: 1.0 if v == tr.right_context else 0.0 for v in prior}
    is_fric = any(cue in tr.arrivals for cue in ('frication_M1_kHz', 'frication_duration_ms'))
    arrival_by_t = {}
    for cue, t in tr.arrivals.items():
        arrival_by_t.setdefault(int(t), []).append(cue)
    voice_seq = _frication_voicing_sequence(tr, dt) if is_fric else None
    for t in range(0, int(tr.end_ms) + dt, dt):
        context_changed = False
        speaker_reveal = tr.right_context_reveal_ms if tr.speaker_reveal_ms is None else None if tr.speaker_reveal_ms < 0 else tr.speaker_reveal_ms
        if speaker_reveal is not None and t >= speaker_reveal and (current_speaker != tr.speaker):
            current_speaker = tr.speaker
            context_changed = True
        if not handed and 'F2_onset_kHz' in arrival_by_t.get(t, ()) and np.isfinite(tr.raw[CUE_INDEX['F2_onset_kHz']]):
            S = Tracker(
                state = P.state,
                obs = dict(P.obs),
                stable = P.stable,
                candidate_label = P.candidate_label,
                committed_label = P.committed_label,
                committed_prob = P.committed_prob,
                last_scores = dict(P.last_scores),
                reprocess_origin = P.reprocess_origin,
            )
            P = Tracker(context_probs = dict(vowel_probs))
            if vowel_known_from_onset:
                P.state = 'COMMITTED'
                P.committed_label = tr.right_context
                P.committed_prob = 1.0
                P.stable = persist
            handed = True
        # after the consonant-to-vowel handoff, primary infers vowel identity from F2 with increasing precision across vowel frames
        # natural trials never reveal the true vowel category directly
        if handed and (not vowel_context_disabled) and (not vowel_known_from_onset):
            onset = int(tr.right_context_reveal_ms) if tr.right_context_reveal_ms is not None else None
            if onset is not None and t >= onset:
                frames = 1 + max(0, (t - onset) // dt)
                f2 = float(tr.raw[CUE_INDEX['F2_onset_kHz']])
                new_probs = scorer.vowel_posterior(f2, speaker = current_speaker, frames = frames, prior = prior)
                delta = max((abs(new_probs.get(v, 0) - vowel_probs.get(v, 0)) for v in set(new_probs) | set(vowel_probs)))
                if delta > 1e-08:
                    context_changed = True
                vowel_probs = new_probs
                P.context_probs = dict(vowel_probs)
                top_v = max(vowel_probs, key = vowel_probs.get)
                vp = float(vowel_probs[top_v])
                runner = max([q for v, q in vowel_probs.items() if v != top_v] or [0.0])
                vmar = vp - runner
                if P.state == 'WAIT':
                    P.state = 'EVALUATING'
                if P.state == 'EVALUATING':
                    _advance_persistence(P, top_v, vp >= theta and vmar >= margin)
                    if P.stable >= persist:
                        P.state = 'COMMITTED'
                        P.committed_label = top_v
                        P.committed_prob = vp
                elif P.state == 'COMMITTED':
                    old = P.committed_label
                    oldp = float(vowel_probs.get(old, 0.0))
                    drop = P.committed_prob - oldp
                    if top_v != old and (drop >= theta_drop or vp - oldp >= margin * 0.5):
                        P.state = 'REPROCESSING'
                        P.reprocess_origin = old
                        P.stable = 0
                        P.candidate_label = None
                    else:
                        P.committed_prob = oldp
                elif P.state == 'REPROCESSING':
                    _advance_persistence(P, top_v, vp >= theta and vmar >= margin)
                    if P.stable >= persist:
                        P.state = 'COMMITTED'
                        P.committed_label = top_v
                        P.committed_prob = vp
        active = S if handed else P
        exact_changed = False
        for cue in arrival_by_t.get(t, []):
            j = CUE_INDEX[cue]
            v = tr.raw[j]
            if np.isfinite(v):
                # following-vowel measurements are evidence about the retained consonant as well as evidence for the primary vowel context
                active.obs[cue] = float(v)
                exact_changed = True
        censored = {}
        partial_voicing = None
        if is_fric:
            # duration is right-censored until its endpoint is observed in the input
            completed = 'frication_duration_ms' in active.obs
            if t >= 30 and not completed:
                censored['frication_duration_ms'] = float(t)
                if voice_seq is not None:
                    n_obs = min(len(voice_seq), max(1, int(math.floor(t / dt))))
                    partial_voicing = (int(voice_seq[:n_obs].sum()), n_obs)
            if completed and voice_seq is not None and len(voice_seq):
                # the completed measurement is the token's stored continuous voicing fraction
                # discretized partial frames approximate it
                active.obs['frication_voicing_proportion'] = float(tr.raw[CUE_INDEX['frication_voicing_proportion']])
        else:
            raw_vot = float(tr.raw[CUE_INDEX['VOT_ms']])
            vot_arrival = int(tr.arrivals['VOT_ms'])
            if t >= 5 and t < vot_arrival and (raw_vot > 0):
                censored['VOT_ms'] = float(t)
            vowel_onset = int(tr.arrivals['F2_onset_kHz'])
            vowel_end = int(tr.arrivals['following_vowel_duration_ms'])
            if t >= vowel_onset + 10 and t < vowel_end:
                censored['following_vowel_duration_ms'] = float(t - vowel_onset)
        has_evidence = bool(active.obs or censored or partial_voicing is not None)
        temporal_changed = bool(censored or partial_voicing is not None)
        new_evidence = exact_changed or temporal_changed
        release_support = 0.0
        duration_support = 0.0
        vowel_support = 0.0
        trace_payload = {}
        commit_reason = None
        exhausted = False
        robust = True
        if has_evidence:
            # identify the segment from available acoustic cues
            # maintain duration independently as ongoing or completed temporal evidence
            category_obs = active.obs
            # only rescore when evidence or contextual expectations actually change
            if active.last_scores and not new_evidence and not context_changed:
                scores = active.last_scores
            else:
                scores = scorer.score(
                    category_obs,
                    speaker = current_speaker,
                    left = tr.left_context,
                    right_probs = vowel_probs,
                    labels = labels,
                    censored = censored,
                    partial_voicing = partial_voicing,
                )
            top, p, mar = _winner(scores)
            active.last_scores = scores
            if trace:
                # snapshot the probabilities and actual observed cues for this frame
                # cached scores do not trigger a fresh scoring call
                trace_payload = {
                    'trace_scores': dict(scores),
                    'trace_obs': dict(category_obs),
                    'trace_censored': dict(censored),
                    'trace_partial_voicing': partial_voicing,
                }
            # the same release estimator applies to all consonants
            # an ongoing cue provides near-future endpoint hazard, not evidence that its endpoint has already occurred
            ongoing_duration = ('frication_duration_ms' in censored or 'VOT_ms' in censored)
            completed_duration = ('frication_duration_ms' in active.obs or 'VOT_ms' in active.obs)
            if completed_duration:
                # the observed completion of either temporal cue is direct positive evidence of segment completion
                duration_support = 1.0
            else:
                duration_support = scorer.temporal_completion_support(
                    float(t), scores, continuing = ongoing_duration, horizon_ms = dt,
                )
            vowel_support = 0.0
            onset = tr.arrivals.get('F2_onset_kHz')
            if onset is not None and t >= int(onset) and 'F2_onset_kHz' in active.obs:
                frames = 1 + max(0, (t - int(onset)) // dt)
                presence = scorer.vowel_presence_strength(float(active.obs['F2_onset_kHz']), speaker = current_speaker, frames = frames)
                vowel_support = 1.0 - math.exp(-(float(frames) * presence) / 1.5)
            release_support = 1.0 - (1.0 - duration_support) * (1.0 - vowel_support)
            if not organizational_release:
                release_support = 1.0
            if active.state == 'WAIT':
                active.state = 'EVALUATING'
            if active.state == 'EVALUATING':
                # category commitment and segment release are different actions
                # release support remains a segmentation diagnostic, while actual handoff is triggered by observed vowel-onset evidence
                # release support cannot veto sufficiently strong acoustic evidence because a fricative may commit during ongoing frication
                # the same decision criteria apply to every category
                confident = p >= theta and mar >= margin
                robust = True
                sources = independent_observation_sources(category_obs)
                if confident and future_robustness and p < high_confidence:
                    # correlated spectral moments remain one source
                    # a partial voicing history counts as an independent ongoing measurement when hypothetical upcoming frames cannot overturn the decision
                    # the partial voicing evidence need not wait for segment offset
                    # extremely decisive single-channel evidence remains enough
                    # posterior uncertainty of the accumulated Bernoulli channel shrinks with each frame
                    # the accumulated channel becomes independently usable once its inferred rate meets the configured standard-error bound
                    # this is an acoustic reliability check, not a test for segment completion
                    voicing_precise = False
                    if partial_voicing is not None:
                        k, n = partial_voicing
                        a, b = k + 1, n - k + 1
                        rate_se = math.sqrt(a * b / ((a + b)**2 * (a + b + 1)))
                        voicing_precise = rate_se <= partial_rate_se_max
                    ongoing_sources = sources | (
                        frozenset({'voicing'}) if voicing_precise else frozenset()
                    )
                    robust = len(ongoing_sources) >= 2 and scorer.future_robust(
                        category_obs,
                        scores,
                        speaker = current_speaker,
                        left = tr.left_context,
                        right_probs = vowel_probs,
                        labels = labels,
                        censored = censored,
                        partial_voicing = partial_voicing,
                    )
                # when all features expected under the winning learned model have been observed, the evidence stream can be exhausted without crossing the ordinary posterior threshold
                # a provisional maximum-posterior commitment is then legitimate
                # this uses neither the token-final timestamp nor future cue arrivals nor the true label
                exhausted = scorer.evidence_exhausted(category_obs, top)
                ready = (confident and robust) or exhausted
                if ready:
                    commit_reason = ('high_confidence' if confident and p >= high_confidence
                        else 'robust_acoustic' if confident and robust
                        else 'evidence_exhaustion')
                _advance_persistence(active, top, ready)
                if active.stable >= persist:
                    active.state = 'COMMITTED'
                    active.committed_label = top
                    active.committed_prob = p
                    if first_commit_ms is None:
                        first_commit_ms = t
                        first_commit_label = top
                        first_commit_reason = commit_reason
                        if stop_after_first_commit:
                            rows.append(dict(
                                t_ms = t,
                                release_support = release_support,
                                duration_support = duration_support,
                                vowel_support = vowel_support,
                                handoff = int(handed),
                                primary_state = P.state,
                                secondary_state = S.state,
                                top = top,
                                top_prob = p,
                                margin = mar,
                                n_cues = len(active.obs),
                                commitment_reason = first_commit_reason,
                                completed_sources = len(sources),
                                evidence_exhausted = int(exhausted),
                                n_censored = len(censored),
                                partial_voicing_n = partial_voicing[1] if partial_voicing is not None else 0,
                                vowel_a_prob = float(vowel_probs.get('a', 0)),
                                vowel_i_prob = float(vowel_probs.get('i', 0)),
                                vowel_context_top = max(vowel_probs, key = vowel_probs.get),
                            ))
                            final_tracker = S if handed else P
                            return dict(
                                rows = rows,
                                initial_commit_ms = first_commit_ms,
                                initial_commit_label = first_commit_label,
                                initial_commit_reason = first_commit_reason,
                                reprocess_ms = None,
                                recommit_ms = None,
                                final_label = final_tracker.committed_label,
                                genuine_revision = False,
                                revisions = [],
                            )
            elif active.state == 'COMMITTED':
                old = active.committed_label
                oldp = scores.get(old, 0.0)
                drop = active.committed_prob - oldp
                if (new_evidence or context_changed) and top != old and (drop >= theta_drop or p - oldp >= margin * 0.5):
                    active.state = 'REPROCESSING'
                    active.reprocess_origin = old
                    active.stable = 0
                    active.candidate_label = None
                    if reprocess_ms is None:
                        reprocess_ms = t
                else:
                    active.committed_prob = oldp
            elif active.state == 'REPROCESSING':
                _advance_persistence(active, top, p >= theta and mar >= margin)
                if active.stable >= persist:
                    old = active.reprocess_origin
                    active.state = 'COMMITTED'
                    active.committed_label = top
                    active.committed_prob = p
                    recommit_ms = t
                    if old != top:
                        revisions.append((t, old, top))
            top_full, prob_full, margin_full = (top, p, mar)
        else:
            top_full = ''
            prob_full = 0
            margin_full = 0
        rows.append(dict(
            t_ms = t,
            release_support = release_support,
            duration_support = duration_support,
            vowel_support = vowel_support,
            handoff = int(handed),
            primary_state = P.state,
            secondary_state = S.state,
            top = top_full,
            top_prob = prob_full,
            margin = margin_full,
            commitment_reason = commit_reason,
            completed_sources = len(independent_observation_sources(active.obs)),
            evidence_exhausted = int(exhausted),
            n_cues = len(active.obs),
            n_censored = len(censored),
            partial_voicing_n = partial_voicing[1] if partial_voicing is not None else 0,
            vowel_a_prob = float(vowel_probs.get('a', 0)),
            vowel_i_prob = float(vowel_probs.get('i', 0)),
            vowel_context_top = max(vowel_probs, key = vowel_probs.get),
            **trace_payload,
        ))
    final_tracker = S if handed else P
    return dict(
        rows = rows,
        initial_commit_ms = first_commit_ms,
        initial_commit_label = first_commit_label,
        initial_commit_reason = first_commit_reason,
        reprocess_ms = reprocess_ms,
        recommit_ms = recommit_ms,
        final_label = final_tracker.committed_label,
        genuine_revision = bool(revisions),
        revisions = revisions,
    )


def arrivals_for(label: str, x: np.ndarray) -> tuple[dict[str, int], int, int]:
    if label in FRIC:
        dur = int(round(float(x[CUE_INDEX['frication_duration_ms']]) / 5) * 5)
        if dur <= 0:
            raise ValueError(f'frication duration must be positive, got {dur} ms')
        arr = {'frication_M1_kHz': 30, 'frication_M2_kHz': 30, 'frication_M3_skewness': 30, 'frication_duration_ms': dur, 'frication_voicing_proportion': dur, 'F2_onset_kHz': dur + 10}
        # keep replay running into the following vowel so the retained consonant tracker is not artificially censored after 30 ms
        return (arr, dur, dur + 80)
    raw_vot = float(x[CUE_INDEX['VOT_ms']])
    if raw_vot <= 5:
        vot_arrival = 5
    else:
        vot_arrival = int(np.clip(round(raw_vot / 5) * 5, 5, 100))
    vowel_onset = max(10, vot_arrival)
    vlen = int(np.clip(round(float(x[CUE_INDEX['following_vowel_duration_ms']]) / 5) * 5, 90, 320))
    arr = {'burst_M1_kHz': 5, 'VOT_ms': vot_arrival, 'vowel_onset_f0_ST': vowel_onset + 10, 'F2_onset_kHz': vowel_onset + 10, 'following_vowel_duration_ms': vowel_onset + vlen}
    return (arr, vowel_onset, vowel_onset + vlen + 20)
