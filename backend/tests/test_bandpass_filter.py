# tests/test_bandpass_filter.py - verification for app/preprocessing/bandpass_filter.py
#
# Three areas, matching the review plan:
#   1. Filter design      - ECG and PPG produce usable, distinct SOS coefficients.
#   2. Noise reduction    - a 50 Hz interference component is attenuated while
#                           the in-band signal survives.
#   3. Error handling     - unsupported modalities and invalid sampling rates
#                           are rejected rather than silently mis-filtering.

import numpy as np
import pytest

from app.preprocessing.bandpass_filter import (
    apply_bandpass_filter,
    design_bandpass_filter,
)

# ---------------------------------------------------------------------------
# 1. Filter design
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("modality", ["ecg", "ppg"])
def test_design_returns_finite_sos(modality):
    # A band-pass of order 4 is built from 4 second-order sections, each of
    # shape (b0, b1, b2, a0, a1, a2).
    sos = design_bandpass_filter(modality, sampling_rate=250)

    assert sos.shape == (4, 6)
    assert np.isfinite(sos).all()


def test_modality_is_case_insensitive():
    # The database stores "ECG"/"PPG", the filter keys are lowercase.
    lower = design_bandpass_filter("ecg", 250)
    upper = design_bandpass_filter("ECG", 250)

    assert np.array_equal(lower, upper)


def test_ecg_and_ppg_filters_differ():
    # Same function, different band -> different coefficients.
    assert not np.allclose(
        design_bandpass_filter("ecg", 250),
        design_bandpass_filter("ppg", 250),
    )


def test_higher_order_adds_sections():
    assert design_bandpass_filter("ecg", 250, order=2).shape[0] == 2
    assert design_bandpass_filter("ecg", 250, order=6).shape[0] == 6


# ---------------------------------------------------------------------------
# 2. Noise reduction
# ---------------------------------------------------------------------------


def _rms_db_ratio(modality, sampling_rate, freq, seconds=30.0):
    """Gain at `freq` in dB, measured away from the padded edges."""
    sos = design_bandpass_filter(modality, sampling_rate)
    t = np.arange(0.0, seconds, 1.0 / sampling_rate)
    raw = np.sin(2 * np.pi * freq * t)
    filtered = apply_bandpass_filter(raw, sos)

    body = slice(len(filtered) // 4, -len(filtered) // 4)
    return 20 * np.log10(
        np.sqrt(np.mean(filtered[body] ** 2)) / np.sqrt(np.mean(raw[body] ** 2))
    )


def test_50hz_interference_is_attenuated():
    # 50 Hz mains hum sits above the 40 Hz ECG upper cutoff.
    assert _rms_db_ratio("ecg", 250, 50) < -20


def test_ecg_passband_is_not_attenuated():
    # 10 Hz is squarely inside 0.5-40 Hz and must pass essentially unchanged.
    assert _rms_db_ratio("ecg", 250, 10) > -1


def test_baseline_wander_is_removed():
    # Slow drift (0.1 Hz) is below the 0.5 Hz lower cutoff.
    assert _rms_db_ratio("ecg", 250, 0.1) < -40


def test_ppg_high_frequencies_are_attenuated():
    # 20 Hz is well above the 8 Hz PPG ceiling.
    assert _rms_db_ratio("ppg", 100, 20) < -40


def test_filtering_reduces_error_against_clean_reference():
    fs = 250
    rng = np.random.default_rng(0)
    t = np.arange(0.0, 10.0, 1.0 / fs)

    clean = 0.5 * np.sin(2 * np.pi * 1.2 * t) + 0.2 * np.sin(2 * np.pi * 15 * t)
    noisy = (
        clean
        + 0.3 * np.sin(2 * np.pi * 0.05 * t)  # baseline wander
        + 0.3 * np.sin(2 * np.pi * 50 * t)  # mains interference
        + 0.05 * rng.normal(size=t.size)  # sensor noise
    )

    filtered = apply_bandpass_filter(noisy, design_bandpass_filter("ECG", fs))

    rmse_noisy = np.sqrt(np.mean((noisy - clean) ** 2))
    rmse_filtered = np.sqrt(np.mean((filtered - clean) ** 2))

    assert filtered.shape == noisy.shape
    assert rmse_filtered < rmse_noisy


# ---------------------------------------------------------------------------
# 3. Error handling
# ---------------------------------------------------------------------------


def test_unsupported_modality_is_rejected():
    with pytest.raises(ValueError, match="Unsupported signal type"):
        design_bandpass_filter("emg", 250)


def test_unsupported_modality_is_rejected_regardless_of_case():
    with pytest.raises(ValueError, match="Unsupported signal type"):
        design_bandpass_filter("EMG", 250)


@pytest.mark.parametrize("sampling_rate", [60, 40, 0, -100])
def test_invalid_sampling_rate_is_rejected(sampling_rate):
    # 60 Hz and 40 Hz are below twice the 40 Hz upper cutoff, so they violate
    # the Nyquist limit; 0 and negative rates are meaningless. SciPy raises
    # ValueError for each, which is the behaviour the pipeline can rely on.
    with pytest.raises(ValueError):
        design_bandpass_filter("ecg", sampling_rate)


def test_too_short_signal_is_rejected():
    # sosfiltfilt pads by 27 samples for this filter (3 * (2 * 4 sections + 1)),
    # so shorter input must fail loudly instead of returning a distorted signal.
    sos = design_bandpass_filter("ecg", 250)

    with pytest.raises(ValueError):
        apply_bandpass_filter(np.zeros(27), sos)

    assert apply_bandpass_filter(np.zeros(28), sos).shape == (28,)
