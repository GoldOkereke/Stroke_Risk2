# app/preprocessing/bandpass_filter.py - band-pass filtering for raw signals.
#
# Where this sits in the pipeline:
#
#     uploaded file -> decode -> band-pass filter -> feature extraction -> model
#
# Raw ECG and PPG recordings carry more than the physiology we care about:
# baseline wander from breathing or electrode drift (very low frequencies),
# and mains hum, muscle noise or sensor noise (high frequencies). A band-pass
# filter keeps the band where the signal of interest lives and attenuates the
# rest, before any features are computed from it.
#
# Two steps, deliberately separate:
#   1. design_bandpass_filter() - build the filter for a modality and rate.
#   2. apply_bandpass_filter()  - run a signal through that filter.
# Keeping them apart lets the coefficients be computed once and reused across
# many recordings, and makes each step testable on its own.

from scipy.signal import butter, sosfiltfilt


def design_bandpass_filter(
    modality: str,
    sampling_rate: float,
    order: int = 4,
):
    # Cutoffs in Hz. ECG keeps 0.5-40 Hz: the diagnostic content (QRS, P and T
    # waves) survives, while baseline wander below 0.5 Hz and muscle/mains noise
    # above 40 Hz are attenuated. PPG needs a much narrower band, because the
    # pulse waveform is slow - 0.5-8 Hz holds the cardiac component and the
    # respiratory-related variation without the harmonics that only add noise.
    filters = {
        "ecg": (0.5, 40),
        "ppg": (0.5, 8),
    }

    if modality not in filters:
        raise ValueError(f"Unsupported signal type: {modality}")

    low, high = filters[modality]

    # butter() returns the coefficients of a Butterworth response: maximally
    # flat in the passband, so nothing inside the band is distorted more than
    # anything else.
    #
    #   N      filter order. Higher orders roll off more sharply but cost more
    #          computation and risk instability in fixed-point contexts; 4 is a
    #          sensible prototype default.
    #   Wn     [low, high] in Hz (not normalised, because `fs` is supplied).
    #   btype  "bandpass" keeps frequencies inside Wn and rejects both sides.
    #   fs     the recording's own sampling frequency - never a hard-coded
    #          value. The cutoffs must stay below fs / 2 (the Nyquist limit) or
    #          SciPy raises an error, so a 40 Hz upper cutoff needs fs > 80 Hz.
    #   output "sos" returns Second-Order Sections: numerically better behaved
    #          than the equivalent (b, a) polynomial form, especially for higher
    #          orders, which is why it is the recommended representation.
    return butter(
        N=order,
        Wn=[low, high],
        btype="bandpass",
        fs=sampling_rate,
        output="sos",
    )


def apply_bandpass_filter(signal, sos):
    # sosfiltfilt applies the filter forwards, then backwards. The two passes
    # cancel each other's phase shift, so wave shapes and peak timing are
    # preserved - important when a delay would move the R peak of an ECG or the
    # systolic peak of a PPG. The cost is that it is non-causal, so it suits
    # offline processing of a complete recording rather than live streaming.
    #
    # It also needs enough samples to pad both ends: the padding length for this
    # filter works out to 27 samples (3 * (2 * 4 sections + 1)), so anything
    # shorter raises a ValueError rather than silently returning a distorted
    # result. That is the floor for a minimum-recording-length check, and the
    # real minimum should be far higher for a meaningful signal.
    return sosfiltfilt(sos, signal)
