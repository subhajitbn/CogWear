import numpy as np
import scipy.stats
from scipy.signal import find_peaks


def extract_bvp_features(signal, top_k=3):
    mean = np.mean(signal)
    std = np.std(signal)
    rms = np.sqrt(np.mean(signal**2))
    skew = scipy.stats.skew(signal)
    kurt = scipy.stats.kurtosis(signal)

    peak_mean, peak_std = top_k_peak_height_stats(signal, k=top_k)

    return [
        mean, std, rms, skew, kurt,
        peak_mean, peak_std
    ]


def top_k_peak_height_stats(signal, k=3):
    peaks, _ = find_peaks(signal)
    if len(peaks) == 0:
        return 0.0, 0.0

    peak_heights = signal[peaks]
    top_k_heights = np.sort(peak_heights)[-k:] if len(peak_heights) >= k else peak_heights
    return np.mean(top_k_heights), np.std(top_k_heights)
