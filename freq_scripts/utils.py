import numpy as np
import scipy.signal
import nptdms
import os
from scipy.signal import spectrogram


VEHICLE_CLASSES = ["PCR", "SUV", "HT", "PT", "VAN", "PTT"]

import numpy as np
import scipy.signal

def generate_spectrogram_windows(
    signal, 
    sampling_rate, 
    window_size=1.0, 
    step_size=1.0, 
    nperseg=256, 
    noverlap=128
):
    """
    Converts time-series signal into overlapping spectrogram windows.

    Args:
        signal (np.ndarray): The raw signal.
        sampling_rate (int): Sampling rate of the signal.
        window_size (float): Window size in seconds.
        step_size (float): Step size in seconds between windows.
        nperseg (int): FFT segment size.
        noverlap (int): Overlap between segments.

    Returns:
        Tuple of (windows: np.ndarray, timestamps: np.ndarray)
    """
    window_samples = int(window_size * sampling_rate)
    step_samples = int(step_size * sampling_rate)

    spectrograms = []
    timestamps = []

    for start in range(0, len(signal) - window_samples + 1, step_samples):
        end = start + window_samples
        segment = signal[start:end]

        freqs, times, Sxx = scipy.signal.spectrogram(
            segment, 
            fs=sampling_rate, 
            nperseg=nperseg, 
            noverlap=noverlap, 
            scaling='spectrum', 
            mode='magnitude'
        )

        spectrograms.append(Sxx)
        timestamps.append(start / sampling_rate)

    return np.array(spectrograms), np.array(timestamps)


def read_tdms_channel(tdms_path, group_name="Measurement", channel_name="739"):
    tdms_file = nptdms.TdmsFile.read(tdms_path)
    group = tdms_file[group_name]
    channel = group[channel_name]
    signal = channel.data.astype(np.float32)
    return signal, channel


def get_sampling_rate(channel):
    try:
        time_track = channel.time_track()
        if len(time_track) >= 2:
            dt = time_track[1] - time_track[0]
            return 1.0 / dt
        else:
            raise ValueError("Insufficient time points for sampling rate calculation.")
    except Exception as e:
        print(f"Warning: failed to get sampling rate, defaulting to 2000 Hz. Error: {e}")
        return 2000.0


def get_windows(signal, fs, window_size_sec=1.0, stride_sec=1.0):
    window_size = int(fs * window_size_sec)
    stride = int(fs * stride_sec)
    windows = []
    timestamps = []
    for start in range(0, len(signal) - window_size + 1, stride):
        end = start + window_size
        windows.append(signal[start:end])
        timestamps.append(start / fs)
    return np.array(windows), np.array(timestamps)


def compute_spectrogram(signal, fs, nperseg=256, noverlap=128):
    f, t, Sxx = scipy.signal.spectrogram(signal, fs=fs, nperseg=nperseg, noverlap=noverlap)
    Sxx_log = 10 * np.log10(Sxx + 1e-10)
    return Sxx_log  # shape: (freq_bins, time_bins)


def save_numpy_array(array, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, array)


def load_numpy_array(path):
    return np.load(path)


def normalize_spectrogram(spec, mean=None, std=None):
    if mean is None:
        mean = np.mean(spec)
    if std is None:
        std = np.std(spec)
    return (spec - mean) / (std + 1e-8)
