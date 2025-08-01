import os
import argparse
import numpy as np
from utils import read_tdms_channel, get_windows, compute_spectrogram, save_numpy_array, get_sampling_rate
import glob

RAW_DIR = "data"
PROCESSED_DIR = "data/processed"
GROUP_NAME = "Measurement"  
CHANNEL_NAME = "739"        
WINDOW_SIZE = 1.0          
STRIDE = 1.0             


def process_file(tdms_path):
    print(f"Processing {tdms_path}...")
    signal, channel = read_tdms_channel(tdms_path, GROUP_NAME, CHANNEL_NAME)
    fs = get_sampling_rate(channel)
    print(f"  Detected sampling rate: {fs:.2f} Hz")

    windows, timestamps = get_windows(signal, fs, window_size_sec=WINDOW_SIZE, stride_sec=STRIDE)

    spectrograms = []
    for window in windows:
        spec = compute_spectrogram(window, fs)
        spectrograms.append(spec)

    spectrograms = np.array(spectrograms)  # shape: (n_windows, freq_bins, time_bins)
    filename = os.path.splitext(os.path.basename(tdms_path))[0]

    save_numpy_array(spectrograms, os.path.join(PROCESSED_DIR, f"{filename}_x.npy"))
    save_numpy_array(timestamps, os.path.join(PROCESSED_DIR, f"{filename}_timestamps.npy"))
    print(f"  Saved {len(spectrograms)} spectrograms.")


if __name__ == "__main__":

    # Clear all previous outputs
    for f in glob.glob("data/processed/*.npy"):
        os.remove(f)

    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default=None, help="Path to a single TDMS file (optional)")
    args = parser.parse_args()

    if args.file:
        process_file(args.file)
    else:
        for fname in os.listdir(RAW_DIR):
            if fname.endswith(".tdms"):
                process_file(os.path.join(RAW_DIR, fname))



