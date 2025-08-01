# import os
# import numpy as np
# from tensorflow.keras.models import load_model
# from utils import read_tdms_channel, generate_spectrogram_windows, get_sampling_rate

# # === Configuration ===
# TEST_DIR = "testFiles"
# MODEL_PATH = "models/vehicle_classifier.h5"
# CHANNEL_NAME = "739"
# VEHICLE_CLASSES = ["PCR", "SUV", "HT", "PT", "VAN", "PTT"]
# WINDOW_SIZE = 1.0  # seconds
# STEP_SIZE = 1.0    # seconds
# THRESHOLD = 0.25
# DEFAULT_SAMPLING_RATE = 2000

# model = load_model(MODEL_PATH)

# def process_file(file_path):
#     print(f"\nProcessing {os.path.basename(file_path)}...")

#     # Read signal
#     signal, channel = read_tdms_channel(file_path, channel_name=CHANNEL_NAME)
#     sampling_rate = get_sampling_rate(channel)

#     print("  Signal length:", len(signal))
    
#     # Generate spectrogram windows
#     spectrograms, timestamps = generate_spectrogram_windows(
#         signal,
#         sampling_rate=sampling_rate,
#         window_size=WINDOW_SIZE,
#         step_size=STEP_SIZE,
#         nperseg=256,
#         noverlap=128
#     )

#     print(f"  Extracted {len(spectrograms)} spectrogram windows")

#     if len(spectrograms) == 0:
#         print("  Skipped: No spectrograms extracted (file too short?)")
#         return


#     # Use same normalization as training: divide by max
#     # Apply the same log scaling as training
#     X = 10 * np.log10(spectrograms + 1e-10)

#     # Normalize using training global max (approx from log values)
#     X = X.astype("float32") / 100.0 

#     X = np.expand_dims(X, axis=-1)

#     print("  After log10: min:", np.min(X), "max:", np.max(X), "mean:", np.mean(X))


#     # Predict
#     y_pred = model.predict(X)
#     print("  Raw prediction shape:", y_pred.shape)
#     print("  First few predictions:", y_pred[:5])

#     # Thresholding (multi-label)
#     predicted_events = []
#     for i, row in enumerate(y_pred):
#         labels = [VEHICLE_CLASSES[j] for j, val in enumerate(row) if val >= THRESHOLD]
#         if labels:
#             predicted_events.append((timestamps[i], labels))

#     print(f"  Total detections: {len(predicted_events)}")
#     for ts, types in predicted_events:
#         print(f"    {ts:.2f}s — {', '.join(types)}")

#     return predicted_events


# if __name__ == "__main__":
#     print(f" Files being evaluated from: {TEST_DIR}")
#     for fname in sorted(os.listdir(TEST_DIR)):
#         if fname.endswith(".tdms"):
#             process_file(os.path.join(TEST_DIR, fname))


import os
import numpy as np
from tensorflow.keras.models import load_model
from utils import read_tdms_channel, generate_spectrogram_windows, get_sampling_rate
from plot_vehicle_events import plot_vehicle_events

# === Configuration ===
TEST_DIR = "testFiles"
MODEL_PATH = "models/vehicle_classifier.h5"
CHANNEL_NAME = "739"
VEHICLE_CLASSES = ["PCR", "SUV", "HT", "PT", "VAN", "PTT"]
WINDOW_SIZE = 1.0 
STEP_SIZE = 0.25    
THRESHOLD = 0.25
DEFAULT_SAMPLING_RATE = 2000

model = load_model(MODEL_PATH)

def group_vehicle_events(timestamps, y_pred, threshold=THRESHOLD):
    events = []
    current_labels = set()
    enter_time = None

    for i in range(len(y_pred)):
        active_labels = {VEHICLE_CLASSES[j] for j, v in enumerate(y_pred[i]) if v >= threshold}

        if active_labels and not current_labels:
            # Start of a new vehicle event
            enter_time = timestamps[i]
            current_labels = active_labels

        elif not active_labels and current_labels:
            # End of the event
            leave_time = timestamps[i - 1] + WINDOW_SIZE
            events.append((enter_time, leave_time, current_labels))
            current_labels = set()
            enter_time = None

        elif active_labels:
            # Ongoing event — combine label types
            current_labels.update(active_labels)

    # Handle case where file ends while a vehicle is still present
    if current_labels and enter_time is not None:
        leave_time = timestamps[-1] + WINDOW_SIZE
        events.append((enter_time, leave_time, current_labels))

    return events


def process_file(file_path):
    print(f"\nProcessing {os.path.basename(file_path)}...")

    # Read signal
    signal, channel = read_tdms_channel(file_path, channel_name=CHANNEL_NAME)

    try:
        sampling_rate = get_sampling_rate(channel)
    except Exception as e:
        print(f"Warning: using default sampling rate {DEFAULT_SAMPLING_RATE} Hz. Error: {e}")
        sampling_rate = DEFAULT_SAMPLING_RATE

    print("  Signal length:", len(signal))
    
    # Generate spectrogram windows
    spectrograms, timestamps = generate_spectrogram_windows(
        signal,
        sampling_rate=sampling_rate,
        window_size=WINDOW_SIZE,
        step_size=STEP_SIZE,
        nperseg=256,
        noverlap=128
    )

    print(f"  Extracted {len(spectrograms)} spectrogram windows")
    if len(spectrograms) == 0:
        print("  Skipped: No spectrograms extracted (file too short?)")
        return

    # Apply log scaling and normalize (match training)
    X = 10 * np.log10(spectrograms + 1e-10)
    X = X.astype("float32") / 100.0
    X = np.expand_dims(X, axis=-1)

    print(f"  After log10: min: {np.min(X):.6f} max: {np.max(X):.6f} mean: {np.mean(X):.6f}")

    # Predict
    y_pred = model.predict(X)
    # print("  Raw prediction shape:", y_pred.shape)
    # print("  First few predictions:", y_pred[:5])

    # Group into vehicle events with entering and leaving times
    vehicle_events = group_vehicle_events(timestamps, y_pred, threshold=THRESHOLD)

    print(f"  Total vehicle events: {len(vehicle_events)}")
    for enter, leave, types in vehicle_events:
        type_str = ", ".join(sorted(types))
        print(f"    Enter: {enter:.2f}s — Leave: {leave:.2f}s ") # Add "— {type_str}" to get the type as well

    return vehicle_events


if __name__ == "__main__":
    print(f"\n Files being evaluated from: {TEST_DIR}")
    for fname in sorted(os.listdir(TEST_DIR)):
        if fname.endswith(".tdms"):
            vehicle_events = process_file(os.path.join(TEST_DIR, fname))
            plot_vehicle_events(fname, vehicle_events)
