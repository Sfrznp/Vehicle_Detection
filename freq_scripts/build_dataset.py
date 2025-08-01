import os
import numpy as np
import argparse
import pandas as pd
from utils import load_numpy_array, save_numpy_array


PROCESSED_DIR = "data/processed"
LABELS_CSV = "data/Vehicles_Data.csv"
VEHICLE_LABELS = {
    'PCR': 0,   # Passenger Car (Regular sedan, coupe, etc.)
    'SUV': 1,   # Sport Utility Vehicle (larger passenger vehicle with off-road capabilities)
    'HT': 2,    # Heavy Truck (large freight trucks, often multi-axle)
    'PT': 3,    # Pickup Truck (light truck with open cargo area)
    'VAN': 4,   # Van (used for transporting people or goods, larger than a car)
    'PTT': 5,   # Pickup Truck with Trailer (pickup towing a trailer)
}

NUM_CLASSES = len(VEHICLE_LABELS)


def parse_labels_csv(csv_path):
    df = pd.read_csv(csv_path)
    parsed = {}

    for _, row in df.iterrows():
        fname = os.path.splitext(row['File Name'].strip())[0]

        if pd.isna(row['Timestamps(s)']) or pd.isna(row['Vehicle Types']):
            continue

        # Split timestamp and type fields
        raw_times = str(row['Timestamps(s)']).replace("'", "").replace(',', ';').split(';')
        raw_types = str(row['Vehicle Types']).replace("'", "").replace(',', ';').split(';')

        timestamps = [float(t.strip()) for t in raw_times if t.strip() != '']
        types = [t.strip().split('+') for t in raw_types if t.strip() != '']

        events = []
        for i in range(min(len(timestamps), len(types))):
            vehicle_ids = [VEHICLE_LABELS.get(v.strip(), -1) for v in types[i] if v.strip() in VEHICLE_LABELS]
            if vehicle_ids:
                events.append((timestamps[i], vehicle_ids))

        parsed[fname] = events

    return parsed


def assign_multilabels(timestamps, car_events):
    y_types = np.zeros((len(timestamps), NUM_CLASSES), dtype=int)
    y_counts = np.zeros(len(timestamps), dtype=int)

    for i, t in enumerate(timestamps):
        for event_time, labels in car_events:
            if abs(t - event_time) <= 0.55:
                for label in labels:
                    y_types[i][label] = 1
                y_counts[i] += len(labels)

    return y_types, y_counts


def process_file(base_filename, label_map):
    print(f"Building dataset from {base_filename}...")
    x_path = os.path.join(PROCESSED_DIR, f"{base_filename}_x.npy")
    t_path = os.path.join(PROCESSED_DIR, f"{base_filename}_timestamps.npy")

    x = load_numpy_array(x_path)
    timestamps = load_numpy_array(t_path)

    car_events = label_map.get(base_filename, [])
    y_types, y_counts = assign_multilabels(timestamps, car_events)

    print(f"  Spectrograms: {x.shape}, Labels: {y_types.shape}, Counts: {y_counts.sum()} total cars")

    save_numpy_array(x, os.path.join(PROCESSED_DIR, f"{base_filename}_x_labeled.npy"))
    save_numpy_array(y_types, os.path.join(PROCESSED_DIR, f"{base_filename}_y_types.npy"))
    save_numpy_array(y_counts, os.path.join(PROCESSED_DIR, f"{base_filename}_y_counts.npy"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, help="Base filename without extension")
    args = parser.parse_args()

    label_map = parse_labels_csv(LABELS_CSV)

    if args.file:
        process_file(args.file, label_map)
    else:
        for fname in os.listdir(PROCESSED_DIR):
            if fname.endswith("_x.npy"):
                base = fname.replace("_x.npy", "")
                process_file(base, label_map)
