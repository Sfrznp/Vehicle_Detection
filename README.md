
# Vehicle Detection from TDMS Files

This project detects and analyzes vehicle activity using TDMS sensor data. It includes scripts for preprocessing, spectrogram extraction, model training, inference, and evaluation.

---

## Project Structure

```
.
├── build_dataset.py           # Builds dataset windows from preprocessed spectrograms
├── eval_report.py             # Runs vehicle detection on test TDMS files and reports events
├── extract_spectrograms.py    # Converts TDMS signals to spectrograms (log-mel)
├── inspect_tdms.py            # Inspect contents of TDMS files
├── plot_vehicle_events.py     # Optional: plots detected vehicle entry/exit events
├── train_model.py             # Trains CNN model on labeled spectrogram data
├── utils.py                   # Shared helper functions
├── testFiles/                 # Folder containing TDMS files for testing
└── model/                     # Folder to store trained model weights
```

---

## Setup Instructions

1. **Clone the repository**

```bash
git clone <your-repo-url>
cd <your-project-folder>
```

2. **Install dependencies**

Make sure you have Python 3.8+ installed. Then install required packages:

```bash
pip install -r requirements.txt
```

If `requirements.txt` does not exist, typical dependencies include:

```bash
pip install numpy scipy pandas matplotlib librosa tensorflow nptdms
```

---

## Pipeline Overview

### 1. Inspect TDMS File (Optional)

```bash
python inspect_tdms.py --file path/to/file.tdms
```

This will print metadata about the TDMS channels and length.

---

### 2. Extract Spectrograms from TDMS Files

```bash
python extract_spectrograms.py --input_dir raw_tdms/ --output_dir spectrograms/
```

This converts raw TDMS time series data into log-mel spectrograms.

---

### 3. Build Dataset from Spectrograms

```bash
python build_dataset.py --spectrogram_dir spectrograms/ --labels_csv labels.csv --output_dir dataset/
```

This script slices spectrograms into model-ready labeled windows.

---

### 4. Train the Model

```bash
python train_model.py --data_dir dataset/ --save_dir model/
```

Trains a 1D or 2D CNN on your dataset and saves model weights in `model/`.

---

### 5. Run Evaluation on Test Files

```bash
python eval_report.py --test_dir testFiles/ --model_path model/trained_model.h5 --output_csv vehicle_data.csv
```

This will:

- Load the trained model
- Predict vehicle events on each file in `testFiles/`
- Output a CSV file with vehicle entry and exit timestamps for each file

---

### 6. (Optional) Plot Events

```bash
python plot_vehicle_events.py --input_csv vehicle_data.csv
```

Visualizes detected vehicle events over time.

---

## Notes

- All TDMS files must be synchronized in structure (same channel count and length).
- Default sampling rate is assumed if not available in metadata.
- Files are grouped or windowed in fixed time intervals (configurable in scripts).
- Adjust model architecture or parameters inside `train_model.py` if needed.
