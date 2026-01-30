# Network-to-Image DDoS Classifier — Detailed README

## Purpose
This project converts network flow/tabular data (CSV format, e.g. `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv`) into image representations and trains a Convolutional Neural Network (CNN) to classify DDoS vs benign/DDoS traffic. Converting tabular network data into images lets CNNs learn spatial patterns that may correspond to temporal or multi-feature correlations in flows. You can create **one image per flow** or **multiple images per flow** (several strategies described below).

### Conversion strategies (how to convert CSV network data into images)
1. **Single-vector to fixed-size image (default simple method)**
   - Select numeric features for each flow (packet counts, byte counts, durations, flags, statistics).
   - Sanitize features: replace infinities, fill NaNs, clip extreme percentiles (e.g., 1st–99th) to reduce outliers.
   - Standardize (zero mean, unit variance) per feature across dataset.
   - For each flow (row), make a feature vector. If vector length < 1024, zero-pad; if > 1024 truncate.
   - Reshape length-1024 vector into a 32×32 grayscale image (values scaled to 0–255). Save PNG.
   - Pros: simple, deterministic, easy to parallelize. Cons: loses explicit temporal order; choice of features matters.

2. **Time-bin matrix (recommended for temporal patterns)**
   - Choose a time window length or packet-count window for each flow; split into `T` bins (e.g., 16 or 32).
   - For each bin compute features (packet count, sum bytes, avg packet size, flag counts) producing `F` features per bin.
   - Build a `T x F` matrix (e.g., 32×32) for each flow and save as an image. This preserves temporal progression across bins.
   - Pros: preserves time progression and per-bin feature dynamics; CNN can learn time-feature correlations.

3. **Multi-image / multi-channel approach**
   - Produce multiple images per flow: e.g. one image for packet-size distribution, one for inter-arrival times, another for flag counts.
   - Combine them as channels (RGB) or train multi-input models that ingest each image separately.
   - Pros: flexible, can encode different aspects of flows without forcing them into one vector.

4. **Sequence-based transforms (advanced)**
   - Use recurrence plots, Gramian Angular Fields (GAF), or Markov transition matrices computed over sequences of a chosen feature (e.g., packet size, inter-arrival time).
   - These methods convert 1D time series into images that highlight temporal recurrence and patterns.

### Which method to use?
- Use **time-bin matrix** or **multi-image** when you want to preserve temporal structure (recommended for DDoS detection).
- Use **single-vector** when you have many engineered features per flow and need a quick baseline.

## Files in this package
- `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv` — the primary CSV input (expected filename).
- `generate_images.py` — script to convert the CSV into images and `dataset_images.npz` index.
- `train.py` — training script for a CNN that loads images and trains a Keras model.
- `pcap_images/` — folder where generated PNG images will be stored.
- `dataset_images.npz` — index file (filenames, labels, class names) produced by `generate_images.py`.

## Detailed explanation of each script and how it works

### 1) `generate_images.py` — full walkthrough
Purpose: read the CSV, sanitize numeric data, convert rows (or per-flow time-bins) into image files, and save an index.

Key steps inside the script:
1. **Load CSV**
   - The script expects `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv` in the working directory.
   - If that file is missing, it will attempt to find similarly-named variants (robustness).

2. **Detect label column**
   - The script scans header names for keywords `label`, `attack`, or `class` (case-insensitive).
   - If a label column is found, it encodes textual labels into integer classes using `sklearn.preprocessing.LabelEncoder`.

3. **Select numeric features**
   - Only numeric columns (integers/floats) are selected for image conversion.
   - This avoids IP address strings or other non-numeric columns polluting the vector.

4. **Sanitize data**
   - Replace `inf` / `-inf` with `NaN`, then fill `NaN` with 0.
   - Clip each numeric column to [1st percentile, 99th percentile] to reduce outlier influence.
   - Optionally, you may instead use robust scaling (e.g., `RobustScaler`) if heavy-tailed distributions exist.

5. **Standardize**
   - Use `StandardScaler` (zero mean, unit variance) fitted on the numeric matrix so all features are comparable.

6. **Vector → image mapping**
   - Default mapping: create a length `L = size*size` vector (size=32 → L=1024). For each flow:
     - If the feature vector length < L, zero-pad at the end.
     - If feature vector length > L, truncate (select first L features) — or alternatively apply PCA to reduce dimensionality before mapping.
     - Normalize per-row to range 0–255 and reshape to `size x size` as a uint8 image.
   - Alternative mappings (recommended): time-bin `T x F` matrix, GAF, recurrence plots (see conversion strategies above).

7. **Saving**
   - Save PNG files under `pcap_images/` and write `dataset_images.npz` containing:
     - `filenames` — array of image path strings,
     - `labels` — integer labels (or zeros if unknown),
     - `class_names` — mapping of label integers to original class strings.

8. **Parameters**
   - `--size`: image width/height (default 32).
   - `--max-rows`: limit how many rows to convert (default 2000 to avoid big operations on modest machines).
   - `--mode`: `vector` (default) or `timebin` (if implemented).

### 2) `train.py` — full walkthrough
Purpose: load images and labels from `dataset_images.npz`, build a CNN with Keras, train, and save the model.

Key parts:
1. **Loading dataset**
   - Loads `dataset_images.npz`. If it is missing, the script suggests running `generate_images.py`.
   - Loads PNG images with `imageio.v2.imread`, converts to float32 and scales to `[0,1]`.

2. **Label processing**
   - Uses one-hot encoding (Keras `to_categorical`) for training with `categorical_crossentropy`.
   - If classes are imbalanced, the script suggests using class weights or oversampling.

3. **Model architecture**
   - Example small CNN:
     - Conv2D(32) → MaxPool → Conv2D(64) → MaxPool → Flatten → Dense(128) → Dropout(0.5) → Dense(num_classes, softmax)
   - This is a baseline; for better performance use:
     - Deeper networks (ResNet family), global average pooling, batch normalization, and learning-rate schedules.
     - Transfer learning with pretrained backbones (treat PNGs as 3-channel by duplicating grayscale into RGB).

4. **Training**
   - Uses `model.fit` with validation split (or an explicit `train_test_split`).
   - Suggests early stopping (monitor `val_loss`) and saving the best model.

5. **Saving**
   - Saves final model as `ddos_cnn_model.h5`.

## How to run (practical steps)
1. Ensure Python 3.8+ is installed.
2. Create a virtual environment and install:
   ```
   pip install numpy pandas scikit-learn imageio pillow
   pip install tensorflow   # optional, required for training
   ```
3. Put `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv` in the working directory.
4. Generate images:
   ```
   python generate_images.py --size 32 --max-rows 5000
   ```
   (Adjust `--max-rows` or omit it to convert all rows — be mindful of disk space.)
5. Train:
   ```
   python train.py
   ```

## Notes, tips, and advanced ideas
- **Balancing classes**: For DDoS detection, classes are often highly imbalanced. Use `class_weight` in `model.fit` or oversample minority classes.
- **Feature selection**: Try domain knowledge: packet counts, flow duration, mean/min/max packet size, flag counts, and entropy of payload sizes are often informative.
- **Cross-validation**: Use k-fold cross-validation across flows/time to estimate generalization robustly.
- **Evaluation**: Use precision, recall, F1-score, and ROC-AUC (not just accuracy) because the dataset is often imbalanced.
- **Deployment**: For real-time detection, consider models that can operate on sliding windows over streaming packet features.

_______________________________________________________________

Student: Tornike Ebralidze
