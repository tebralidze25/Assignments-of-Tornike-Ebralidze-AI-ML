import argparse, numpy as np, pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from PIL import Image

def main(args):
    csv_path = Path(args.csv)
    if not csv_path.exists():
        # try fallback variants
        alt = list(Path(csv_path.parent).glob(csv_path.stem + "*"))
        if alt:
            csv_path = alt[0]
            print("Using fallback CSV:", csv_path)
        else:
            raise FileNotFoundError(f"CSV not found: {args.csv}")
    df = pd.read_csv(csv_path, low_memory=False)
    possible_label_cols = [c for c in df.columns if 'label' in c.lower() or 'attack' in c.lower() or 'class' in c.lower()]
    label_col = possible_label_cols[0] if possible_label_cols else None
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    print("Numeric columns:", len(numeric_cols))
    numeric_df = df[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0).copy()

    # Clip percentiles
    for c in numeric_df.columns:
        col = numeric_df[c].values
        lo = np.percentile(col, 1)
        hi = np.percentile(col, 99)
        if hi - lo > 0:
            numeric_df[c] = np.clip(col, lo, hi)

    numeric_data = numeric_df.values.astype('float32')
    scaler = StandardScaler()
    scaler.fit(numeric_data)
    scaled = scaler.transform(numeric_data)

    if label_col:
        le = LabelEncoder()
        labels = le.fit_transform(df[label_col].astype(str).fillna("none"))
        class_names = list(le.classes_)
    else:
        labels = np.zeros(len(df), dtype=int)
        class_names = ["unknown"]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    size = args.size
    L = size * size
    filenames = []
    y = []
    max_rows = min(args.max_rows if args.max_rows>0 else len(df), len(df))
    for i in range(max_rows):
        vec = scaled[i]
        v = vec.copy()
        if v.size < L:
            v = np.concatenate([v, np.zeros(L - v.size)])
        elif v.size > L:
            v = v[:L]
        v_min, v_max = v.min(), v.max()
        if v_max - v_min < 1e-6:
            img_arr = np.zeros(L, dtype=np.uint8)
        else:
            img_arr = ((v - v_min) / (v_max - v_min) * 255).astype(np.uint8)
        img = img_arr.reshape((size, size))
        fname = out_dir / f"img_{i:05d}.png"
        Image.fromarray(img).save(fname)
        filenames.append(str(fname))
        y.append(int(labels[i]))
        if (i+1) % 500 == 0:
            print(f"Created {i+1} images...")
    np.savez_compressed("dataset_images.npz", filenames=np.array(filenames), labels=np.array(y), class_names=np.array(class_names))
    print("Done. Saved", len(filenames), "images and dataset_images.npz")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default='Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv')
    parser.add_argument('--size', type=int, default=32)
    parser.add_argument('--max-rows', type=int, default=1500)
    parser.add_argument('--out-dir', default='pcap_images')
    args = parser.parse_args()
    main(args)