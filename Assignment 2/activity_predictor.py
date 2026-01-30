"""
activity_predictor.py
---------------------------
Task 2: Predict the type of network activity (Label2) using Darknet.csv.

Features:
- Safe preprocessing: removes inf/NaN, caps extreme values
- Encodes all categorical columns (including IPs)
- Scales numeric features with StandardScaler
- Uses RandomForestClassifier with class_weight="balanced"
- Prints detailed accuracy and classification report
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from collections import Counter


# Load Dataset
print("Loading dataset...")
df = pd.read_csv("Darknet.csv")
print("Original shape:", df.shape)

# Drop unnecessary columns if present
df = df.drop(columns=["Flow ID", "Timestamp"], errors="ignore")

# Encode categorical features
print("Encoding categorical columns...")
for c in df.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    df[c] = le.fit_transform(df[c].astype(str))

print("All object columns encoded.")

# Handle numeric issues (inf, NaN, large values)
print("Cleaning numeric data...")

# Replace infinities with NaN
df = df.replace([np.inf, -np.inf], np.nan)

# Fill missing with median values
df = df.fillna(df.median(numeric_only=True))

# Cap extremely large outliers
for col in df.select_dtypes(include=[np.number]).columns:
    df[col] = df[col].clip(lower=-1e12, upper=1e12)

print("Cleaned. Remaining inf:", np.isinf(df.values).sum(), "NaN:", np.isnan(df.values).sum())

# Prepare Data
target = "Label2"
if target not in df.columns:
    raise ValueError(f"Target column '{target}' not found in dataset!")

X = df.drop(columns=[target])
y = df[target]

print("Feature shape:", X.shape)
print("Target distribution:", Counter(y))

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Scale numeric data
print("Scaling features...")
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print("Scaling done.")

# Train RandomForest
print("Training RandomForest model...")
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)

rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Test Accuracy: {accuracy:.3f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))