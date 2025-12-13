!git clone https://github.com/MahdiFarvardin/MEDVSE.git

import os, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers

DATA_PATH = './MEDVSE/MTHS/Data'
FS_SIGNAL = 30

WINDOW_SEC = 10
STRIDE_SEC = 1
WINDOW = FS_SIGNAL * WINDOW_SEC
STRIDE = FS_SIGNAL * STRIDE_SEC

TEST_SIZE = 0.2
SEED = 42

F_LOW, F_HIGH = 0.7, 3.0

np.random.seed(SEED)
tf.random.set_seed(SEED)

print("GPU available:", tf.config.list_physical_devices('GPU'))

"""data loading"""

def load_mths_data(path):
    subjects = []
    for sig_file in sorted(glob.glob(os.path.join(path, 'signal_*.npy'))):
        pid = os.path.basename(sig_file).split('_')[1].split('.')[0]
        lbl_file = os.path.join(path, f'label_{pid}.npy')
        if not os.path.exists(lbl_file):
            continue

        sig = np.load(sig_file)
        lbl = np.load(lbl_file)

        if sig.ndim == 2 and sig.shape[0] == 3:
            sig = sig.T

        subjects.append({"id": pid, "signal": sig, "label": lbl})
    return subjects

subjects = load_mths_data(DATA_PATH)
print("Loaded subjects:", len(subjects))

"""data inspection"""

for s in subjects[:3]:
    print(f"ID={s['id']} | signal={s['signal'].shape} | label={s['label'].shape}")

durations = [
    min(len(s["signal"])//FS_SIGNAL, len(s["label"]))
    for s in subjects
]
pd.Series(durations).describe()

"""data processing"""

def normalize_subject_signal(sig):
    mu = sig.mean(axis=0, keepdims=True)
    sd = sig.std(axis=0, keepdims=True) + 1e-8
    return (sig - mu) / sd

def make_windows(sig, lbl, sid):
    if sig.ndim != 2 or lbl.ndim != 2:
        return None
    if sig.shape[1] != 3:
        sig = sig.T

    sig = np.clip(sig, 0, 255).astype(np.float32)
    hr = lbl[:,0].astype(np.float32)

    total_sec = min(len(sig)//FS_SIGNAL, len(hr))
    if total_sec < WINDOW_SEC:
        return None

    sig = normalize_subject_signal(sig[:total_sec*FS_SIGNAL])
    hr = hr[:total_sec]
    hr_base = hr.mean()

    X, y_delta, y_abs, base = [], [], [], []
    for st in range(0, len(sig)-WINDOW+1, STRIDE):
        en = st + WINDOW
        w = sig[st:en]
        s0 = st//FS_SIGNAL
        s1 = s0 + WINDOW_SEC
        y = hr[s0:s1].mean()

        X.append(w)
        y_abs.append(y)
        y_delta.append(y - hr_base)
        base.append(hr_base)

    return {
        "X": np.stack(X),
        "y_delta": np.array(y_delta),
        "y_abs": np.array(y_abs),
        "base": np.array(base),
        "g": np.array([sid]*len(y_delta))
    }

def build_dataset(subjects):
    Xs, yd, ya, b, g = [], [], [], [], []
    for s in subjects:
        out = make_windows(s["signal"], s["label"], s["id"])
        if out is None: continue
        Xs.append(out["X"])
        yd.append(out["y_delta"])
        ya.append(out["y_abs"])
        b.append(out["base"])
        g.append(out["g"])

    return (
        np.concatenate(Xs),
        np.concatenate(yd),
        np.concatenate(ya),
        np.concatenate(b),
        np.concatenate(g)
    )

X, y_delta, y_abs, base, groups = build_dataset(subjects)
print("Final X:", X.shape)

"""eda"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def perform_eda_subjectwise(subjects, fs_signal=30, sample_subject_idx=0, sample_second=0):
    print("--- Exploratory Data Analysis ---")

    hr_all = []
    spo2_all = []

    for s in subjects:
        lbl = np.asarray(s["label"])
        if lbl.ndim == 2 and lbl.shape[1] >= 2:
            hr_all.append(lbl[:, 0])
            spo2_all.append(lbl[:, 1])

    hr_all = np.concatenate(hr_all)
    spo2_all = np.concatenate(spo2_all)

    plt.figure(figsize=(10, 4))
    sns.histplot(hr_all, kde=True, color='red')
    plt.title('Distribution of Heart Rates (Ground Truth)')
    plt.xlabel('BPM')
    plt.show()

    s = subjects[sample_subject_idx]
    sig = np.asarray(s["signal"])
    lbl = np.asarray(s["label"])

    if sig.ndim == 2 and sig.shape[0] == 3:
        sig = sig.T

    start = sample_second * fs_signal
    end = start + fs_signal

    if sig.ndim != 2 or sig.shape[1] != 3 or end > len(sig):
        print("Cannot plot RGB window — check signal shape or length.")
    else:
        window = sig[start:end]

        hr_value = lbl[sample_second, 0] if sample_second < len(lbl) else np.nan

        plt.figure(figsize=(10, 4))
        plt.plot(window[:, 0], color='r', label='Red')
        plt.plot(window[:, 1], color='g', label='Green')
        plt.plot(window[:, 2], color='b', label='Blue')
        plt.title(f'RGB Signal for 1 Second (Label HR: {hr_value:.1f})')
        plt.xlabel('Frame')
        plt.ylabel('Raw RGB')
        plt.legend()
        plt.show()

    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=hr_all, y=spo2_all)
    plt.title('Correlation: HR vs SpO2')
    plt.xlabel('Heart Rate')
    plt.ylabel('SpO2')
    plt.show()

perform_eda_subjectwise(
    subjects,
    fs_signal=FS_SIGNAL,
    sample_subject_idx=0,
    sample_second=0
)

"""x and y"""

gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SEED)
tr, te = next(gss.split(X, y_delta, groups))

Xtr, Xte = X[tr], X[te]
yd_tr, yd_te = y_delta[tr], y_delta[te]
ya_tr, ya_te = y_abs[tr], y_abs[te]
b_tr, b_te = base[tr], base[te]
g_te = groups[te]

print("Train subjects:", len(np.unique(groups[tr])))
print("Test subjects:", len(np.unique(groups[te])))

"""feature eng."""

def bandpower_features(X):
    feats = []
    freqs = np.fft.rfftfreq(X.shape[1], d=1/FS_SIGNAL)
    band = (freqs>=F_LOW)&(freqs<=F_HIGH)

    for w in X:
        mu = w.mean(axis=0)
        sd = w.std(axis=0)
        fft = np.abs(np.fft.rfft(w, axis=0))[band]
        bp = fft.mean(axis=0)
        peak = freqs[band][fft.argmax(axis=0)]
        feats.append(np.concatenate([mu, sd, bp, peak]))
    return np.array(feats)

Ftr = bandpower_features(Xtr)
Fte = bandpower_features(Xte)

"""model 1: random forest"""

from sklearn.model_selection import learning_curve
from sklearn.metrics import mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt

rf_baseline = RandomForestRegressor(
    n_estimators=600,
    max_depth=18,
    min_samples_leaf=8,
    random_state=SEED,
    n_jobs=-1
)

train_sizes, train_scores, val_scores = learning_curve(
    rf_baseline,
    Ftr,
    yd_tr,
    cv=3,
    scoring="neg_mean_absolute_error",
    train_sizes=np.linspace(0.1, 1.0, 5),
    n_jobs=-1
)

train_mae = -np.mean(train_scores, axis=1)
val_mae = -np.mean(val_scores, axis=1)


plt.figure(figsize=(10, 5))
plt.plot(train_sizes, train_mae, 'o-', color='red', label='Training Error (MAE)')
plt.plot(train_sizes, val_mae, 'o-', color='green', label='Validation Error (MAE)')
plt.title("Baseline Learning Curve (RandomForest)")
plt.xlabel("Number of Training Samples")
plt.ylabel("Mean Absolute Error (BPM)")
plt.legend()
plt.grid(True)
plt.show()
baseline_mae = mean_absolute_error(ya_te, pred_rf)

print("----- RandomForest Baseline Performance -----")
print(f"Baseline Test MAE: {baseline_mae:.2f} BPM")

"""extra trees"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.ensemble import ExtraTreesRegressor


et_baseline = ExtraTreesRegressor(
    n_estimators=800,
    max_depth=18,
    min_samples_leaf=6,
    random_state=SEED,
    n_jobs=-1
)

train_sizes = np.linspace(0.1, 1.0, 8)

ts_mse, tr_mse, va_mse = learning_curve(
    et_baseline, Ftr, yd_tr,
    cv=3,
    scoring="neg_mean_squared_error",
    train_sizes=train_sizes,
    n_jobs=-1
)
tr_mse_mean = -tr_mse.mean(axis=1)
va_mse_mean = -va_mse.mean(axis=1)

tr_rmse_mean = np.sqrt(tr_mse_mean)
va_rmse_mean = np.sqrt(va_mse_mean)

ts_r2, tr_r2, va_r2 = learning_curve(
    et_baseline, Ftr, yd_tr,
    cv=3,
    scoring="r2",
    train_sizes=train_sizes,
    n_jobs=-1
)
tr_r2_mean = tr_r2.mean(axis=1)
va_r2_mean = va_r2.mean(axis=1)

plt.figure(figsize=(18, 5))

plt.subplot(1, 3, 1)
plt.plot(ts_mse, tr_mse_mean, 'o-', color='red', label='Train Loss')
plt.plot(ts_mse, va_mse_mean, 'o-', color='green', label='Val Loss')
plt.title("ExtraTrees Model: Loss (MSE)")
plt.xlabel("Training Samples")
plt.ylabel("MSE")
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(ts_mse, tr_rmse_mean, 'o-', color='red', label='Train RMSE')
plt.plot(ts_mse, va_rmse_mean, 'o-', color='green', label='Val RMSE')
plt.title("ExtraTrees Model: RMSE")
plt.xlabel("Training Samples")
plt.ylabel("RMSE")
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 3)
plt.plot(ts_r2, tr_r2_mean, 'o-', color='red', label='Train R2')
plt.plot(ts_r2, va_r2_mean, 'o-', color='green', label='Val R2')
plt.title("ExtraTrees Model: R2 Score")
plt.xlabel("Training Samples")
plt.ylabel("R2")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

"""model 3 tcn"""

import tensorflow as tf
import tensorflow.keras.backend as K

def r2_metric(y_true, y_pred):
    ss_res = K.sum(K.square(y_true - y_pred))
    ss_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1.0 - ss_res / (ss_tot + K.epsilon())

def rmse_metric(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_true - y_pred)))

tcn.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="mse",
    metrics=[rmse_metric, r2_metric]
)

history_tcn = tcn.fit(
    Xtr, yd_tr_s,
    validation_split=0.1,
    epochs=80,
    batch_size=256,
    callbacks=[tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)],
    verbose=1
)

import matplotlib.pyplot as plt

def plot_tcn_history_like_extratrees(history, model_name="TCN Model"):
    h = history.history
    epochs = range(1, len(h["loss"]) + 1)

    plt.figure(figsize=(18, 5))

    plt.subplot(1, 3, 1)
    plt.plot(epochs, h["loss"], 'o-', color='red', label="Train Loss")
    plt.plot(epochs, h["val_loss"], 'o-', color='green', label="Val Loss")
    plt.title(f"{model_name}: Loss (MSE)")
    plt.xlabel("Epochs")
    plt.ylabel("MSE")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(epochs, h["rmse_metric"], 'o-', color='red', label="Train RMSE")
    plt.plot(epochs, h["val_rmse_metric"], 'o-', color='green', label="Val RMSE")
    plt.title(f"{model_name}: RMSE")
    plt.xlabel("Epochs")
    plt.ylabel("RMSE")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.plot(epochs, h["r2_metric"], 'o-', color='red', label="Train R2")
    plt.plot(epochs, h["val_r2_metric"], 'o-', color='green', label="Val R2")
    plt.title(f"{model_name}: R2 Score")
    plt.xlabel("Epochs")
    plt.ylabel("R2")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

plot_tcn_history_like_extratrees(history_tcn, "TCN (fast)")

"""metrics"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

models = ["RandomForest", "ExtraTrees", "TCN"]
colors = ["#6CCF7D", "#F2C94C", "#F2994A"]

r2_scores = [
    r2_score(ya_te, pred_rf),
    r2_score(ya_te, pred_et),
    r2_score(ya_te, pred_tcn)
]

mse_scores = [
    mean_squared_error(ya_te, pred_rf),
    mean_squared_error(ya_te, pred_et),
    mean_squared_error(ya_te, pred_tcn)
]

rmse_scores = [np.sqrt(m) for m in mse_scores]

mae_scores = [
    mean_absolute_error(ya_te, pred_rf),
    mean_absolute_error(ya_te, pred_et),
    mean_absolute_error(ya_te, pred_tcn)
]

plt.figure(figsize=(14, 10))

plt.subplot(2, 2, 1)
plt.bar(models, r2_scores, color=colors)
plt.title("R² Score Comparison (Higher is Better)")
plt.ylabel("R²")
plt.ylim(0, 1)
plt.grid(axis="y")
for i, v in enumerate(r2_scores):
    plt.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontweight="bold")

plt.subplot(2, 2, 2)
plt.bar(models, rmse_scores, color=colors)
plt.title("RMSE Comparison (Lower is Better)")
plt.ylabel("RMSE (BPM)")
plt.grid(axis="y")
for i, v in enumerate(rmse_scores):
    plt.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontweight="bold")

plt.subplot(2, 2, 3)
plt.bar(models, mae_scores, color=colors)
plt.title("MAE Comparison (Lower is Better)")
plt.ylabel("MAE (BPM)")
plt.grid(axis="y")
for i, v in enumerate(mae_scores):
    plt.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontweight="bold")

plt.subplot(2, 2, 4)
plt.bar(models, mse_scores, color=colors)
plt.title("MSE Comparison (Lower is Better)")
plt.ylabel("MSE")
plt.grid(axis="y")
for i, v in enumerate(mse_scores):
    plt.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontweight="bold")

plt.tight_layout()
plt.show()
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np

metrics = {
    "RandomForest": {
        "R2": r2_score(ya_te, pred_rf),
        "MAE": mean_absolute_error(ya_te, pred_rf),
        "MSE": mean_squared_error(ya_te, pred_rf)
    },
    "ExtraTrees": {
        "R2": r2_score(ya_te, pred_et),
        "MAE": mean_absolute_error(ya_te, pred_et),
        "MSE": mean_squared_error(ya_te, pred_et)
    },
    "TCN": {
        "R2": r2_score(ya_te, pred_tcn),
        "MAE": mean_absolute_error(ya_te, pred_tcn),
        "MSE": mean_squared_error(ya_te, pred_tcn)
    }
}

for m in metrics.values():
    m["RMSE"] = np.sqrt(m["MSE"])

print("\n========== FINAL TEST METRICS ==========\n")

for model, vals in metrics.items():
    print(f"{model}")
    print(f"  R2   : {vals['R2']:.3f}")
    print(f"  MAE  : {vals['MAE']:.2f} BPM")
    print(f"  RMSE : {vals['RMSE']:.2f} BPM")
    print(f"  MSE  : {vals['MSE']:.2f}")
    print("-" * 40)

"""Application"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

class HeartRateMonitor:
    def __init__(self, model, model_type='rf', fs=30):
        self.model = model
        self.model_type = model_type
        self.fs = fs
        self.scaler = StandardScaler()

    def extract_signal_from_video(self, video_path):

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return None

        raw_signal = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            h, w, _ = frame_rgb.shape
            min_dim = min(h, w)
            roi_size = int(min_dim * 0.5)

            center_x, center_y = w // 2, h // 2
            x1 = center_x - roi_size // 2
            x2 = center_x + roi_size // 2
            y1 = center_y - roi_size // 2
            y2 = center_y + roi_size // 2

            roi_frame = frame_rgb[y1:y2, x1:x2]

            avg_color = np.mean(roi_frame, axis=(0, 1))
            raw_signal.append(avg_color)

        cap.release()
        return np.array(raw_signal)

    def preprocess_signal(self, raw_signal):
        signal = np.clip(raw_signal, 0, 255)
        signal_scaled = self.scaler.fit_transform(signal)
        return signal_scaled

    def segment_signal(self, signal):
        num_seconds = len(signal) // self.fs
        windows = []
        for i in range(num_seconds):
            start_idx = i * self.fs
            end_idx = start_idx + self.fs
            window = signal[start_idx:end_idx, :]
            windows.append(window)
        return np.array(windows)

    def predict(self, video_path):
        print(f"Processing video: {video_path}...")

        raw_signal = self.extract_signal_from_video(video_path)
        if raw_signal is None or len(raw_signal) == 0:
            print("No signal extracted.")
            return

        print(f"Video duration: {len(raw_signal)/self.fs:.1f} seconds")

        clean_signal = self.preprocess_signal(raw_signal)

        X_windows = self.segment_signal(clean_signal)
        if len(X_windows) == 0:
            print("Video too short (needs at least 1 second).")
            return

        if self.model_type == 'rf':
            X_input = X_windows.reshape(X_windows.shape[0], -1)
            predictions = self.model.predict(X_input)
        else:
            X_input = X_windows
            predictions = self.model.predict(X_input, verbose=0)

        if predictions.ndim > 1 and predictions.shape[1] >= 2:
            hr_values = predictions[:, 0]
            spo2_values = predictions[:, 1]
        else:
            print("Model output format not recognized. Returning only HR if available.")
            hr_values = predictions[:, 0] if predictions.ndim > 1 else predictions
            spo2_values = np.zeros_like(hr_values)

        self.plot_results(hr_values, spo2_values)

        return hr_values, spo2_values

    def plot_results(self, hr_values, spo2_values):
        time_axis = np.arange(len(hr_values))

        plt.figure(figsize=(12, 8))

        plt.subplot(2, 1, 1)
        plt.plot(time_axis, hr_values, 'o-', color='r', label='Predicted HR')
        avg_hr = np.mean(hr_values)
        plt.axhline(y=avg_hr, color='darkred', linestyle='--', alpha=0.5, label=f'Avg HR: {avg_hr:.1f}')
        plt.ylabel('BPM')
        plt.title('Heart Rate (BPM) vs Time')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(time_axis, spo2_values, 'o-', color='b', label='Predicted SpO2')
        avg_spo2 = np.mean(spo2_values)
        plt.axhline(y=avg_spo2, color='darkblue', linestyle='--', alpha=0.5, label=f'Avg SpO2: {avg_spo2:.1f}%')
        plt.ylabel('SpO2 (%)')
        plt.title('Oxygen Saturation (SpO2) vs Time')
        plt.xlabel('Time (seconds)')
        plt.ylim(80, 100)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()

        plt.tight_layout()
        plt.show()

        print(f"Summary -> Avg HR: {avg_hr:.1f} BPM | Avg SpO2: {avg_spo2:.1f}%")


app = HeartRateMonitor(model=rf_model, model_type='rf', fs=30)
hr, spo2 = app.predict('video.mp4')
