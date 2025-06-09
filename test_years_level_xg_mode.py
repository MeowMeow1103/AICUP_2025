import os
import numpy as np
import pandas as pd
from scipy.fft import fft, fftfreq
from scipy.stats import skew, kurtosis
from scipy.spatial.transform import Rotation as R
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

# === Section 1: Augmentation Utilities ===
def augment_rotation(accel, gyro):
    rot = R.random()
    return rot.apply(accel), rot.apply(gyro)

def augment_bias(gyro, bias_std=0.5):
    b = np.random.normal(0, bias_std, size=(1, 3))
    return gyro + b

def augment_noise(accel, gyro, accel_noise_std=0.02, gyro_noise_std=0.15):
    n_acc = np.random.normal(0, accel_noise_std, size=accel.shape)
    n_gyro = np.random.normal(0, gyro_noise_std, size=gyro.shape)
    return accel + n_acc, gyro + n_gyro

# === Section 2: Feature Extraction ===
FS = 85.0  # Sampling frequency

def load_signal(txt_path):
    return np.loadtxt(txt_path)

def extract_time_features(seg6):
    feats = []
    for i in range(seg6.shape[1]):
        vals = seg6[:, i]
        feats += [vals.mean(), vals.var(), np.sqrt((vals**2).mean())]
    for cols in [slice(0, 3), slice(3, 6)]:
        arr = np.linalg.norm(seg6[:, cols], axis=1)
        feats += [arr.mean(), arr.max(), arr.min(), skew(arr), kurtosis(arr)]
    return np.array(feats)

def extract_freq_features(seg6):
    feats = []
    n = seg6.shape[0]
    freqs = fftfreq(n, d=1 / FS)
    for cols in [slice(0, 3), slice(3, 6)]:
        arr = np.linalg.norm(seg6[:, cols], axis=1)
        Y = np.abs(fft(arr))
        psd = (Y ** 2)[freqs > 0]
        freqs_pos = freqs[freqs > 0]
        sum_psd = psd.sum()
        p_norm = psd / (sum_psd + 1e-12)
        entropy = -np.sum(p_norm * np.log2(p_norm + 1e-12))
        centroid = np.sum(freqs_pos * psd) / (sum_psd + 1e-12)
        feats += [sum_psd, entropy, centroid]
    return np.array(feats)

# === Section 3: Feature Matrix Builder ===
def build_feature_matrix(df, data_dir, label_col='level', augment=False, n_augments=1):
    has_label = label_col in df.columns
    feats_list, labels = [], []
    for _, row in df.iterrows():
        file_path = os.path.join(data_dir, f"{row['unique_id']}.txt")
        if not os.path.exists(file_path):
            continue
        sig6 = load_signal(file_path)
        cut_points = np.fromstring(row['cut_point'].strip('[]'), sep=' ').astype(int)

        for i_aug in range(n_augments if augment else 1):  # ← 關鍵處
            seg_feats = []
            for i in range(len(cut_points) - 1):
                seg = sig6[cut_points[i]:cut_points[i + 1], :6]
                if augment:
                    accel, gyro = augment_rotation(seg[:, :3], seg[:, 3:6])
                    accel, gyro = augment_noise(accel, gyro)
                    gyro = augment_bias(gyro)
                    seg = np.hstack([accel, gyro])
                tf = extract_time_features(seg)
                ff = extract_freq_features(seg)
                seg_feats.append(np.hstack([tf, ff]))
            feats_list.append(np.mean(seg_feats, axis=0))
            if has_label:
                labels.append(row[label_col])

    X = np.vstack(feats_list)
    y = np.array(labels) if has_label else None
    return X, y


# === Main Execution ===
if __name__ == '__main__':
    train_csv = './39_Training_Dataset/train_info_34.csv'
    test_csv = './39_Test_Dataset/test_info.csv'
    train_dir = './39_Training_Dataset/train_data_34'
    test_dir = './39_Test_Dataset/test_data'
    label_col = 'level'

    train_info = pd.read_csv(train_csv)
    test_info = pd.read_csv(test_csv)
    submission = []

    for mode in sorted(train_info['mode'].unique()):
        print(f"\n🔧 Training for mode {mode}...")

        train_mode_df = train_info[train_info['mode'] == mode]
        test_mode_df = test_info[test_info['mode'] == mode]

        if len(train_mode_df) == 0 or len(test_mode_df) == 0:
            continue

        # 訓練
        # X_train, y_train = build_feature_matrix(train_mode_df, train_dir, label_col)
        augment = (mode <= 10)
        n_augments = 3 if augment else 1
        X_train, y_train = build_feature_matrix(train_mode_df, train_dir, label_col, augment=augment, n_augments=n_augments)

        le = LabelEncoder().fit(y_train)
        y_train_enc = le.transform(y_train)

        X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train_enc, test_size=0.2, stratify=y_train_enc, random_state=42)
        scaler = StandardScaler().fit(X_tr)
        X_tr_s = scaler.transform(X_tr)
        X_val_s = scaler.transform(X_val)

        model = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, eval_metric='mlogloss', random_state=42)
        model.fit(X_tr_s, y_tr)

        val_acc = accuracy_score(y_val, model.predict(X_val_s))
        print(f"✅ Mode {mode} validation accuracy: {val_acc:.4f}")

        # 測試資料預測
        X_test, _ = build_feature_matrix(test_mode_df, test_dir, label_col)
        X_test_s = scaler.transform(X_test)
        probs = model.predict_proba(X_test_s)

        mode_submission = pd.DataFrame({
            'unique_id': test_mode_df['unique_id'].values,
            **{f'prob_{cls}': probs[:, i] for i, cls in enumerate(le.classes_)}
        })
        submission.append(mode_submission)

    # 合併結果
    final_df = pd.concat(submission).sort_values('unique_id')
    final_df.to_csv('submission_level_probs_by_mode.csv', index=False, float_format='%.4f')
