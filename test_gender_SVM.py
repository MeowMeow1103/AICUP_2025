import os
import numpy as np
import pandas as pd
from scipy.fft import fft, fftfreq
from scipy.stats import skew, kurtosis
from scipy.spatial.transform import Rotation as R
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier


# === Section 1: Augmentation Utilities ===
def augment_rotation(accel, gyro):
    rot = R.random()
    return rot.apply(accel), rot.apply(gyro)

def augment_bias(gyro, bias_std=0.5):
    b = np.random.normal(0, bias_std, size=(1,3))
    return gyro + b

def augment_noise(accel, gyro, accel_noise_std=0.02, gyro_noise_std=0.15):
    n_acc = np.random.normal(0, accel_noise_std, size=accel.shape)
    n_gyro = np.random.normal(0, gyro_noise_std, size=gyro.shape)
    return accel + n_acc, gyro + n_gyro

# === Section 2: Feature Extraction Helpers ===
FS = 85.0  # Sampling frequency

def load_signal(txt_path):
    return np.loadtxt(txt_path)

def extract_time_features(seg6):
    feats = []
    # Time-domain features: mean, variance, RMS for each axis
    for i in range(seg6.shape[1]):
        vals = seg6[:, i]
        feats += [vals.mean(), vals.var(), np.sqrt((vals**2).mean())]
    # Overall magnitude statistics for accel and gyro
    for cols in [slice(0,3), slice(3,6)]:
        arr = np.linalg.norm(seg6[:, cols], axis=1)
        feats += [arr.mean(), arr.max(), arr.min(), skew(arr), kurtosis(arr)]
    return np.array(feats)

def extract_freq_features(seg6):
    feats = []
    n = seg6.shape[0]
    freqs = fftfreq(n, d=1/FS)
    # Compute features for acceleration and gyro magnitudes separately
    for cols in [slice(0,3), slice(3,6)]:
        arr = np.linalg.norm(seg6[:, cols], axis=1)
        # FFT and power spectral density
        Y = np.abs(fft(arr))
        psd = (Y**2)[freqs > 0]
        freqs_pos = freqs[freqs > 0]
        sum_psd = psd.sum()
        # Normalized power spectrum
        p_norm = psd / (sum_psd + 1e-12)
        # Spectral entropy
        entropy = -np.sum(p_norm * np.log2(p_norm + 1e-12))
        # Spectral centroid
        centroid = np.sum(freqs_pos * psd) / (sum_psd + 1e-12)
        # Append the three features
        feats += [sum_psd, entropy, centroid]
    return np.array(feats)  # returns 6 features in total

# === Section 3: Feature Matrix Builder ===
def build_feature_matrix(info_df, data_dir, label_col='gender', augment=False): #hold racket handed
    df = pd.read_csv(info_df) if isinstance(info_df, str) else info_df
    has_label = label_col in df.columns
    feats_list, labels = [], []
    for _, row in df.iterrows():
        sig6 = load_signal(os.path.join(data_dir, f"{row['unique_id']}.txt"))
        cut_points = np.fromstring(row['cut_point'].strip('[]'), sep=' ').astype(int)
        seg_feats = []
        for i in range(len(cut_points)-1):
            seg = sig6[cut_points[i]:cut_points[i+1], :6]
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



if __name__ == '__main__':
    # Paths
    train_csv = './39_Training_Dataset/train_info.csv'
    train_dir = './39_Training_Dataset/train_data'
    test_csv  = './39_Test_Dataset/test_info.csv'
    test_dir  = './39_Test_Dataset/test_data'

    # Build train features & labels
    X_train_all, y_train_all = build_feature_matrix(train_csv, train_dir, augment=False)
    lbl_enc = LabelEncoder().fit(y_train_all)
    y_enc = lbl_enc.transform(y_train_all)

    # Split train/validation
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_all, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    # Standardize
    scaler = StandardScaler().fit(X_tr)
    X_tr_s = scaler.transform(X_tr)
    X_val_s = scaler.transform(X_val)

    # --- Hyperparameter tuning for SVM ---
    param_grid = {
        'C': [0.1, 1, 10],
        'gamma': ['scale', 0.01, 0.1, 1],
        'kernel': ['rbf']
    }
    grid = GridSearchCV(
        SVC(probability=True, random_state=42),
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
    grid.fit(X_tr_s, y_tr)
    best_params = grid.best_params_
    print(f"Best SVM params: {best_params}")

    # Train final SVM
    svm = SVC(
        C=best_params['C'],
        gamma=best_params['gamma'],
        kernel=best_params['kernel'],
        probability=True,
        random_state=42
    )
    svm.fit(X_tr_s, y_tr)

    # 驗證集評估
    y_pred = svm.predict(X_val_s)
    print("Validation Accuracy:", accuracy_score(y_val, y_pred))
    print(classification_report(y_val, y_pred))
    print(confusion_matrix(y_val, y_pred))

    # 建立測試集特徵
    X_test, _ = build_feature_matrix(test_csv, test_dir, label_col='gender', augment=False)
    X_test_s = scaler.transform(X_test)

    # 測試集機率預測
    probs = svm.predict_proba(X_test_s)

    # 取回原始類別順序
    classes = list(lbl_enc.classes_)

    # 讀取 unique_id
    test_ids = pd.read_csv(test_csv)['unique_id']

    # 組成 submission
    submission = pd.DataFrame({
        'unique_id': test_ids,
        **{f'prob_{cls}': probs[:, i] for i, cls in enumerate(classes)}
    })

    submission.to_csv('submission_gender_probs_svm.csv', index=False, float_format='%.4f')

