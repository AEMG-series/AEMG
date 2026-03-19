import os
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import bisect
import json
import scipy.io
from scipy.signal import butter, filtfilt, iirnotch, resample

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from NCT import NCT, TARGET_FS, WINDOW_LEN, TARGET_CHANNELS


def bandpass_filter(X, low=5, high=500, sfreq=2000, order=3):
    nyq = 0.5 * sfreq
    low_n = low / nyq
    high_n = min(high / nyq, 0.99)
    b, a = butter(order, [low_n, high_n], btype='bandpass')
    return filtfilt(b, a, X, axis=1)


def notch_filter(X, f0=50, Q=30, sfreq=2000):
    w0 = f0 / (0.5 * sfreq)
    b, a = iirnotch(w0, Q)
    return filtfilt(b, a, X, axis=1)


def preprocess_emg(X, sfreq=2000):
    if X.shape[1] < 100:
        return X
    X = bandpass_filter(X, 1, 150, sfreq)
    for f in [50, 100, 150, 200]:
        if f < sfreq / 2:
            X = notch_filter(X, f, 30, sfreq)
    return X


def load_MCS_EMG_raw(data_root, subject_ids=None):
    data_root = Path(data_root)
    raw_dir = data_root / "MCS_EMG" / "sEMG-dataset" / "raw" / "csv"
    if not raw_dir.exists():
        raw_dir = data_root / "sEMG-dataset" / "raw" / "csv"
    all_signals = []
    all_subject_ids = []
    subject_ids = subject_ids or range(1, 41)
    for sid in subject_ids:
        fpath = raw_dir / f"{sid}_raw.csv"
        if not fpath.exists():
            continue
        data = np.loadtxt(fpath, delimiter=',', dtype=np.float32)
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        if data.shape[0] > data.shape[1]:
            data = data.T
        if data.shape[1] < 500:
            continue
        all_signals.append(preprocess_emg(data, 2000))
        all_subject_ids.append(sid)
    return all_signals, all_subject_ids


def load_UCI_EMG_raw(data_root, subject_ids=None):
    data_root = Path(data_root)
    uci_dir = data_root / "uciEMG"
    if not uci_dir.exists():
        return [], []
    all_signals = []
    all_subject_ids = []
    for d in sorted(uci_dir.iterdir()):
        if not d.is_dir():
            continue
        try:
            sid = int(d.name)
        except ValueError:
            continue
        if subject_ids and sid not in subject_ids:
            continue
        for f in d.glob("*.txt"):
            data = np.loadtxt(f, dtype=np.float32, skiprows=1)
            if data.ndim == 1:
                continue
            emg = data[:, 1:-1].T
            if emg.shape[0] < 4 or emg.shape[1] < 500:
                continue
            all_signals.append(preprocess_emg(emg, 1000))
            all_subject_ids.append(sid)
            break
    return all_signals, all_subject_ids


class EMGPretrainDataset(Dataset):
    def __init__(self, data_list, nct, subject_ids=None):
        self.sentences = []
        self.in_chans = []
        self.in_times = []
        for i, X in enumerate(data_list):
            if X.shape[0] < 2 or X.shape[1] < 100:
                continue
            sent, ic, it = nct.process_signal(X)
            self.sentences.append(sent)
            self.in_chans.append(ic)
            self.in_times.append(it)
        self.sentences = np.array(self.sentences, dtype=np.float32)
        self.in_chans = np.array(self.in_chans, dtype=np.int32)
        self.in_times = np.array(self.in_times, dtype=np.int32)

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.sentences[idx]),
            torch.from_numpy(self.in_chans[idx]),
            torch.from_numpy(self.in_times[idx])
        )


class EMGPretrainDatasetFromFolder(Dataset):
    def __init__(self, folder_paths, stage='train'):
        if isinstance(folder_paths, (str, Path)):
            folder_paths = [Path(folder_paths)]
        self.folder_paths = [Path(p) for p in folder_paths]
        self.stage = stage
        self.data = []
        self.in_chans = []
        self.in_times = []
        for folder in self.folder_paths:
            d_path = folder / f"{stage}_data.npy"
            c_path = folder / f"{stage}_data_in_chans.npy"
            t_path = folder / f"{stage}_data_in_times.npy"
            if d_path.exists():
                self.data.append(np.load(d_path))
                self.in_chans.append(np.load(c_path))
                self.in_times.append(np.load(t_path))
        if self.data:
            self.data = np.concatenate(self.data, axis=0)
            self.in_chans = np.concatenate(self.in_chans, axis=0)
            self.in_times = np.concatenate(self.in_times, axis=0)
        else:
            self.data = np.zeros((0, 256, 800), dtype=np.float32)
            self.in_chans = np.zeros((0, 256), dtype=np.int32)
            self.in_times = np.zeros((0, 256), dtype=np.int32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.data[idx]),
            torch.from_numpy(self.in_chans[idx]),
            torch.from_numpy(self.in_times[idx])
        )


def _chunk_long_signals(signals, chunk_len_samples=20000, overlap_ratio=0.5):
    chunked = []
    step = max(1, int(chunk_len_samples * (1 - overlap_ratio)))
    for X in signals:
        C, T = X.shape
        if T <= chunk_len_samples:
            chunked.append(X)
            continue
        for start in range(0, T - chunk_len_samples + 1, step):
            chunked.append(X[:, start:start + chunk_len_samples].copy())
    return chunked


def build_aemg_pretrain_dataset(data_root, nct, datasets=['MCS_EMG', 'UCI'], 
                                train_ratio=0.9, seed=0, chunk_len=20000, use_chunks=True):
    np.random.seed(seed)
    all_signals = []
    for ds in datasets:
        if ds == 'MCS_EMG':
            sigs, _ = load_MCS_EMG_raw(data_root)
        elif ds == 'UCI':
            sigs, _ = load_UCI_EMG_raw(data_root)
        else:
            continue
        if use_chunks:
            cl = chunk_len if ds == 'MCS_EMG' else chunk_len // 2
            sigs = _chunk_long_signals(sigs, chunk_len_samples=cl)
        all_signals.extend(sigs)
    if not all_signals:
        raise FileNotFoundError(f"No data found in {data_root}. Check data_aemg structure.")
    idx = np.random.permutation(len(all_signals))
    split = int(len(idx) * train_ratio)
    train_sigs = [all_signals[i] for i in idx[:split]]
    val_sigs = [all_signals[i] for i in idx[split:]]
    train_ds = EMGPretrainDataset(train_sigs, nct)
    val_ds = EMGPretrainDataset(val_sigs, nct)
    return train_ds, val_ds


def load_ninapro_db4_for_aemg(folder_path, nct, sfreq=2000):
    import sys
    proj_root = Path(__file__).resolve().parents[1]
    if str(proj_root) not in sys.path:
        sys.path.insert(0, str(proj_root))
    from run_tests.testingEnvironment.dataLoader import load_ninaproDB4
    X, Y = load_ninaproDB4(folder_path)
    nct.fs = sfreq
    results = []
    for s in range(len(X)):
        xs = X[s]
        ys = Y[s]
        sentences = []
        in_chans_list = []
        in_times_list = []
        for i in range(len(xs)):
            sent, ic, it = nct.process_signal(xs[i])
            sentences.append(sent)
            in_chans_list.append(ic)
            in_times_list.append(it)
        results.append({
            'X': np.array(sentences, dtype=np.float32),
            'y': ys.astype(np.int64),
            'in_chans': np.array(in_chans_list, dtype=np.int32),
            'in_times': np.array(in_times_list, dtype=np.int32)
        })
    return results


def load_toro_ossaba_for_aemg(folder_path, nct, sfreq=1024):
    import sys
    proj_root = Path(__file__).resolve().parents[1]
    if str(proj_root) not in sys.path:
        sys.path.insert(0, str(proj_root))
    from run_tests.testingEnvironment.dataLoader import load_ToroOssaba
    X, Y = load_ToroOssaba(folder_path)
    orig_fs = nct.fs
    nct.fs = sfreq
    results = []
    for s in range(len(X)):
        xs = X[s]
        ys = Y[s]
        sentences = []
        in_chans_list = []
        in_times_list = []
        for i in range(len(xs)):
            sent, ic, it = nct.process_signal(np.asarray(xs[i], dtype=np.float32))
            sentences.append(sent)
            in_chans_list.append(ic)
            in_times_list.append(it)
        results.append({
            'X': np.array(sentences, dtype=np.float32),
            'y': np.array(ys, dtype=np.int64),
            'in_chans': np.array(in_chans_list, dtype=np.int32),
            'in_times': np.array(in_times_list, dtype=np.int32)
        })
    nct.fs = orig_fs
    return results


def load_emg_epn_for_aemg(folder_path, nct, sfreq=1000):
    import sys
    proj_root = Path(__file__).resolve().parents[1]
    if str(proj_root) not in sys.path:
        sys.path.insert(0, str(proj_root))
    from run_tests.testingEnvironment.dataLoader import load_EMG_EPN_612
    X, Y = load_EMG_EPN_612(folder_path)
    orig_fs = nct.fs
    nct.fs = sfreq
    results = []
    for s in range(len(X)):
        xs = X[s]
        ys = Y[s]
        sentences = []
        in_chans_list = []
        in_times_list = []
        for i in range(len(xs)):
            sent, ic, it = nct.process_signal(np.asarray(xs[i], dtype=np.float32))
            sentences.append(sent)
            in_chans_list.append(ic)
            in_times_list.append(it)
        results.append({
            'X': np.array(sentences, dtype=np.float32),
            'y': np.array(ys, dtype=np.int64),
            'in_chans': np.array(in_chans_list, dtype=np.int32),
            'in_times': np.array(in_times_list, dtype=np.int32)
        })
    nct.fs = orig_fs
    return results
