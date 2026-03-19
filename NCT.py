import numpy as np
import torch
import torch.nn as nn
from scipy.signal import resample
from scipy.signal import butter, filtfilt, iirnotch


TARGET_FS = 200
TARGET_CHANNELS = 16
WINDOW_MS = 250
STRIDE_MS = 50
WINDOW_LEN = int(WINDOW_MS / 1000 * TARGET_FS)
STRIDE_LEN = int(STRIDE_MS / 1000 * TARGET_FS)
EPS = 1e-8


def bandpass_filter(data, low=5, high=500, fs=2000, order=3):
    nyq = 0.5 * fs
    low = low / nyq
    high = min(high / nyq, 0.99)
    b, a = butter(order, [low, high], btype='bandpass')
    return filtfilt(b, a, data, axis=0)


def notch_filter(data, f0=50, Q=30, fs=2000):
    w0 = f0 / (0.5 * fs)
    b, a = iirnotch(w0, Q)
    return filtfilt(b, a, data, axis=0)


def channel_map_to_16(X, num_channels, dataset_name='generic'):
    C, T = X.shape
    if C == TARGET_CHANNELS:
        return X
    elif C < TARGET_CHANNELS:
        mapped = np.zeros((TARGET_CHANNELS, T), dtype=np.float32)
        for i in range(TARGET_CHANNELS):
            src_idx = int(i * C / TARGET_CHANNELS) % C
            mapped[i] = X[src_idx]
        return mapped
    else:
        mapped = np.zeros((TARGET_CHANNELS, T), dtype=np.float32)
        for i in range(TARGET_CHANNELS):
            start = int(i * C / TARGET_CHANNELS)
            end = int((i + 1) * C / TARGET_CHANNELS)
            if end > start:
                mapped[i] = np.mean(X[start:end], axis=0)
            else:
                mapped[i] = X[min(start, C - 1)]
        return mapped


class NCT(nn.Module):
    def __init__(self, fs, max_len=256, token_len=50, target_channels=16, 
                 stride=None, used_channels=None):
        super().__init__()
        self.fs = fs
        self.max_len = max_len
        self.token_len = token_len
        self.target_channels = target_channels
        self.stride = stride if stride is not None else max(1, token_len // 5)
        self.used_channels = used_channels

    def _resample_to_200hz(self, X):
        C, T = X.shape
        if self.fs == TARGET_FS:
            return X
        new_T = int(T * TARGET_FS / self.fs)
        resampled = np.zeros((C, new_T), dtype=np.float32)
        for c in range(C):
            resampled[c] = resample(X[c], new_T)
        return resampled

    def _compute_energy(self, window):
        return np.mean(window ** 2)

    def _adaptive_threshold(self, X, calibration_ratio=0.1):
        C, T = X.shape
        calib_len = int(T * calibration_ratio)
        calib = X[:, :calib_len]
        energies = []
        for t in range(0, calib_len - self.token_len, self.stride):
            w = calib[:, t:t + self.token_len]
            energies.append(self._compute_energy(w))
        if len(energies) == 0:
            return 0.0
        energies = np.array(energies)
        theta = np.mean(energies) + 0.5 * np.std(energies)
        return max(theta, 1e-6)

    def _extract_tokens(self, X, theta):
        C, T = X.shape
        tokens = []
        in_chans = []
        in_times = []
        t = 0
        token_idx = 0
        while t + self.token_len <= T:
            window = X[:, t:t + self.token_len]
            Ew = self._compute_energy(window)
            if Ew > theta:
                seg = window.copy()
                for c in range(seg.shape[0]):
                    mu = np.mean(seg[c])
                    sigma = np.std(seg[c]) + EPS
                    seg[c] = (seg[c] - mu) / sigma
                tokens.append(seg)
                channel_energies = np.mean(seg ** 2, axis=1)
                dom_ch = np.argmax(channel_energies) + 1
                in_chans.append(dom_ch)
                in_times.append(min(token_idx + 1, 15))
                token_idx += 1
                t += self.stride
            else:
                t += self.stride
        return tokens, in_chans, in_times

    def process_signal(self, X, subject_id=None, calibration_theta=None):
        X = self._resample_to_200hz(X)
        X = channel_map_to_16(X, X.shape[0])
        C, T = X.shape
        theta = calibration_theta if calibration_theta is not None else self._adaptive_threshold(X)
        tokens, in_chans, in_times = self._extract_tokens(X, theta)
        if len(tokens) == 0:
            sentence = np.zeros((self.max_len, self.target_channels, self.token_len), dtype=np.float32)
            in_chan_matrix = np.zeros(self.max_len, dtype=np.int32)
            in_time_matrix = np.zeros(self.max_len, dtype=np.int32)
            return sentence, in_chan_matrix, in_time_matrix
        tokens = np.array(tokens)
        token_dim = self.target_channels * self.token_len
        seq = tokens.reshape(len(tokens), -1)
        in_chans = np.array(in_chans)
        in_times = np.array(in_times)
        if len(seq) < self.max_len:
            pad_len = self.max_len - len(seq)
            seq = np.concatenate([seq, np.zeros((pad_len, token_dim), dtype=np.float32)], axis=0)
            in_chans = np.concatenate([in_chans, np.zeros(pad_len, dtype=np.int32)], axis=0)
            in_times = np.concatenate([in_times, np.zeros(pad_len, dtype=np.int32)], axis=0)
        else:
            seq = seq[:self.max_len]
            in_chans = in_chans[:self.max_len]
            in_times = in_times[:self.max_len]
        return seq.astype(np.float32), in_chans.astype(np.int32), in_times.astype(np.int32)

    def forward(self, x, calibration_theta=None):
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
        bs = x.shape[0]
        batch_sentences = []
        batch_in_chans = []
        batch_in_times = []
        for i in range(bs):
            sent, ic, it = self.process_signal(x[i], calibration_theta=calibration_theta)
            batch_sentences.append(sent)
            batch_in_chans.append(ic)
            batch_in_times.append(it)
        return (
            np.array(batch_sentences, dtype=np.float32),
            np.array(batch_in_chans, dtype=np.int32),
            np.array(batch_in_times, dtype=np.int32)
        )


def nct_process_offline(data_dir, output_dir, dataset_name='MCS_EMG', fs=2000):
    import os
    from tqdm import tqdm
    nct = NCT(fs=fs, max_len=256, token_len=WINDOW_LEN, target_channels=TARGET_CHANNELS)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir
