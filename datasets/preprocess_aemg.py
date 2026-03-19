import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from NCT import NCT, WINDOW_LEN, TARGET_CHANNELS
try:
    from .data_loader import load_MCS_EMG_raw, load_UCI_EMG_raw
except ImportError:
    from data_loader import load_MCS_EMG_raw, load_UCI_EMG_raw


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default="./data_aemg")
    p.add_argument("--output_dir", type=str, default="./data_aemg_processed")
    p.add_argument("--datasets", type=str, nargs="+", default=["MCS_EMG", "UCI"])
    p.add_argument("--train_ratio", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    np.random.seed(args.seed)
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    nct_2000 = NCT(fs=2000, max_len=256, token_len=WINDOW_LEN, target_channels=TARGET_CHANNELS)
    nct_1000 = NCT(fs=1000, max_len=256, token_len=WINDOW_LEN, target_channels=TARGET_CHANNELS)

    all_data = []
    all_in_chans = []
    all_in_times = []

    for ds in args.datasets:
        if ds == "MCS_EMG":
            sigs, _ = load_MCS_EMG_raw(data_root)
            nct = nct_2000
        elif ds == "UCI":
            sigs, _ = load_UCI_EMG_raw(data_root)
            nct = nct_1000
        else:
            print(f"Unknown dataset {ds}, skip")
            continue

        print(f"Processing {ds}: {len(sigs)} signals")
        for X in tqdm(sigs, desc=ds):
            if X.shape[0] < 2 or X.shape[1] < 100:
                continue
            sent, ic, it = nct.process_signal(X)
            all_data.append(sent)
            all_in_chans.append(ic)
            all_in_times.append(it)

    if not all_data:
        print("No data processed. Check data_root and dataset paths.")
        return

    all_data = np.array(all_data, dtype=np.float32)
    all_in_chans = np.array(all_in_chans, dtype=np.int32)
    all_in_times = np.array(all_in_times, dtype=np.int32)

    n = len(all_data)
    idx = np.random.permutation(n)
    split = int(n * args.train_ratio)
    train_idx, val_idx = idx[:split], idx[split:]

    for stage, indices in [("train", train_idx), ("val", val_idx)]:
        out = output_dir / stage
        out.mkdir(exist_ok=True)
        np.save(out / "data.npy", all_data[indices])
        np.save(out / "data_in_chans.npy", all_in_chans[indices])
        np.save(out / "data_in_times.npy", all_in_times[indices])
        print(f"Saved {stage}: {len(indices)} samples to {out}")

    print("Done. Use EMGPretrainDatasetFromFolder with these paths.")


if __name__ == "__main__":
    main()
