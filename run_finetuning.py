import argparse
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))

from NCT import NCT, WINDOW_LEN, TARGET_CHANNELS
from modeling_pretrain import aemg_pretrain_base, aemg_pretrain_large
from modeling_nst import NSTBackbone, get_aemg_base_config, get_aemg_large_config


def get_args():
    p = argparse.ArgumentParser("AEMG Fine-tuning")
    p.add_argument("--pretrain_ckpt", type=str, required=True, help="AEMG pretrain checkpoint")
    p.add_argument("--data_dir", type=str, default=None, help="Path to downstream data (run_tests/data)")
    p.add_argument("--dataset", type=str, default="ninaproDB4",
                   choices=["ninaproDB4", "ToroOssaba", "EMG_EPN_612"])
    p.add_argument("--output_dir", type=str, default="./output_finetune")
    p.add_argument("--linear_probe", action="store_true", help="Freeze backbone, train only classifier")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--model_size", type=str, default="base")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--few_shot_ratio", type=float, default=1.0, help="Fraction of target data for few-shot (e.g. 0.05)")
    return p.parse_args()


class AEMGClassifier(nn.Module):
    def __init__(self, backbone, num_classes, embed_dim=256, linear_probe=True):
        super().__init__()
        self.backbone = backbone
        self.linear_probe = linear_probe
        if linear_probe:
            for p in self.backbone.parameters():
                p.requires_grad = False
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x, in_chan_matrix=None, in_time_matrix=None):
        h = self.backbone(x, in_chan_matrix=in_chan_matrix, in_time_matrix=in_time_matrix,
                          return_qrs_tokens=True)
        h = h.mean(1)
        return self.head(h)


def load_downstream_data(data_dir, dataset_name, nct):
    data_dir = Path(data_dir or str(ROOT / "run_tests" / "data"))
    if dataset_name == "ninaproDB4":
        path = data_dir / "ninaproDB4_folder"
        from datasets.data_loader import load_ninapro_db4_for_aemg
        return load_ninapro_db4_for_aemg(str(path), nct)
    elif dataset_name == "ToroOssaba":
        path = data_dir / "ToroOssaba_folder"
        from datasets.data_loader import load_toro_ossaba_for_aemg
        return load_toro_ossaba_for_aemg(str(path), nct)
    elif dataset_name == "EMG_EPN_612":
        path = data_dir / "EMG-EPN612_folder"
        from datasets.data_loader import load_emg_epn_for_aemg
        return load_emg_epn_for_aemg(str(path), nct)
    else:
        raise NotImplementedError(f"Dataset {dataset_name}")


def loso_evaluate(model, data_list, device, batch_size=32):
    n_subjects = len(data_list)
    scores = []
    for test_idx in range(n_subjects):
        train_data = []
        train_labels = []
        for i in range(n_subjects):
            if i == test_idx:
                continue
            train_data.append(data_list[i]["X"])
            train_labels.append(data_list[i]["y"])
        X_train = np.concatenate(train_data, axis=0)
        y_train = np.concatenate(train_labels, axis=0)
        X_test = data_list[test_idx]["X"]
        y_test = data_list[test_idx]["y"]
        in_chans_train = np.concatenate([data_list[i]["in_chans"] for i in range(n_subjects) if i != test_idx], axis=0)
        in_times_train = np.concatenate([data_list[i]["in_times"] for i in range(n_subjects) if i != test_idx], axis=0)
        in_chans_test = data_list[test_idx]["in_chans"]
        in_times_test = data_list[test_idx]["in_times"]
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
        ds = TensorDataset(
            torch.from_numpy(X_train).float(),
            torch.from_numpy(y_train).long(),
            torch.from_numpy(in_chans_train),
            torch.from_numpy(in_times_train)
        )
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
        n_epochs = 100
        for _ in range(n_epochs):
            for batch in loader:
                x, y, ic, it = [b.to(device) for b in batch]
                logits = model(x, in_chan_matrix=ic, in_time_matrix=it)
                loss = nn.CrossEntropyLoss()(logits, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        model.eval()
        with torch.no_grad():
            x = torch.from_numpy(X_test).float().to(device)
            ic = torch.from_numpy(in_chans_test).to(device)
            it = torch.from_numpy(in_times_test).to(device)
            logits = model(x, in_chan_matrix=ic, in_time_matrix=it)
            pred = logits.argmax(1).cpu().numpy()
        acc = (pred == y_test).mean()
        scores.append(acc)
        print(f"  Subject {test_idx+1}/{n_subjects}: acc={acc:.4f}")
    return np.mean(scores), scores


def main():
    args = get_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    nct = NCT(fs=2000, max_len=256, token_len=WINDOW_LEN, target_channels=TARGET_CHANNELS)
    data_list = load_downstream_data(args.data_dir, args.dataset, nct)
    if not data_list:
        print("No data loaded. Check data_dir and dataset.")
        return

    if args.model_size == "large":
        pretrain = aemg_pretrain_large()
    else:
        pretrain = aemg_pretrain_base()
    ckpt = torch.load(args.pretrain_ckpt, map_location="cpu")
    state = ckpt.get("model", ckpt)
    if any(k.startswith("backbone.") for k in state.keys()):
        backbone_state = {k.replace("backbone.", ""): v for k, v in state.items() if k.startswith("backbone.")}
        pretrain.backbone.load_state_dict(backbone_state, strict=False)
    else:
        pretrain.load_state_dict(state, strict=False)
    num_classes = int(max(d["y"].max() for d in data_list)) + 1
    embed_dim = get_aemg_base_config()["embed_dim"] if args.model_size == "base" else get_aemg_large_config()["embed_dim"]
    model = AEMGClassifier(pretrain.backbone, num_classes, embed_dim=embed_dim, linear_probe=args.linear_probe).to(device)

    print(f"LOSO evaluation on {args.dataset}, {len(data_list)} subjects")
    mean_acc, per_subject = loso_evaluate(model, data_list, device, args.batch_size)
    print(f"Mean LOSO accuracy: {mean_acc:.4f}")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "loso_results.txt"), "w") as f:
        f.write(f"Dataset: {args.dataset}\nMean: {mean_acc:.4f}\nPer-subject: {per_subject}\n")


if __name__ == "__main__":
    main()
