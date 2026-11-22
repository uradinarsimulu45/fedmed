"""Hospital node with CKKS encryption: sends ciphertext, decrypts the global average."""
import argparse
import numpy as np
import torch
import tenseal as ts
import flwr as fl
from torch.utils.data import DataLoader
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from monai.transforms import Activations, AsDiscrete, Compose

from src.data.transforms import make_dataset, ROI
from src.federated.client import get_weights, set_weights
from src.federated.partition import get_partitions
from src.models.unet import build_model

CHUNK = 4096


def load_secret_ctx():
    ctx = ts.context_from(open("certs/he_secret.ctx", "rb").read())
    ctx.global_scale = 2 ** 40
    return ctx


def is_encrypted(params):
    return len(params) > 0 and params[0].dtype == np.uint8


class HEClient(fl.client.NumPyClient):
    def __init__(self, hospital_id, local_epochs, max_train, max_val):
        shards, val = get_partitions(max_train=max_train, max_val=max_val)
        self.files = shards[hospital_id]
        self.val_files = val[hospital_id::3]
        self.hospital_id = hospital_id
        self.local_epochs = local_epochs
        self.model = build_model()
        self.ctx = load_secret_ctx()
        self.loss_fn = DiceLoss(sigmoid=True, squared_pred=True)
        self.loader = DataLoader(make_dataset(self.files, True), batch_size=1,
                                 shuffle=True, num_workers=0)
        self.total = sum(v.numel() for v in self.model.state_dict().values())

    # ---- encryption helpers ----
    def _encrypt(self):
        flat = np.concatenate([v.detach().cpu().numpy().ravel().astype(np.float64)
                               for v in self.model.state_dict().values()])
        out = []
        for i in range(0, len(flat), CHUNK):
            ct = ts.ckks_vector(self.ctx, flat[i:i + CHUNK].tolist())
            out.append(np.frombuffer(ct.serialize(), dtype=np.uint8).copy())
        return out

    def _decrypt(self, arrays):
        parts = [np.array(ts.ckks_vector_from(self.ctx, a.tobytes()).decrypt()) for a in arrays]
        return np.concatenate(parts)[:self.total]

    def _load(self, parameters):
        if not is_encrypted(parameters):
            set_weights(self.model, parameters)  # round 1: random initial model
            return
        flat = self._decrypt(parameters)
        new, i = {}, 0
        for k, v in self.model.state_dict().items():
            n = v.numel()
            new[k] = torch.tensor(flat[i:i + n].reshape(tuple(v.shape)), dtype=v.dtype)
            i += n
        self.model.load_state_dict(new)

    # ---- Flower API ----
    def get_parameters(self, config):
        return get_weights(self.model)

    def fit(self, parameters, config):
        self._load(parameters)
        self.model.train()
        opt = torch.optim.AdamW(self.model.parameters(), lr=1e-3, weight_decay=1e-5)
        losses = []
        for _ in range(self.local_epochs):
            for b in self.loader:
                opt.zero_grad()
                loss = self.loss_fn(self.model(b["image"]), b["label"])
                loss.backward()
                opt.step()
                losses.append(loss.item())
        avg = sum(losses) / len(losses)
        enc = self._encrypt()
        mb = sum(a.nbytes for a in enc) / 1e6
        print(f"[Hospital {self.hospital_id}] local loss {avg:.4f} | "
              f"sent {len(enc)} ciphertexts, {mb:.1f} MB", flush=True)
        return enc, len(self.files), {"loss": avg}

    def evaluate(self, parameters, config):
        self._load(parameters)
        self.model.eval()
        metric = DiceMetric(include_background=True, reduction="mean_batch")
        post = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
        loader = DataLoader(make_dataset(self.val_files, False), batch_size=1, num_workers=0)
        with torch.no_grad():
            for b in loader:
                pred = sliding_window_inference(b["image"], ROI, 2, self.model)
                metric(y_pred=[post(i) for i in pred], y=list(b["label"]))
        tc, wt, et = metric.aggregate().tolist()
        mean = (tc + wt + et) / 3
        return float(1 - mean), len(self.val_files), {"dice_tc": tc, "dice_wt": wt, "dice_et": et}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--id", type=int, required=True)
    p.add_argument("--server", default="127.0.0.1:8080")
    p.add_argument("--local-epochs", type=int, default=1)
    p.add_argument("--max-train", type=int, default=60)
    p.add_argument("--max-val", type=int, default=12)
    p.add_argument("--tls", action="store_true")
    a = p.parse_args()
    fl.client.start_client(
        server_address=a.server,
        client=HEClient(a.id, a.local_epochs, a.max_train, a.max_val).to_client(),
        root_certificates=open("certs/ca.crt", "rb").read() if a.tls else None,
    )


if __name__ == "__main__":
    main()
