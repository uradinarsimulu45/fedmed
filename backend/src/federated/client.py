"""Hospital node: trains locally on its private shard, returns only weights."""
import numpy as np
import argparse
import numpy as np
import torch
import flwr as fl
from collections import OrderedDict
from torch.utils.data import DataLoader
from monai.losses import DiceLoss

from src.data.transforms import make_dataset
from src.federated.partition import get_partitions
from src.models.unet import build_model


def get_weights(model):
    return [v.cpu().numpy() for v in model.state_dict().values()]


def set_weights(model, weights):
    keys = model.state_dict().keys()
    state = OrderedDict({k: torch.tensor(w) for k, w in zip(keys, weights)})
    model.load_state_dict(state, strict=True)


class HospitalClient(fl.client.NumPyClient):
    dp_clip = 0.0
    dp_noise = 0.0
    def __init__(self, hospital_id, local_epochs, max_train, crash_at_round=0):
        self.crash_at_round = crash_at_round
        self.rounds_seen = 0
        shards, _ = get_partitions(max_train=max_train)
        self.files = shards[hospital_id]
        self.hospital_id = hospital_id
        self.local_epochs = local_epochs
        self.model = build_model()
        self.loss_fn = DiceLoss(sigmoid=True, squared_pred=True)
        self.loader = DataLoader(make_dataset(self.files, True), batch_size=1,
                                 shuffle=True, num_workers=0)

    def get_parameters(self, config):
        return get_weights(self.model)

    def fit(self, parameters, config):
        self.rounds_seen += 1
        if self.crash_at_round and self.rounds_seen == self.crash_at_round:
            print(f"[Hospital {self.hospital_id}] simulated crash", flush=True)
            import os; os._exit(1)
        set_weights(self.model, parameters)
        global_w = [np.array(x, copy=True) for x in parameters]
        self.model.train()
        opt = torch.optim.AdamW(self.model.parameters(), lr=1e-3, weight_decay=1e-5)
        losses = []
        for _ in range(self.local_epochs):
            for batch in self.loader:
                opt.zero_grad()
                loss = self.loss_fn(self.model(batch["image"]), batch["label"])
                loss.backward()
                opt.step()
                losses.append(loss.item())
        avg = sum(losses) / len(losses)
        print(f"[Hospital {self.hospital_id}] local loss {avg:.4f}", flush=True)
        new_w = get_weights(self.model)
        if self.dp_clip > 0:
            idx = [i for i, w in enumerate(new_w) if np.issubdtype(w.dtype, np.floating)]
            delta = [new_w[i] - global_w[i] for i in idx]
            norm = float(np.sqrt(sum((d.astype(np.float64) ** 2).sum() for d in delta)))
            scale = min(1.0, self.dp_clip / (norm + 1e-12))
            rng = np.random.default_rng()
            for i, d in zip(idx, delta):
                noisy = d * scale + rng.normal(0, self.dp_noise * self.dp_clip, d.shape)
                new_w[i] = (global_w[i] + noisy).astype(new_w[i].dtype)
            print(f"[Hospital {self.hospital_id}] DP: update norm {norm:.6f}, "
                  f"clip {self.dp_clip}, noise std {self.dp_noise * self.dp_clip:.5f}", flush=True)
        return new_w, len(self.files), {"loss": avg}

    def evaluate(self, parameters, config):
        return 0.0, len(self.files), {}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--id", type=int, required=True)
    p.add_argument("--server", default="127.0.0.1:8080")
    p.add_argument("--local-epochs", type=int, default=1)
    p.add_argument("--max-train", type=int, default=60)
    p.add_argument("--tls", action="store_true")
    p.add_argument("--dp-clip", type=float, default=0.0)
    p.add_argument("--dp-noise", type=float, default=0.0)
    p.add_argument("--crash-at-round", type=int, default=0)
    a = p.parse_args()
    torch.manual_seed(0)
    client = HospitalClient(a.id, a.local_epochs, a.max_train, a.crash_at_round)
    client.dp_clip, client.dp_noise = a.dp_clip, a.dp_noise
    fl.client.start_client(
        server_address=a.server,
        client=client.to_client(),
        root_certificates=open("certs/ca.crt","rb").read() if a.tls else None,
    )


if __name__ == "__main__":
    main()
