"""Hospital node: trains locally on its private shard, returns only weights."""
import argparse
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
    def __init__(self, hospital_id, local_epochs, max_train):
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
        set_weights(self.model, parameters)
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
        return get_weights(self.model), len(self.files), {"loss": avg}

    def evaluate(self, parameters, config):
        return 0.0, len(self.files), {}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--id", type=int, required=True)
    p.add_argument("--server", default="127.0.0.1:8080")
    p.add_argument("--local-epochs", type=int, default=1)
    p.add_argument("--max-train", type=int, default=60)
    p.add_argument("--tls", action="store_true")
    a = p.parse_args()
    fl.client.start_client(
        server_address=a.server,
        client=HospitalClient(a.id, a.local_epochs, a.max_train).to_client(),
        root_certificates=open("certs/ca.crt","rb").read() if a.tls else None,
    )


if __name__ == "__main__":
    main()
