"""Central aggregator: FedAvg + global-model evaluation on shared validation set."""
import argparse, json, os
import torch
import flwr as fl
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from monai.transforms import Activations, AsDiscrete, Compose
from torch.utils.data import DataLoader

from src.data.transforms import make_dataset, ROI
from src.federated.client import set_weights
from src.federated.partition import get_partitions
from src.models.unet import build_model


def make_evaluate_fn(val_files, history, out_path):
    model = build_model()
    loader = DataLoader(make_dataset(val_files, False), batch_size=1, num_workers=0)
    metric = DiceMetric(include_background=True, reduction="mean_batch")
    post = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

    def evaluate(server_round, parameters, config):
        if server_round == 0:
            return None
        set_weights(model, parameters)
        model.eval()
        with torch.no_grad():
            for b in loader:
                pred = sliding_window_inference(b["image"], ROI, 2, model)
                metric(y_pred=[post(i) for i in pred], y=list(b["label"]))
            tc, wt, et = metric.aggregate().tolist()
            metric.reset()
        mean = (tc + wt + et) / 3
        rec = dict(round=server_round, dice_tc=round(tc, 4), dice_wt=round(wt, 4),
                   dice_et=round(et, 4), dice_mean=round(mean, 4))
        history.append(rec)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump({"history": history}, f, indent=2)
        print(f"[Server] round {server_round} global Dice: {rec}", flush=True)
        return 1 - mean, {"dice_mean": mean}

    return evaluate


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rounds", type=int, default=1)
    p.add_argument("--address", default="0.0.0.0:8080")
    p.add_argument("--max-val", type=int, default=12)
    p.add_argument("--out", default="results/federated_metrics.json")
    a = p.parse_args()

    _, val_files = get_partitions(max_val=a.max_val)
    history = []
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        min_fit_clients=3,
        min_available_clients=3,
        fraction_evaluate=0.0,
        evaluate_fn=make_evaluate_fn(val_files, history, a.out),
    )
    fl.server.start_server(
        server_address=a.address,
        config=fl.server.ServerConfig(num_rounds=a.rounds),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
