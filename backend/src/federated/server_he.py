"""Server that aggregates CKKS ciphertexts. It has NO secret key and never decrypts."""
import argparse, json, os
import numpy as np
import tenseal as ts
import flwr as fl
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters


class EncryptedFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, ctx, history, out_path, **kw):
        super().__init__(**kw)
        self.ctx, self.history, self.out_path = ctx, history, out_path
        self.upload_mb = 0.0

    def configure_fit(self, server_round, parameters, client_manager):
        if server_round == 1:
            client_manager.wait_for(3, timeout=120)
        return super().configure_fit(server_round, parameters, client_manager)

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}
        payloads = [(parameters_to_ndarrays(r.parameters), r.num_examples) for _, r in results]
        total = sum(n for _, n in payloads)
        self.upload_mb = sum(a.nbytes for a in payloads[0][0]) / 1e6
        out = []
        for c in range(len(payloads[0][0])):
            acc = None
            for arrays, n in payloads:
                ct = ts.ckks_vector_from(self.ctx, arrays[c].tobytes()) * (n / total)
                acc = ct if acc is None else acc + ct
            out.append(np.frombuffer(acc.serialize(), dtype=np.uint8).copy())
        print(f"[Server] round {server_round}: added {len(payloads)} hospitals' "
              f"{len(out)} ciphertexts WITHOUT decrypting", flush=True)
        return ndarrays_to_parameters(out), {}

    def aggregate_evaluate(self, server_round, results, failures):
        if not results:
            return None, {}
        n = sum(r.num_examples for _, r in results)
        m = {k: sum(r.num_examples * r.metrics[k] for _, r in results) / n
             for k in ("dice_tc", "dice_wt", "dice_et")}
        mean = sum(m.values()) / 3
        rec = dict(round=server_round, dice_tc=round(m["dice_tc"], 4),
                   dice_wt=round(m["dice_wt"], 4), dice_et=round(m["dice_et"], 4),
                   dice_mean=round(mean, 4), upload_mb_per_hospital=round(self.upload_mb, 1))
        self.history.append(rec)
        os.makedirs(os.path.dirname(self.out_path), exist_ok=True)
        json.dump({"history": self.history}, open(self.out_path, "w"), indent=2)
        print(f"[Server] round {server_round} global Dice: {rec}", flush=True)
        return 1 - mean, {"dice_mean": mean}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rounds", type=int, default=1)
    p.add_argument("--address", default="0.0.0.0:8080")
    p.add_argument("--tls", action="store_true")
    p.add_argument("--timeout", type=float, default=900.0)
    p.add_argument("--out", default="results/federated_he_metrics.json")
    a = p.parse_args()

    ctx = ts.context_from(open("certs/he_public.ctx", "rb").read())
    ctx.global_scale = 2 ** 40
    print("Server holds secret key:", ctx.is_private(), flush=True)

    strategy = EncryptedFedAvg(
        ctx, [], a.out,
        fraction_fit=1.0, min_fit_clients=2, min_available_clients=2, accept_failures=True,
        fraction_evaluate=1.0, min_evaluate_clients=2,
    )
    fl.server.start_server(
        server_address=a.address,
        config=fl.server.ServerConfig(num_rounds=a.rounds, round_timeout=a.timeout),
        strategy=strategy,
        certificates=(open("certs/ca.crt", "rb").read(), open("certs/server.pem", "rb").read(),
                      open("certs/server.key", "rb").read()) if a.tls else None,
    )


if __name__ == "__main__":
    main()
