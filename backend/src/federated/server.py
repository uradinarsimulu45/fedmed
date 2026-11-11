"""Central aggregator: broadcasts weights, waits, averages with FedAvg."""
import argparse
import flwr as fl


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rounds", type=int, default=1)
    p.add_argument("--address", default="0.0.0.0:8080")
    a = p.parse_args()

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        min_fit_clients=3,
        min_available_clients=3,
        fraction_evaluate=0.0,
    )
    fl.server.start_server(
        server_address=a.address,
        config=fl.server.ServerConfig(num_rounds=a.rounds),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
