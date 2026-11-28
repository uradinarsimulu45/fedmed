"""Approximate (epsilon, delta) for the Gaussian mechanism with simple composition.
Assumptions: per-hospital update clipped to L2 norm C, noise std = z*C per value,
each round is one release, delta = 1e-5 per round, no subsampling amplification.
Valid classic bound needs epsilon_round <= 1; values above that are only indicative."""
import math

def eps_per_round(z, delta=1e-5):
    return math.sqrt(2 * math.log(1.25 / delta)) / z

if __name__ == "__main__":
    rounds = 5
    print(f"{'noise z':>8} | {'eps/round':>10} | {'eps after '+str(rounds)+' rounds':>18}")
    for z in (0.005, 0.02, 0.05, 0.2, 1.0, 4.0):
        e = eps_per_round(z)
        print(f"{z:>8} | {e:>10.1f} | {e*rounds:>18.1f}")
