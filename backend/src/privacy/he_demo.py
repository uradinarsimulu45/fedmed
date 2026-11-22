"""CKKS demo: server averages encrypted hospital vectors without decrypting."""
import numpy as np
import tenseal as ts

# --- Key setup (done by the hospitals, NOT the server) ---
ctx = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=8192,
                 coeff_mod_bit_sizes=[60, 40, 40, 60])
ctx.global_scale = 2 ** 40

# Public context for the server: no secret key inside
server_ctx = ts.context_from(ctx.serialize(save_secret_key=False))
print("Server context has secret key:", server_ctx.is_private())

# --- Three hospitals' fake weight vectors ---
rng = np.random.default_rng(0)
weights = [rng.normal(0, 0.1, 4000) for _ in range(3)]

# --- Hospitals encrypt and send bytes ---
sent = [ts.ckks_vector(ctx, w).serialize() for w in weights]
print(f"Plain size per hospital: {weights[0].nbytes/1e3:.1f} KB | "
      f"Encrypted size: {len(sent[0])/1e3:.1f} KB")

# --- Server: works on ciphertext only ---
enc = [ts.ckks_vector_from(server_ctx, b) for b in sent]
enc_avg = (enc[0] + enc[1] + enc[2]) * (1 / 3)
result_bytes = enc_avg.serialize()

# --- Hospitals decrypt the aggregate ---
dec = np.array(ts.ckks_vector_from(ctx, result_bytes).decrypt())
plain = np.mean(weights, axis=0)
print("Max error vs plain FedAvg:", float(np.abs(dec - plain).max()))
print("Match (error < 1e-4):", bool(np.abs(dec - plain).max() < 1e-4))
