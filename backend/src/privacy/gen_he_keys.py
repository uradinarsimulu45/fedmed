"""Make CKKS keys. Hospitals get the secret context, the server only the public one."""
import os
import tenseal as ts

os.makedirs("certs", exist_ok=True)
ctx = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=8192,
                 coeff_mod_bit_sizes=[60, 40, 40, 60])
ctx.global_scale = 2 ** 40
open("certs/he_secret.ctx", "wb").write(ctx.serialize(save_secret_key=True))
open("certs/he_public.ctx", "wb").write(ctx.serialize(save_secret_key=False))
print("Keys written: certs/he_secret.ctx (hospitals), certs/he_public.ctx (server)")
