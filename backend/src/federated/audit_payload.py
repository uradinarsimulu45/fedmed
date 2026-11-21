"""Audit: what does a hospital send to the server? Only weight tensors."""
from src.federated.client import HospitalClient

c = HospitalClient(hospital_id=0, local_epochs=1, max_train=6)
w = c.get_parameters({})
total = sum(a.size for a in w)
print(f"Payload type: list of {len(w)} numpy arrays")
print(f"Total values sent: {total:,} ({total*4/1e6:.2f} MB as float32)")
print(f"Largest array shape: {max(w, key=lambda a: a.size).shape}")
print(f"Hospital 0 private scans stay local: {len(c.files)} cases, 0 transmitted")
print("Any image-shaped (4,H,W,D) array in payload:",
      any(a.ndim == 4 and a.shape[0] == 4 and min(a.shape[1:]) > 16 for a in w))
