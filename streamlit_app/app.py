import glob, json, math, os, re
import pandas as pd
import streamlit as st

D = os.path.join(os.path.dirname(__file__), "data")
st.set_page_config(page_title="FedMed", layout="wide")


def hist(name):
    return json.load(open(os.path.join(D, name)))["history"]


st.title("FedMed: Federated Brain Tumor Segmentation")
st.caption("Results viewer. Training ran on a CPU laptop at small scale, so Dice is far "
           "below published BraTS results. No patient data is stored here.")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Overview", "Federated vs baseline", "Encryption", "Differential privacy", "Tumor masks"])

with tab1:
    st.image(os.path.join(D, "architecture.png"), caption="Architecture")
    st.markdown("""
Three simulated hospitals train a 3D U-Net on private MRI data. Only model weights leave
a hospital, over gRPC + TLS. The server aggregates with FedAvg, optionally on CKKS-encrypted
weights, and clients can add clipped Gaussian noise (differential privacy).
""")

with tab2:
    b, f = hist("baseline_audit_15epochs.json"), hist("federated_audit_15rounds.json")
    n = min(len(b), len(f))
    df = pd.DataFrame({"step": range(1, n + 1),
                       "Centralized baseline": [r["dice_mean"] for r in b[:n]],
                       "Federated (FedAvg)": [r["dice_mean"] for r in f[:n]]}).set_index("step")
    bb, ff = max(r["dice_mean"] for r in b), max(r["dice_mean"] for r in f)
    c1, c2, c3 = st.columns(3)
    c1.metric("Best baseline Dice", f"{bb:.4f}")
    c2.metric("Best federated Dice", f"{ff:.4f}")
    c3.metric("Ratio", f"{ff / bb:.0%}")
    st.line_chart(df)
    reg = pd.DataFrame(f).set_index("round")[["dice_tc", "dice_wt", "dice_et", "dice_mean"]]
    st.subheader("Federated Dice by tumor region")
    st.line_chart(reg)

with tab3:
    p, e = hist("federated_audit_15rounds.json"), hist("federated_he_audit_15rounds.json")
    n = min(len(p), len(e))
    df = pd.DataFrame({"round": range(1, n + 1),
                       "Plain": [r["dice_mean"] for r in p[:n]],
                       "Encrypted (CKKS)": [r["dice_mean"] for r in e[:n]]}).set_index("round")
    st.line_chart(df)
    st.write(f"Upload per hospital per round: plain 4.8 MB, encrypted "
             f"{e[0]['upload_mb_per_hospital']} MB (about 20x). The server held no secret key.")
    st.info(f"Compared over {n} rounds. The server ran out of memory at round 6 on a 15 GB laptop. "
            "All hospitals share one secret key.")

with tab4:
    rows = []
    for fpath in glob.glob(os.path.join(D, "dp_clip1.0_noise*.json")):
        z = float(re.search(r"noise([\d.]+)\.json", fpath).group(1))
        h = json.load(open(fpath))["history"]
        eps = math.sqrt(2 * math.log(1.25 / 1e-5)) / z if z > 0 else None
        rows.append({"noise": z, "final Dice": h[-1]["dice_mean"],
                     "eps per round": round(eps, 1) if eps else "none"})
    st.dataframe(pd.DataFrame(sorted(rows, key=lambda r: r["noise"])), hide_index=True)
    st.image(os.path.join(D, "dp_tradeoff.png"))
    st.warning("Epsilon is a loose simple-composition bound. Accuracy fell at noise 0.05 and "
               "learning stalled at 4.0. Strong privacy was not affordable at this scale.")

with tab5:
    st.caption("Left to right: MRI, true tumor, predicted tumor. These come from the "
               "centralized baseline model (Dice about 0.3), not the federated model.")
    for img in sorted(glob.glob(os.path.join(D, "masks", "*.png"))):
        st.image(img)
