"""
Regenerate policy_rl.onnx from policy_rl.pt and verify the two agree.

Development-only: this is the one step that still needs PyTorch. The deployed
gateway ships the .onnx and never imports torch (see agents/policy_runtime.py
for why). Re-run this after retraining:

    python scripts/export_onnx.py

Exit status is non-zero if the exported graph disagrees with the TorchScript
model, so it is safe to wire into CI.
"""
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")   # skip pynvml probe; CPU-only build

import argparse
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.bc_trainer import policyNetwork

# Float32 transformer math reassociates between backends; anything at this scale
# is rounding, not a behavioural difference. Argmax agreement is checked too.
LOGIT_TOLERANCE = 1e-4
PROBE_ROWS = 64


def export(torchscript_path, onnx_path, opset):
    scripted = torch.jit.load(torchscript_path, map_location="cpu")
    scripted.eval()
    weights = scripted.state_dict()

    vocab_size, embedding_dim = weights["embedding.weight"].shape
    window_size = weights["position_emb.weight"].shape[0]
    print(f"model: vocab={vocab_size:,} dim={embedding_dim} window={window_size}")

    # policy_rl.pt is a TorchScript archive; ONNX export wants an eager module,
    # so rebuild the original architecture and load the scripted weights into it.
    eager = policyNetwork(
        vocab_size=vocab_size, embedding_dim=embedding_dim, window_size=window_size
    )
    eager.load_state_dict(weights, strict=True)   # raises on any key/shape drift
    eager.eval()

    generator = torch.Generator().manual_seed(0)
    probe = torch.randint(0, vocab_size, (PROBE_ROWS, window_size), generator=generator)

    with torch.no_grad():
        reference = scripted(probe)
        rebuilt = eager(probe)

    rebuild_drift = (reference - rebuilt).abs().max().item()
    if rebuild_drift != 0.0:
        raise SystemExit(
            f"FAIL: rebuilt eager module does not match policy_rl.pt "
            f"(max abs diff {rebuild_drift:g}) — architecture drift in bc_trainer.py?"
        )
    print("eager rebuild matches TorchScript exactly")

    torch.onnx.export(
        eager,
        (probe[:1],),
        onnx_path,
        input_names=["state_ids"],
        output_names=["logits"],
        dynamic_axes={"state_ids": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=opset,
        dynamo=False,
    )
    print(f"wrote {onnx_path} ({os.path.getsize(onnx_path) / 1e6:.2f} MB)")

    import onnxruntime as ort

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    exported = session.run(None, {"state_ids": probe.numpy().astype(np.int64)})[0]

    logit_drift = np.abs(reference.numpy() - exported).max()
    argmax_agrees = bool((reference.numpy().argmax(1) == exported.argmax(1)).all())
    print(f"onnx vs torchscript: max abs logit diff {logit_drift:.3g}, "
          f"argmax agrees on all {PROBE_ROWS} probes: {argmax_agrees}")

    if logit_drift > LOGIT_TOLERANCE or not argmax_agrees:
        raise SystemExit(
            f"FAIL: exported graph disagrees with policy_rl.pt "
            f"(drift {logit_drift:.3g} > {LOGIT_TOLERANCE:g})"
        )
    print("PASS — policy_rl.onnx is a faithful export")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--torchscript", default="policy_rl.pt")
    parser.add_argument("--onnx", default="policy_rl.onnx")
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()
    export(args.torchscript, args.onnx, args.opset)


if __name__ == "__main__":
    main()
