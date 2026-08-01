"""
Torch-free inference path for the shipped RL policy.

policy_rl.pt is a TorchScript archive, so loading it requires PyTorch — roughly
250-350 MB resident before a single agent has acted. A Render free instance has
512 MB total and also has to hold uvicorn, which is the real reason the live
simulation could never run in the same process as the dashboard.

policy_rl.onnx is the same 452k-parameter network (2985-word vocab, 64-dim
embeddings, 2 transformer encoder layers) exported to ONNX by
scripts/export_onnx.py, which asserts logit parity against the .pt file as part
of the export. onnxruntime executes it in ~30 MB.

The backend preference works the way you would expect: the lighter path
(onnxruntime) wins when it is importable, the TorchScript model is the fallback,
and the numbers are the same either way -- scripts/check_parity.py verifies that
the two select products identically, not just that the logits match.

(An earlier version of this docstring described this as mirroring "the C++/Python
fallback already used by scheduler.py and social_graph.py". No such fallback
exists: both of those modules are pure Python and never attempt a C++ import.
The C++ ports in agents/cpp/ are not wired into the runtime.)
"""
import os
import random as _random
import sys

import numpy as np

DEFAULT_ONNX_PATH = "policy_rl.onnx"
DEFAULT_TORCHSCRIPT_PATH = "policy_rl.pt"

# Sampling constants — these mirror agents/rl_trainer.select_action_with_social,
# which is still the training-time path (it needs autograd-capable tensors).
# Keep the two in sync: same temperature, same window, same boost.
TEMPERATURE = 1.5
SIGNAL_WINDOW = 5
SOCIAL_BOOST = 2.0


class _OnnxPolicy:
    backend = "onnxruntime"

    def __init__(self, path):
        import onnxruntime as ort

        options = ort.SessionOptions()
        # One instance is serving both the web app and the simulation on a
        # shared 0.1-CPU box; letting ORT spawn a thread pool per session just
        # adds contention for a model this small.
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self._session = ort.InferenceSession(
            path, sess_options=options, providers=["CPUExecutionProvider"]
        )
        self._input_name = self._session.get_inputs()[0].name
        self.vocab_size = int(self._session.get_outputs()[0].shape[-1])

    def logits(self, state_ids):
        batch = np.asarray([list(state_ids)], dtype=np.int64)
        out = self._session.run(None, {self._input_name: batch})[0]
        return out[0]


class _TorchPolicy:
    backend = "torch"

    def __init__(self, path):
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")   # CPU-only build
        import torch

        self._torch = torch
        self._model = torch.jit.load(path, map_location="cpu")
        self._model.eval()
        with torch.no_grad():
            probe = self._model(torch.zeros(1, 3, dtype=torch.long))
        self.vocab_size = int(probe.shape[-1])

    def logits(self, state_ids):
        torch = self._torch
        with torch.no_grad():
            out = self._model(torch.tensor([list(state_ids)], dtype=torch.long))
        return out[0].numpy()


def load_policy(onnx_path=DEFAULT_ONNX_PATH, torchscript_path=DEFAULT_TORCHSCRIPT_PATH):
    """Return the lightest available policy backend.

    Raises RuntimeError only when neither backend can be loaded, so callers can
    degrade to something useful instead of crashing the process that owns them.
    """
    errors = []

    if os.path.exists(onnx_path):
        try:
            return _OnnxPolicy(onnx_path)
        except Exception as exc:            # missing onnxruntime, corrupt file
            errors.append(f"onnx ({onnx_path}): {exc}")
    else:
        errors.append(f"onnx ({onnx_path}): file not found")

    if os.path.exists(torchscript_path):
        try:
            return _TorchPolicy(torchscript_path)
        except Exception as exc:            # torch not installed, or OOM
            errors.append(f"torch ({torchscript_path}): {exc}")
    else:
        errors.append(f"torch ({torchscript_path}): file not found")

    raise RuntimeError("no usable policy backend — " + "; ".join(errors))


def social_logits(
    logits,
    social_signal,
    item_to_idx,
    temperature=TEMPERATURE,
    signal_window=SIGNAL_WINDOW,
    social_boost=SOCIAL_BOOST,
):
    """Temperature-scale the logits and apply the social-contagion boost.

    Mirror of the arithmetic in rl_trainer.select_action_with_social: divide by
    the temperature, then add ``strength * SOCIAL_BOOST`` for every product
    still in the agent's recent social signal. Split out from the sampling step
    so scripts/check_parity.py can compare it against the torch path directly.
    """
    adjusted = np.asarray(logits, dtype=np.float64) / temperature

    for product_id, strength in social_signal[-signal_window:]:
        idx = item_to_idx.get(product_id)
        if idx is not None:
            adjusted[idx] += strength * social_boost

    return adjusted


def sample_logits(adjusted, rng=_random):
    """Draw one index from ``softmax(adjusted)``.

    Equivalent to torch's ``Categorical(logits=...).sample()``, but drawing from
    the ``random`` module so a single seed governs the whole simulation.
    """
    weights = np.exp(adjusted - adjusted.max())   # shift for numerical stability
    cumulative = np.cumsum(weights)
    target = rng.random() * cumulative[-1]        # rng.random() < 1.0, never out of range
    return int(np.searchsorted(cumulative, target, side="right"))


def sample_with_social(
    logits,
    social_signal,
    item_to_idx,
    temperature=TEMPERATURE,
    signal_window=SIGNAL_WINDOW,
    social_boost=SOCIAL_BOOST,
    rng=_random,
):
    """Pick the next product: socially-boosted, temperature-scaled sampling."""
    return sample_logits(
        social_logits(
            logits, social_signal, item_to_idx, temperature, signal_window, social_boost
        ),
        rng=rng,
    )


def describe_backend(policy):
    """One-line banner used by both the CLI and the gateway startup log."""
    return f"policy backend: {policy.backend} ({policy.vocab_size:,} products)"


if __name__ == "__main__":                  # quick manual check
    loaded = load_policy()
    print(describe_backend(loaded), file=sys.stderr)
