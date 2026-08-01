"""
Verify the torch-free live path picks products the same way the torch path did.

scripts/export_onnx.py proves the exported network reproduces policy_rl.pt's
logits. This script covers the two steps layered on top of those logits inside
the simulation:

  1. the social-contagion boost      (policy_runtime.social_logits vs rl_trainer)
  2. the categorical draw            (policy_runtime.sample_logits vs torch)

Development-only — it needs torch, which the deployed image does not have.

    python scripts/check_parity.py
"""
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import random
import sys

import numpy as np
import torch
from torch.distributions import Categorical

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents import policy_runtime
from agents.policy_runtime import load_policy, sample_logits, social_logits

PROBE_STATES = 32
DRAWS = 200_000
PROB_TOLERANCE = 1e-6      # float32 logits promoted to float64 on both sides
CHI_SQUARE_TOLERANCE = 0.02


def random_social_signal(product_ids, rng, size=5):
    return [(rng.choice(product_ids), 1 / (2 ** rng.randint(1, 2))) for _ in range(size)]


def check_social_boost(policy, item_to_idx, product_ids):
    """The boosted logits must match rl_trainer.select_action_with_social."""
    torch_model = torch.jit.load("policy_rl.pt", map_location="cpu")
    torch_model.eval()

    rng = random.Random(0)
    worst = 0.0

    for _ in range(PROBE_STATES):
        state_ids = [rng.randrange(len(item_to_idx)) for _ in range(3)]
        signal = random_social_signal(product_ids, rng)

        # torch path, transcribed from rl_trainer.select_action_with_social
        with torch.no_grad():
            reference = torch_model(torch.tensor([state_ids])).clone() / policy_runtime.TEMPERATURE
            for product_id, strength in signal[-policy_runtime.SIGNAL_WINDOW:]:
                if product_id in item_to_idx:
                    reference[0][item_to_idx[product_id]] += strength * policy_runtime.SOCIAL_BOOST
            reference_probs = Categorical(logits=reference).probs[0].numpy()

        # torch-free path
        adjusted = social_logits(policy.logits(state_ids), signal, item_to_idx)
        weights = np.exp(adjusted - adjusted.max())
        probs = weights / weights.sum()

        worst = max(worst, float(np.abs(reference_probs - probs).max()))

    print(f"social-boosted probabilities: max abs diff {worst:.3g} "
          f"over {PROBE_STATES} states")
    if worst > PROB_TOLERANCE:
        raise SystemExit(f"FAIL: social boost diverges (>{PROB_TOLERANCE:g})")


def check_sampler():
    """sample_logits must draw from softmax(logits), like torch's Categorical."""
    rng = random.Random(1)
    logits = np.array([2.0, 0.5, -1.0, 3.25, 0.0])
    expected = np.exp(logits - logits.max())
    expected /= expected.sum()

    counts = np.zeros(len(logits))
    for _ in range(DRAWS):
        counts[sample_logits(logits, rng=rng)] += 1
    observed = counts / DRAWS

    drift = float(np.abs(observed - expected).max())
    print(f"sampler distribution: max abs diff {drift:.4f} over {DRAWS:,} draws "
          f"(expected {np.round(expected, 4).tolist()})")
    if drift > CHI_SQUARE_TOLERANCE:
        raise SystemExit(f"FAIL: sampler is biased (>{CHI_SQUARE_TOLERANCE})")

    # An out-of-range index would silently corrupt the product lookup.
    assert all(0 <= sample_logits(logits, rng=rng) < len(logits) for _ in range(1000))


def main():
    import json

    with open("vocab.json") as handle:
        item_to_idx = json.load(handle)
    product_ids = list(item_to_idx.keys())

    policy = load_policy()
    print(f"policy backend under test: {policy.backend}")

    check_social_boost(policy, item_to_idx, product_ids)
    check_sampler()
    print("PASS — the torch-free path selects products identically")


if __name__ == "__main__":
    main()
