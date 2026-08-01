import numpy as np
from collections import Counter, defaultdict


# ---------------------------------------------------------------------------
# Single source of truth for every benchmark range.
#
# This used to be defined in three places per metric -- the `real_benchmark`
# string written into the JSON, the numeric threshold in the verdict
# expression, and the label in the printed summary -- and they had drifted
# apart. Conversion, for example, was reported against "2-5%" in the JSON,
# scored against 0.02-0.20 in the verdict, and printed as "2-20%". A 16%
# result was therefore recorded as a PASS against a stated 2-5% benchmark.
#
# All three now derive from this dict. Change a range here or nowhere. If a
# run falls outside a range, it fails -- the range does not move to meet it.
# ---------------------------------------------------------------------------
BENCHMARKS = {
    "session_length": {
        # Compared against the real sessions in the same run, not a literature
        # figure. The tolerance is loose relative to a mean of ~8 items.
        "tolerance": 4.0,
        "label": "within 4.0 items of the real mean",
    },
    "conversion_rate": {
        "low": 0.02,
        "high": 0.05,
        "label": "2-5%",
        "source": "industry standard e-commerce conversion rate",
    },
    "abandonment_rate": {
        "label": "70-85% (industry cart abandonment) -- NOT APPLICABLE, see note",
        "note": (
            "This metric is the fraction of sessions containing no 'checkout', "
            "which is exactly 1 - conversion_rate by construction -- the two "
            "always sum to 1.0. Industry cart abandonment (~70%) is measured "
            "over sessions that created a cart, a different denominator that "
            "excludes browse-only sessions. Scoring this number against the "
            "70-85% range compares two different quantities, so it is reported "
            "but deliberately left unscored."
        ),
    },
    "social_influence_coefficient": {
        "low": 0.20,
        "high": 0.40,
        "label": "0.20-0.40",
        "source": "power-law concentration, Amazon trend data",
    },
}


def _verdict(passed: bool) -> str:
    return "PASS" if passed else "FAIL"


def session_length_stats(sessions: list) -> dict:
    lengths = [len(s) for s in sessions]
    return {
        "mean":   float(np.mean(lengths)),
        "std":    float(np.std(lengths)),
        "min":    int(np.min(lengths)),
        "max":    int(np.max(lengths))
    }


def conversion_rate(sessions: list, checkout_action: str = "checkout") -> float:
    converted = sum(1 for s in sessions if checkout_action in s)
    return converted / len(sessions) if sessions else 0.0


def abandonment_rate(sessions: list) -> float:
    """
    Fraction of sessions where agent browsed but never reached checkout.
    Real e-commerce abandonment rate is typically 70-85%.
    """
    abandoned = sum(1 for s in sessions if "checkout" not in s and len(s) > 0)
    return abandoned / len(sessions) if sessions else 0.0


def social_influence_coefficient(sessions: list, top_k: int = 5) -> float:
    """
    Measures how concentrated purchases are around top products.
    High concentration = social contagion working (trending products
    dominate). Real Amazon shows power-law: top 5 products capture
    20-40% of interactions during trend events.
    Returns: fraction of total interactions captured by top_k products.
    """
    all_items = [item for s in sessions for item in s if item != "checkout"]
    if not all_items:
        return 0.0
    counts = Counter(all_items)
    total = sum(counts.values())
    top_items = counts.most_common(top_k)
    top_count = sum(count for _, count in top_items)
    return round(top_count / total, 4)


def budget_sensitivity(agents: list) -> dict:
    """
    Checks if low-budget agents buy less than high-budget agents.
    This validates that the virtual wallet constraint is working
    realistically. Returns avg checkouts per persona.
    """
    persona_checkouts = defaultdict(list)
    for agent in agents:
        checkout_count = agent.session_log.count("checkout")
        persona_checkouts[agent.persona].append(checkout_count)

    return {
        persona: round(sum(counts) / len(counts), 2)
        for persona, counts in persona_checkouts.items()
    }


def full_validation_report(real_sessions: list, sim_sessions: list,
                            agents: list = None) -> dict:
    """
    Behavioral similarity report — compares HOW agents behave,
    not WHAT they buy. Product choice is personal and unique to
    each user; behavior patterns (attention span, abandonment,
    social influence) are universal and measurable.
    """
    real_stats  = session_length_stats(real_sessions)
    sim_stats   = session_length_stats(sim_sessions)
    real_conv   = conversion_rate(real_sessions)
    sim_conv    = conversion_rate(sim_sessions)
    sim_abandon = abandonment_rate(sim_sessions)
    sim_social  = social_influence_coefficient(sim_sessions)

    conv_bm = BENCHMARKS["conversion_rate"]
    social_bm = BENCHMARKS["social_influence_coefficient"]
    session_tol = BENCHMARKS["session_length"]["tolerance"]

    report = {
        "session_length": {
            "real":      real_stats,
            "simulated": sim_stats,
            "tolerance": f"|mean difference| < {session_tol}",
            "verdict": _verdict(
                abs(real_stats["mean"] - sim_stats["mean"]) < session_tol
            ),
        },
        "conversion_rate": {
            "real_benchmark": conv_bm["label"],
            "benchmark_source": conv_bm["source"],
            "simulated": round(sim_conv, 4),
            "verdict": _verdict(conv_bm["low"] <= sim_conv <= conv_bm["high"]),
        },
        "abandonment_rate": {
            "real_benchmark": BENCHMARKS["abandonment_rate"]["label"],
            "simulated": round(sim_abandon, 4),
            # Deliberately not scored. See BENCHMARKS["abandonment_rate"]["note"]:
            # this quantity is 1 - conversion_rate by construction and does not
            # measure the same thing as the industry cart-abandonment figure.
            "verdict": "NOT_COMPARABLE",
            "note": BENCHMARKS["abandonment_rate"]["note"],
        },
        "social_influence_coefficient": {
            "description": "fraction of interactions captured by top 5 products",
            "real_benchmark": social_bm["label"],
            "benchmark_source": social_bm["source"],
            "simulated": sim_social,
            "verdict": _verdict(
                social_bm["low"] <= sim_social <= social_bm["high"]
            ),
        },
    }

    if agents:
        report["budget_sensitivity"] = budget_sensitivity(agents)

    def verdict_label(v):
        return {"PASS": "[PASS]", "FAIL": "[FAIL]"}.get(v, f"[{v}]")

    print("\n========== SHOPPING SIMULATION REPORT ==========")
    print(f"\n  Session Length (avg products viewed per shopper)")
    print(f"    Real shoppers:      {real_stats['mean']:.1f}  (±{real_stats['std']:.1f})")
    print(f"    Simulated:          {sim_stats['mean']:.1f}  (±{sim_stats['std']:.1f})")
    print(f"    Tolerance:          {BENCHMARKS['session_length']['label']}")
    print(f"    {verdict_label(report['session_length']['verdict'])}")

    print(f"\n  Purchase Rate")
    print(f"    Industry benchmark: {conv_bm['label']}  ({conv_bm['source']})")
    print(f"    Simulated:          {sim_conv:.1%}")
    print(f"    {verdict_label(report['conversion_rate']['verdict'])}")

    print(f"\n  Non-converting sessions  (reported, not scored)")
    print(f"    Simulated:          {sim_abandon:.1%}")
    print(f"    Note:               = 1 - purchase rate by construction; not")
    print(f"                        comparable to industry cart abandonment.")
    print(f"    {verdict_label(report['abandonment_rate']['verdict'])}")

    print(f"\n  Trend Concentration  (top 5 products share of traffic)")
    print(f"    Industry benchmark: {social_bm['label']}  ({social_bm['source']})")
    print(f"    Simulated:          {sim_social:.1%}")
    print(f"    {verdict_label(report['social_influence_coefficient']['verdict'])}")

    if agents:
        bs = report["budget_sensitivity"]
        print(f"\n  Average Purchases Per Shopper Type")
        persona_labels = {
            "power_buyer":   "Power Buyers  ",
            "average_buyer": "Regular Buyers",
            "browser":       "Browsers      ",
        }
        for persona, label in persona_labels.items():
            if persona in bs:
                print(f"    {label}  {bs[persona]:.2f}")

    print("\n=================================================\n")

    return report
