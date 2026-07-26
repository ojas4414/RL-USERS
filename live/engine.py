"""
The continuous agent simulation, as an importable engine.

This is the loop that used to live inside ``simulate_live.py``'s
``if __name__ == "__main__":`` block. Nothing about the simulation changed —
same funnel gate, same RL product choice, same social contagion, same persona
mix — it is just reachable now from something other than a shell prompt, which
is what lets the deployed gateway run it in-process (see live/runner.py).

Two deliberate differences from the original script:

* Inference goes through agents/policy_runtime, so the loop runs without torch.
* Event delivery is a generator. The engine yields events and holds no opinion
  about where they go; simulate_live.py writes them to a file and Redis, the
  gateway pushes them onto a websocket.
"""
import datetime
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.agent import Agent
from agents.funnel import (
    Funnel_graph as FUNNEL_GRAPH,
    build as build_prerequisites,
    allowed as is_action_allowed,
)
from agents.policy_runtime import load_policy, sample_with_social
from agents.scheduler import Scheduler
from agents.social_graph import (
    social_graph as build_social_graph,
    bsf as propagate_influence,
    signal_strenght as compute_signal_strength,
)
from data.persona_cluster import assign_agent_personas

N_AGENTS = 50
ACTIONS = ["browse", "product_detail", "cart", "checkout"]
WINDOW_SIZE = 3
EPSILON = 0.25
MAX_IN_FILE = 60          # rolling window of events kept for the dashboard
CONNECTION_PROB = 0.1
INFLUENCE_MAX_DEPTH = 2
PRICE_MIN, PRICE_MAX = 9.99, 199.99
REFILL_THRESHOLD = 10.0
VOCAB_PATH = "vocab.json"


def load_vocab(path=VOCAB_PATH, window_size=WINDOW_SIZE):
    """Return {product_id: index}, rebuilding it from raw reviews if needed.

    The rebuild path pulls in pandas and torch, so it is imported lazily — the
    deployed image ships vocab.json and never touches it.
    """
    if os.path.exists(path):
        with open(path) as handle:
            return json.load(handle)

    from agents.bc_trainer import build_vocab
    from data.loader import load_reviews, filter_, sort_split, build, dataset

    frame = load_reviews("data/raw/All_Beauty.jsonl")
    sorted_frame, train_cutoff, val_cutoff = sort_split(filter_(frame))
    pairs = build(sorted_frame, window_size=window_size)
    train_pairs, _, _ = dataset(pairs, train_cutoff, val_cutoff)
    item_to_idx = build_vocab(train_pairs)
    with open(path, "w") as handle:
        json.dump(item_to_idx, handle)
    return item_to_idx


def default_persona_mix(n_agents=N_AGENTS):
    """The 35 average / 10 power / 5 browser split the benchmarks were tuned on."""
    clusters = (
        {f"u{i}": "average_buyer" for i in range(35)}
        | {f"u{i + 35}": "power_buyer" for i in range(10)}
        | {f"u{i + 45}": "browser" for i in range(5)}
    )
    return assign_agent_personas(n_agents, clusters)


class LiveSimulation:
    """Runs agents through the funnel forever, yielding one event at a time."""

    def __init__(self, policy, item_to_idx, n_agents=N_AGENTS, max_events=MAX_IN_FILE):
        self.policy = policy
        self.item_to_idx = item_to_idx
        # Mirrors rl_trainer.build_reverse_vocab, inlined so the engine does not
        # have to import a torch module for one dict comprehension.
        self.idx_to_item = {idx: item for item, idx in item_to_idx.items()}
        self.product_ids = list(item_to_idx.keys())
        self.n_agents = n_agents
        self.max_events = max_events

        self.graph = build_social_graph(n_agents, connection_prob=CONNECTION_PROB)
        self.prereqs = build_prerequisites(FUNNEL_GRAPH)
        self.agents = [
            Agent(agent_id=i, persona=persona)
            for i, persona in enumerate(default_persona_mix(n_agents))
        ]

        self.events = []
        self.rounds = 0
        self.stats = {
            "total_events": 0,
            "total_purchases": 0,
            "total_spend": 0.0,
            "started_at": datetime.datetime.now().isoformat(),
        }

    @classmethod
    def build(cls, n_agents=N_AGENTS, vocab_path=VOCAB_PATH, **kwargs):
        """Load the policy and vocabulary, then construct the simulation."""
        policy = load_policy()
        return cls(policy, load_vocab(vocab_path), n_agents=n_agents, **kwargs)

    # ── payload ───────────────────────────────────────────────────────────────
    def payload(self, status="running"):
        """The exact shape the dashboard's live feed consumes."""
        elapsed = (
            datetime.datetime.now()
            - datetime.datetime.fromisoformat(self.stats["started_at"])
        ).total_seconds()
        return {
            "status": status,
            "events": list(reversed(self.events)),   # newest first
            "stats": {
                **self.stats,
                "events_per_minute": round((self.stats["total_events"] / max(elapsed, 1)) * 60, 1),
                "elapsed_seconds": round(elapsed),
                "policy_backend": self.policy.backend,
                "agents": self.n_agents,
                "rounds": self.rounds,
            },
        }

    def _record(self, agent, action, product_id, price=None):
        event = {
            "time": datetime.datetime.now().strftime("%H:%M:%S"),
            "agent_id": agent.agent_id,
            "persona": agent.persona,
            "action": action,
            "product": product_id[-10:],     # last 10 chars of the ASIN, for display
            "product_id": product_id,        # full id — what Kafka consumers expect
            "amount": round(price, 2) if price else None,
            "balance": round(agent.balance, 2),
        }
        self.events.append(event)
        del self.events[:-self.max_events]
        self.stats["total_events"] += 1
        if action == "checkout" and price:
            self.stats["total_purchases"] += 1
            self.stats["total_spend"] += price
        return event

    # ── simulation ────────────────────────────────────────────────────────────
    def next_product(self, agent):
        history = list(agent.history)
        while len(history) < WINDOW_SIZE:
            history = [None] + history
        state_ids = [
            self.item_to_idx.get(item, 0) if item else 0
            for item in history[-WINDOW_SIZE:]
        ]
        chosen = sample_with_social(
            self.policy.logits(state_ids),
            agent.social_signal,
            self.item_to_idx,
        )
        product_id = self.idx_to_item.get(chosen, self.product_ids[0])
        agent.social_signal = agent.social_signal[-3:]
        if random.random() < EPSILON:
            product_id = random.choice(self.product_ids)
        return product_id

    def propagate_social(self, agent_id, product_id):
        influenced = propagate_influence(self.graph, agent_id, max_depth=INFLUENCE_MAX_DEPTH)
        for influenced_id, signal in compute_signal_strength(influenced, product_id).items():
            self.agents[influenced_id].social_signal.append(signal)

    def _reset_scheduler(self):
        scheduler = Scheduler()
        for i in range(self.n_agents):
            scheduler.push(timestamp=random.uniform(0, 1.0), agent_id=i, action="browse")
        for agent in self.agents:
            agent.action_history = []
            agent.session_log = []
            agent.history.clear()
        return scheduler

    def iter_events(self):
        """Yield every agent action forever. One round = one pass of the heap."""
        while True:
            self.rounds += 1
            scheduler = self._reset_scheduler()

            while scheduler.heap:
                result = scheduler.pop()
                if result is None:
                    break
                timestamp, agent_id, action = result
                agent = self.agents[agent_id]

                # Funnel gate — an agent cannot check out before it has a cart.
                if not is_action_allowed(agent.action_history, action, self.prereqs):
                    scheduler.push(timestamp + random.uniform(0.5, 3.0), agent_id, "browse")
                    continue

                product_id = self.next_product(agent)

                price = None
                if action == "checkout":
                    price = round(random.uniform(PRICE_MIN, PRICE_MAX), 2)
                    if not agent.can_afford(price):
                        scheduler.push(timestamp + random.uniform(0.5, 3.0), agent_id, "browse")
                        continue
                    agent.complete_purchase(price)
                    self.propagate_social(agent_id, product_id)

                agent.record_action(product_id)
                agent.action_history.append(action)
                agent.session_log.append(product_id)

                yield self._record(agent, action, product_id, price)

                if agent.should_quit():
                    continue

                if action == "checkout":
                    if agent.balance < REFILL_THRESHOLD:
                        agent.refill_balance()
                    scheduler.push(timestamp + random.uniform(0.5, 3.0), agent_id, "browse")
                else:
                    if agent.should_advance_funnel():
                        next_action = ACTIONS[ACTIONS.index(action) + 1]
                    else:
                        next_action = "browse"
                    scheduler.push(timestamp + random.uniform(0.5, 3.0), agent_id, next_action)
