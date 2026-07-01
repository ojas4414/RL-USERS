"""
simulate_live.py — continuous agent simulation
Agents run in an infinite loop and write every action to
visuals/live_events.json so the dashboard can show a live feed.
"""
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""   # skip pynvml hang; torch build is CPU-only
import json
import random
import datetime
import time
import torch
import redis

sys.path.insert(0, ".")

from comms.kafka_client import get_producer

from agents.agent import Agent
from agents.social_graph import (
    social_graph as build_social_graph,
    bsf as propagate_influence,
    signal_strenght as compute_signal_strength,
)
from agents.scheduler import Scheduler
from agents.funnel import (
    Funnel_graph as FUNNEL_GRAPH,
    build as build_prerequisites,
    allowed as is_action_allowed,
)
from agents.bc_trainer import build_vocab
from agents.rl_trainer import select_action_with_social, build_reverse_vocab
from data.persona_cluster import assign_agent_personas
from data.loader import load_reviews, filter_, sort_split, build, dataset

N_AGENTS      = 50
MODEL_PATH    = "policy_rl.pt"
ACTIONS       = ["browse", "product_detail", "cart", "checkout"]
WINDOW_SIZE   = 3
EPSILON       = 0.25
LIVE_FILE     = "visuals/live_events.json"
MAX_IN_FILE   = 60   # rolling window kept in the JSON
LIVE_CHANNEL  = "rl_users_live"
BROADCAST_INTERVAL = 0.5   # seconds between dashboard updates — keeps the live feed readable
KAFKA_TOPIC   = "agent_actions"   # same topic the gateway publishes to

DEVICE = torch.device("cpu")

_redis = redis.Redis(host=os.environ.get("REDIS_HOST", "127.0.0.1"), port=6379, db=0)
_last_broadcast = 0.0

# simulate_live.py runs on the host (not in docker), so it needs Kafka's host-facing
# listener (localhost:29092), not the container-internal one (kafka:9092) the other
# services use — see docker-compose.yml's KAFKA_ADVERTISED_LISTENERS.
os.environ.setdefault("KAFKA_BOOTSTRAP_SERVERS", "localhost:29092")
try:
    _kafka_producer = get_producer()
except Exception:
    _kafka_producer = None


# ── global state ──────────────────────────────────────────────────────────────
event_buffer = []
stats = {
    "total_events": 0,
    "total_purchases": 0,
    "total_spend": 0.0,
    "started_at": datetime.datetime.now().isoformat(),
}


def _write_live(force=False):
    global _last_broadcast
    now = time.monotonic()
    if not force and (now - _last_broadcast) < BROADCAST_INTERVAL:
        return
    _last_broadcast = now

    elapsed = (
        datetime.datetime.now()
        - datetime.datetime.fromisoformat(stats["started_at"])
    ).total_seconds()
    payload = {
        "events": list(reversed(event_buffer)),   # newest first
        "stats": {
            **stats,
            "events_per_minute": round((stats["total_events"] / max(elapsed, 1)) * 60, 1),
            "elapsed_seconds": round(elapsed),
        },
    }
    tmp_path = LIVE_FILE + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(payload, f)
    os.replace(tmp_path, LIVE_FILE)   # atomic swap — avoids the browser reading a half-written file

    try:
        _redis.publish(LIVE_CHANNEL, json.dumps(payload))
    except Exception:
        pass  # never let dashboard/Redis failure kill the simulation


def record_event(agent, action, product_id, price=None):
    global event_buffer
    ev = {
        "time": datetime.datetime.now().strftime("%H:%M:%S"),
        "agent_id": agent.agent_id,
        "persona": agent.persona,
        "action": action,
        "product": product_id[-10:],     # last 10 chars of ASIN
        "amount": round(price, 2) if price else None,
        "balance": round(agent.balance, 2),
    }
    event_buffer.append(ev)
    event_buffer = event_buffer[-MAX_IN_FILE:]
    stats["total_events"] += 1
    if action == "checkout" and price:
        stats["total_purchases"] += 1
        stats["total_spend"] += price

    if _kafka_producer is not None:
        try:
            _kafka_producer.send(KAFKA_TOPIC, value={
                "agent_id": agent.agent_id,
                "action": action,
                "product_id": product_id,
                "timestamp": datetime.datetime.utcnow().isoformat(),
            })
        except Exception:
            pass  # never let a dead Kafka broker kill the simulation

    _write_live()


# ── RL helpers (same as simulate.py) ──────────────────────────────────────────
def get_next_product(model, agent, item_to_idx, idx_to_item):
    history = list(agent.history)
    while len(history) < WINDOW_SIZE:
        history = [None] + history
    state_ids = [item_to_idx.get(item, 0) if item else 0 for item in history[-WINDOW_SIZE:]]
    state_tensor = torch.tensor([state_ids]).to(DEVICE)
    next_idx, _ = select_action_with_social(model, state_tensor, agent, item_to_idx, temperature=1.5)
    product_id = idx_to_item.get(next_idx.item(), list(item_to_idx.keys())[0])
    agent.social_signal = agent.social_signal[-3:]
    if random.random() < EPSILON:
        product_id = random.choice(list(item_to_idx.keys()))
    return product_id


def propagate_social(graph, agents, agent_id, product_id):
    influenced = propagate_influence(graph, agent_id, max_depth=2)
    signals = compute_signal_strength(influenced, product_id)
    for influenced_id, (prod, strength) in signals.items():
        agents[influenced_id].social_signal.append((prod, strength))


def build_agents(personas):
    agents = []
    for i in range(N_AGENTS):
        agents.append(Agent(agent_id=i, persona=personas[i]))
    return agents


def reset_scheduler(agents):
    scheduler = Scheduler()
    for i in range(N_AGENTS):
        scheduler.push(timestamp=random.uniform(0, 1.0), agent_id=i, action="browse")
    for agent in agents:
        agent.action_history = []
        agent.session_log = []
        agent.history.clear()
    return scheduler


# ── main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading recommendation model...")
    model = torch.jit.load(MODEL_PATH, map_location=DEVICE)
    model.eval()
    print(f"  Running on: {DEVICE}")

    print("Building product vocabulary...")
    if os.path.exists("vocab.json"):
        with open("vocab.json") as _vf:
            item_to_idx = json.load(_vf)
        idx_to_item = build_reverse_vocab(item_to_idx)
        print(f"  Loaded {len(item_to_idx):,} products from vocab.json")
    else:
        print("  vocab.json not found — running full data pipeline (slow)...")
        df = load_reviews("data/raw/All_Beauty.jsonl")
        filtered_df = filter_(df)
        df_sorted, train_cutoff, val_cutoff = sort_split(filtered_df)
        pairs = build(df_sorted, window_size=WINDOW_SIZE)
        train_pairs, _, _ = dataset(pairs, train_cutoff, val_cutoff)
        item_to_idx = build_vocab(train_pairs)
        idx_to_item = build_reverse_vocab(item_to_idx)
        with open("vocab.json", "w") as _vf:
            json.dump(item_to_idx, _vf)
        print(f"  {len(item_to_idx):,} products in catalog — vocab.json saved for next run")
    product_ids = list(item_to_idx.keys())

    print("Setting up social graph and funnel...")
    graph = build_social_graph(N_AGENTS, connection_prob=0.1)
    prereqs = build_prerequisites(FUNNEL_GRAPH)

    user_clusters = (
        {f"u{i}": "average_buyer" for i in range(35)}
        | {f"u{i+35}": "power_buyer" for i in range(10)}
        | {f"u{i+45}": "browser" for i in range(5)}
    )
    personas = assign_agent_personas(N_AGENTS, user_clusters)
    agents = build_agents(personas)

    round_num = 0
    print("Live simulation started. Open http://localhost:8080/visuals/dashboard.html")
    print("Press Ctrl+C to stop.\n")

    while True:
        round_num += 1
        scheduler = reset_scheduler(agents)
        round_events = 0

        while scheduler.heap:
            result = scheduler.pop()
            if result is None:
                break
            timestamp, agent_id, action = result
            agent = agents[agent_id]

            # Funnel gate
            if not is_action_allowed(agent.action_history, action, prereqs):
                scheduler.push(timestamp + random.uniform(0.5, 3.0), agent_id, "browse")
                continue

            # RL product recommendation
            product_id = get_next_product(model, agent, item_to_idx, idx_to_item)

            # Checkout: check affordability
            price = None
            if action == "checkout":
                price = round(random.uniform(9.99, 199.99), 2)   # local price (no HTTP needed)
                if not agent.can_afford(price):
                    scheduler.push(timestamp + random.uniform(0.5, 3.0), agent_id, "browse")
                    continue
                agent.complete_purchase(price)
                propagate_social(graph, agents, agent_id, product_id)

            # Record
            agent.record_action(product_id)
            agent.action_history.append(action)
            agent.session_log.append(product_id)

            record_event(agent, action, product_id, price)

            # Quit?
            if agent.should_quit():
                continue

            # Schedule next
            if action == "checkout":
                if agent.balance < 10.0:
                    agent.refill_balance()
                scheduler.push(timestamp + random.uniform(0.5, 3.0), agent_id, "browse")
            else:
                if agent.should_advance_funnel():
                    next_action = ACTIONS[ACTIONS.index(action) + 1]
                else:
                    next_action = "browse"
                scheduler.push(timestamp + random.uniform(0.5, 3.0), agent_id, next_action)

            round_events += 1

        print(f"Round {round_num} complete — {round_events} events, "
              f"{stats['total_purchases']} total purchases so far")
