import sys
import torch
import requests
import json
import random
from collections import Counter

sys.path.insert(0, ".")

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
from validation.metrics import full_validation_report

N_AGENTS = 50
GATEWAY_URL = "http://localhost:8000"
MODEL_PATH = "policy_rl.pt"
ACTIONS = ["browse", "product_detail", "cart", "checkout"]
WINDOW_SIZE = 3
MAX_ACTIONS = 500
EPSILON = 0.25


def get_next_product(model, agent, item_to_idx, idx_to_item):
    """RL model recommends next product, with epsilon-greedy exploration."""
    history = list(agent.history)
    while len(history) < WINDOW_SIZE:
        history = [None] + history
    state_ids = [
        item_to_idx.get(item, 0) if item is not None else 0
        for item in history[-WINDOW_SIZE:]
    ]
    state_tensor = torch.tensor([state_ids])

    next_action_idx, _ = select_action_with_social(
        model, state_tensor, agent, item_to_idx, temperature=1.5
    )
    product_id = idx_to_item.get(next_action_idx.item(), list(item_to_idx.keys())[0])

    # Trim stale social signals — keep only last 3 to prevent persistent bias
    agent.social_signal = agent.social_signal[-3:]

    if random.random() < EPSILON:
        product_id = random.choice(list(item_to_idx.keys()))

    return product_id


def get_price(product_id):
    try:
        price_resp = requests.get(f"{GATEWAY_URL}/products/{product_id}", timeout=3)
        return price_resp.json().get("price", 0.0)
    except Exception:
        return 0.0


def propagate_social(graph, agents, agent_id, product_id):
    influenced = propagate_influence(graph, agent_id, max_depth=2)
    signals = compute_signal_strength(influenced, product_id)
    for influenced_id, (prod, strength) in signals.items():
        agents[influenced_id].social_signal.append((prod, strength))


if __name__ == "__main__":

    # Step 1 — Load trained model
    print("Loading recommendation model...")
    model = torch.jit.load(MODEL_PATH)
    model.eval()

    # Step 2 — Rebuild vocab from same data pipeline as train.py
    print("Building product vocabulary...")
    df = load_reviews("data/raw/All_Beauty.jsonl")
    filtered_df = filter_(df)
    df_sorted, train_cutoff, val_cutoff = sort_split(filtered_df)
    pairs = build(df_sorted, window_size=WINDOW_SIZE)
    train_pairs, _, _ = dataset(pairs, train_cutoff, val_cutoff)
    item_to_idx = build_vocab(train_pairs)
    idx_to_item = build_reverse_vocab(item_to_idx)
    print(f"  Products in catalog: {len(item_to_idx):,}")

    # Save vocab so simulate_live.py can skip the 700K-row reload
    with open("vocab.json", "w") as _vf:
        json.dump(item_to_idx, _vf)

    product_ids = list(item_to_idx.keys())

    # Step 3 — Social graph and funnel prereqs
    print("Setting up shopper network and purchase funnel...")
    graph = build_social_graph(N_AGENTS, connection_prob=0.1)
    prereqs = build_prerequisites(FUNNEL_GRAPH)

    # Step 4 — Create 50 agents with persona-realistic budgets
    print("Creating simulated shoppers...")
    user_clusters = (
        {f"u{i}": "average_buyer" for i in range(35)}
        | {f"u{i+35}": "power_buyer" for i in range(10)}
        | {f"u{i+45}": "browser" for i in range(5)}
    )
    personas = assign_agent_personas(N_AGENTS, user_clusters)

    agents = []
    for i in range(N_AGENTS):
        agent = Agent(agent_id=i, persona=personas[i])
        agents.append(agent)
    print(f"  {N_AGENTS} shoppers created — "
          f"{sum(p == 'power_buyer' for p in personas)} power buyers, "
          f"{sum(p == 'average_buyer' for p in personas)} regular buyers, "
          f"{sum(p == 'browser' for p in personas)} browsers")

    # Step 5 — Schedule initial browse for every agent
    print("Starting simulation...")
    scheduler = Scheduler()
    for i in range(N_AGENTS):
        scheduler.push(timestamp=random.uniform(0, 1.0), agent_id=i, action="browse")

    # Step 6 — Simulation loop
    print(f"Running {MAX_ACTIONS} shopping events across {N_AGENTS} agents...")
    total_actions = 0

    while scheduler.heap and total_actions < MAX_ACTIONS:
        result = scheduler.pop()
        if result is None:
            break
        timestamp, agent_id, action = result
        agent = agents[agent_id]

        # a. Funnel gate
        if not is_action_allowed(agent.action_history, action, prereqs):
            scheduler.push(timestamp + random.uniform(0.5, 3.0), agent_id, "browse")
            continue

        # b. RL model recommends next product
        product_id = get_next_product(model, agent, item_to_idx, idx_to_item)

        # c. Budget constraint at checkout
        if action == "checkout":
            price = get_price(product_id)
            if not agent.can_afford(price):
                # Can't afford — bounce back to browse
                scheduler.push(timestamp + random.uniform(0.5, 3.0), agent_id, "browse")
                continue
            agent.complete_purchase(price)

        # d. Record action in both histories
        agent.record_action(product_id)
        agent.action_history.append(action)
        agent.session_log.append(product_id)

        # e. Social propagation on checkout
        if action == "checkout":
            propagate_social(graph, agents, agent_id, product_id)

        # f. Persona-weighted early exit — drives realistic session length variance
        if agent.should_quit():
            continue

        # g. Schedule next action
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

        total_actions += 1
        if total_actions % 50 == 0:
            print(f"  {total_actions} of {MAX_ACTIONS} events processed...")

    print(f"  Simulation complete — {total_actions} events across {N_AGENTS} shoppers")

    # Step 7 — Collect agent sessions (skip agents with no interactions)
    agent_sessions = [agent.session_log for agent in agents if agent.session_log]

    # Step 8 — Real sessions: first 50 users' item sequences from filtered data
    print("Loading real shopping sessions for comparison...")
    user_sequences = (
        df_sorted.groupby("user_id")["item_id"]
        .apply(list)
        .to_dict()
    )
    real_sessions = list(user_sequences.values())[:50]

    all_purchases = [p for agent in agents for p in agent.session_log if p != "checkout"]
    top_trending = Counter(all_purchases).most_common(5)
    print("Top 5 trending products:")
    for rank, (product, count) in enumerate(top_trending, 1):
        print(f"  {rank}. {product}  ({count} interactions)")

    # Step 9 — Validation report
    print("Running behavioral validation...")
    report = full_validation_report(real_sessions, agent_sessions, agents=agents)

    # Enrich report with raw data needed by the dashboard
    report["trending_products"] = [
        {"product": product, "count": count} for product, count in top_trending
    ]
    report["sim_session_lengths"] = [len(s) for s in agent_sessions]
    report["real_session_lengths"] = [len(s) for s in real_sessions]

    # Step 10 — Save report (root + visuals/ for the dashboard)
    with open("validation_report.json", "w") as f:
        json.dump(report, f, indent=2)
    with open("visuals/validation_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print("Report saved to validation_report.json")

    from visualize import run_all_visualizations
    agent_personas = [agent.persona for agent in agents]
    run_all_visualizations(graph, agents, real_sessions, agent_sessions, agent_personas)
    print("Charts saved to visuals/")
