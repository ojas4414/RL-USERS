import sys
import torch

sys.path.insert(0, ".")

from data.loader import load_reviews, filter_, sort_split, build, dataset
from data.graph_builder import build_graph
from data.persona_cluster import extract_user_features, cluster_users, label_clusters
from agents.bc_trainer import build_vocab, tensor, policyNetwork, train
from agents.rl_trainer import build_reverse_vocab, train_rl


if __name__ == "__main__":
    # Step 1 — Data loading
    print("Step 1: Loading reviews...")
    df = load_reviews("data/raw/All_Beauty.jsonl")
    print(f"  Loaded {len(df)} reviews")

    print("  Applying 5-core filter...")
    filtered_df = filter_(df)
    print(f"  Filtered to {len(filtered_df)} reviews")

    print("  Sorting and computing time cutoffs...")
    df_sorted, train_cutoff, val_cutoff = sort_split(filtered_df)

    print("  Building state-action pairs (window=3)...")
    pairs = build(df_sorted, window_size=3)
    print(f"  Built {len(pairs)} pairs")

    print("  Splitting pairs by time...")
    train_pairs, val_pairs, test_pairs = dataset(pairs, train_cutoff, val_cutoff)
    print(f"  Train: {len(train_pairs)}  Val: {len(val_pairs)}  Test: {len(test_pairs)}")

    # Step 2 — Product graph
    print("\nStep 2: Building product graph...")
    graph = build_graph(df_sorted)
    print(f"  Graph nodes: {len(graph)}")

    # Step 3 — Persona clustering
    print("\nStep 3: Clustering user personas...")
    user_features = extract_user_features(train_pairs)
    user_clusters = cluster_users(user_features, n_clusters=3)
    user_personas = label_clusters(user_clusters, user_features)

    persona_counts = {}
    for persona in user_personas.values():
        persona_counts[persona] = persona_counts.get(persona, 0) + 1
    print("  Persona distribution:")
    for persona, count in sorted(persona_counts.items()):
        print(f"    {persona}: {count} users")

    # Step 4 — Vocabulary
    print("\nStep 4: Building item vocabulary...")
    item_to_idx = build_vocab(train_pairs)
    print(f"  Vocab size: {len(item_to_idx)}")

    # Step 5 — Tensor conversion
    print("\nStep 5: Converting pairs to tensors...")
    state_tensor, action_tensor = tensor(train_pairs, item_to_idx)
    print(f"  state_tensor: {tuple(state_tensor.shape)}  action_tensor: {tuple(action_tensor.shape)}")

    # Step 6 — BC Training
    print("\nStep 6: Behavioural cloning (BC) training...")
    model = policyNetwork(vocab_size=len(item_to_idx))
    train(model, state_tensor, action_tensor, epoch=10, batch_size=64)
    torch.jit.script(model).save("policy_bc.pt")
    print("BC training complete. Model saved to policy_bc.pt")

    # Step 7 — RL Fine-tuning
    print("\nStep 7: RL fine-tuning (PPO)...")
    idx_to_item = build_reverse_vocab(item_to_idx)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    train_rl(model, optimizer, state_tensor, item_to_idx, graph, episodes=5)
    torch.jit.script(model).save("policy_rl.pt")
    print("RL fine-tuning complete. Model saved to policy_rl.pt")

    # Step 8 — Summary
    print("\n--- Run summary ---")
    print(f"  Total reviews loaded : {len(df)}")
    print(f"  Vocab size           : {len(item_to_idx)}")
    print(f"  Train pairs          : {len(train_pairs)}")
    print(f"  Val pairs            : {len(val_pairs)}")
    print(f"  Test pairs           : {len(test_pairs)}")
    print("Run docker-compose up to start simulation")
