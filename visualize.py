import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from collections import Counter

PERSONA_COLORS = {
    "power_buyer":   "red",
    "average_buyer": "blue",
    "browser":       "green",
}


def plot_social_graph(social_graph: dict, agent_personas: list,
                      save_path="visuals/social_graph.png"):
    G = nx.DiGraph()
    for node, neighbors in social_graph.items():
        G.add_node(node)
        if isinstance(neighbors, dict):
            for neighbor, weight in neighbors.items():
                G.add_edge(node, neighbor, weight=weight)

    pos = nx.spring_layout(G, seed=42)
    node_colors = [PERSONA_COLORS.get(agent_personas[n], "gray") for n in G.nodes()]
    node_sizes  = [300 + 50 * G.degree(n) for n in G.nodes()]

    edge_data    = list(G.edges(data=True))
    edge_weights = [d.get("weight", 1) for _, _, d in edge_data]
    max_w        = max(edge_weights) if edge_weights else 1
    edge_widths  = [1.0 + 2.0 * w / max_w for w in edge_weights]

    fig, ax = plt.subplots(figsize=(12, 10))
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes,
                           alpha=0.8, ax=ax)
    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.4,
                           arrows=True, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=7, ax=ax)

    for persona, color in PERSONA_COLORS.items():
        ax.scatter([], [], c=color, label=persona, s=100)
    ax.legend(loc="upper left", fontsize=10)
    ax.set_title("Shopper Social Network — Who Influences Whom",
                 fontsize=13)
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved {save_path}")


def plot_purchase_heatmap(agents: list, top_n: int = 20,
                          save_path="visuals/purchase_heatmap.png"):
    all_items = [item for agent in agents for item in agent.session_log if item != "checkout"]
    if not all_items:
        print("  No purchases — skipping heatmap")
        return

    top_products = [item for item, _ in Counter(all_items).most_common(top_n)]
    matrix = np.zeros((len(agents), len(top_products)))
    for i, agent in enumerate(agents):
        for j, product in enumerate(top_products):
            matrix[i][j] = agent.session_log.count(product)

    fig, ax = plt.subplots(figsize=(14, 8))
    im = ax.imshow(matrix, aspect="auto", cmap="hot", interpolation="nearest")
    plt.colorbar(im, ax=ax, label="Interaction count")

    ax.set_xticks(range(len(top_products)))
    ax.set_xticklabels([p[:10] for p in top_products], rotation=45,
                       ha="right", fontsize=7)
    ax.set_yticks(range(len(agents)))
    ax.set_yticklabels([str(i) for i in range(len(agents))], fontsize=6)
    ax.set_xlabel("Product")
    ax.set_ylabel("Shopper")
    ax.set_title(
        "Purchase Heatmap — Shopper × Product Interaction Count",
        fontsize=12
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved {save_path}")


def plot_trending_products(agents: list, top_n: int = 10,
                           save_path="visuals/trending_products.png"):
    all_items = [item for agent in agents for item in agent.session_log if item != "checkout"]
    if not all_items:
        print("  No purchases — skipping trending products chart")
        return

    top_items = Counter(all_items).most_common(top_n)
    products  = [p[:12] for p, _ in reversed(top_items)]
    counts    = [c for _, c in reversed(top_items)]

    cmap   = plt.cm.Blues
    colors = [cmap(0.4 + 0.6 * c / max(counts)) for c in counts]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(products, counts, color=colors)
    ax.set_xlabel("Interaction Count")
    ax.set_ylabel("Product ASIN (truncated)")
    ax.set_title(
        "Top Trending Products",
        fontsize=12
    )
    for bar, count in zip(bars, counts):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                str(count), va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved {save_path}")


def plot_session_length_distribution(real_sessions: list, sim_sessions: list,
                                     save_path="visuals/session_lengths.png"):
    real_lengths = [len(s) for s in real_sessions]
    sim_lengths  = [len(s) for s in sim_sessions]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(real_lengths, bins=20, alpha=0.6, color="blue",   label="Real sessions")
    ax.hist(sim_lengths,  bins=20, alpha=0.6, color="orange", label="Simulated sessions")

    if real_lengths:
        ax.axvline(np.mean(real_lengths), color="blue", linestyle="--", linewidth=2,
                   label=f"Real mean ({np.mean(real_lengths):.1f})")
    if sim_lengths:
        ax.axvline(np.mean(sim_lengths), color="orange", linestyle="--", linewidth=2,
                   label=f"Sim mean ({np.mean(sim_lengths):.1f})")

    ax.set_xlabel("Session Length (interactions)")
    ax.set_ylabel("Frequency")
    ax.set_title("Session Length: Real Shoppers vs Simulated", fontsize=12)
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved {save_path}")


def run_all_visualizations(social_graph, agents, real_sessions,
                            sim_sessions, agent_personas):
    os.makedirs("visuals", exist_ok=True)
    plot_social_graph(social_graph, agent_personas)
    plot_purchase_heatmap(agents)
    plot_trending_products(agents)
    plot_session_length_distribution(real_sessions, sim_sessions)
