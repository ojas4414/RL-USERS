import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from collections import defaultdict


def extract_user_features(pairs: list) -> dict:
    """
    Extracts 4 behavioral features per user from their
    state-action pairs. These features capture HOW users
    shop, not WHAT they buy — that's what K-means clusters on.
    """
    user_sessions = defaultdict(list)
    for p in pairs:
        user_sessions[p["user_id"]].append(p["action"])

    user_features = {}
    for user_id, actions in user_sessions.items():
        total_actions    = len(actions)
        unique_products  = len(set(actions))
        checkouts        = actions.count("checkout")
        conversion_rate  = checkouts / total_actions if total_actions > 0 else 0.0

        user_features[user_id] = {
            "avg_session_length": total_actions,
            "unique_products":    unique_products,
            "conversion_rate":    conversion_rate,
            "checkout_count":     checkouts
        }

    return user_features


def cluster_users(user_features: dict, n_clusters: int = 3) -> dict:
    """
    Runs K-means on behavioral features to discover natural
    user personas in the data. Returns {user_id: cluster_id}.
    """
    user_ids = list(user_features.keys())

    feature_matrix = np.array([
        [
            user_features[uid]["avg_session_length"],
            user_features[uid]["unique_products"],
            user_features[uid]["conversion_rate"],
            user_features[uid]["checkout_count"]
        ]
        for uid in user_ids
    ])

    scaler = StandardScaler()
    feature_matrix = scaler.fit_transform(feature_matrix)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(feature_matrix)

    return {user_id: int(label) for user_id, label in zip(user_ids, labels)}


def label_clusters(user_clusters: dict, user_features: dict) -> dict:
    """
    Gives human-readable names to each cluster based on its
    average behavioral features — so cluster 0 becomes
    "power_buyer", cluster 1 becomes "browser", etc.
    """
    cluster_stats = defaultdict(list)
    for user_id, cluster_id in user_clusters.items():
        cluster_stats[cluster_id].append(
            user_features[user_id]["conversion_rate"]
        )

    cluster_avg_conversion = {
        cid: np.mean(rates)
        for cid, rates in cluster_stats.items()
    }

    sorted_clusters = sorted(
        cluster_avg_conversion.items(),
        key=lambda x: x[1],
        reverse=True
    )

    persona_names = ["power_buyer", "average_buyer", "browser"]
    cluster_to_persona = {
        cid: persona_names[i]
        for i, (cid, _) in enumerate(sorted_clusters)
    }

    return {
        user_id: cluster_to_persona[cluster_id]
        for user_id, cluster_id in user_clusters.items()
    }


def assign_agent_personas(n_agents: int, user_clusters: dict) -> list:
    """
    Assigns a persona to each of the N simulated agents by
    randomly sampling from the real user persona distribution.
    Agents get the same MIX of personas as real users had.
    """
    personas = list(user_clusters.values())
    return [personas[i % len(personas)] for i in range(n_agents)]