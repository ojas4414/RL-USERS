import pandas as pd


def build_graph(df: pd.DataFrame) -> dict:
    graph = {}
    for _, group in df.groupby("user_id"):
        items = group["item_id"].tolist()

        for i in range(len(items) - 1):
            current = items[i]
            next_ = items[i + 1]
            if current not in graph:
                graph[current] = {}
            if next_ not in graph[current]:
                graph[current][next_] = 0

            graph[current][next_] += 1

    return graph


def top_neighbors(graph: dict, product_id: str, top_k: int = 5):
    neighbors = graph[product_id]

    ranked = sorted(
        neighbors.items(),
        key=lambda pair: pair[1],
        reverse=True
    )

    return ranked[:top_k]
