Funnel_graph={
    "browse":["product_detail"],
    "product_detail": ["cart"],
    "cart": ["checkout"],
    "checkout": []
}



def build(graph: dict)-> dict:
    prereqs = {node: [] for node in graph}

    for node, children in graph.items():
        for child in children:
            prereqs[child].append(node)
    return prereqs

def allowed(agent_history: list,action:str, prereqs: dict)-> bool:
    required_steps = prereqs.get(action,[])

    for step in required_steps:
        if step not in agent_history:
            return False
        
    return True
def topo_sort(graph: dict)-> list:

    visited = set()
    temp_mark = set()

    order = []

    def dfs(node):
        if node in temp_mark:
            raise ValueError(f"Cycle detected at node: {node}")
        if node in visited:
            return

        temp_mark.add(node)# this is used to detect cyclic graph acts as storage  temporaty for to check if it has visited thee current node earliear ot not
        for neighbor in graph[node]:
            dfs(neighbor)
        temp_mark.remove(node)

        visited.add(node)
        order.append(node)

    for node in graph:
        if node not in visited:
            dfs(node)
    order.reverse()

    return order