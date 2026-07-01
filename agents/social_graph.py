import random 
from collections import deque
def social_graph(number_agents:int,connection_prob:float=0.1):
    graph={i:[] for i in range(number_agents)}
    for i in range(number_agents):
        for j in range(i+1,number_agents):
            if random.random() <connection_prob:
                graph[i].append(j)
                graph[j].append(i)
    return graph 

def bsf(social_graph:dict,source_agent:int, max_depth: int =3):
    visited = {source_agent: 0}      # agent_id → depth from source
    queue = deque([(source_agent, 0)])

    while queue:
        current_agent, depth = queue.popleft()

        if depth >= max_depth:
            continue   # don't expand further, we've hit the limit

        for neighbor in social_graph[current_agent]:
            if neighbor not in visited:
                visited[neighbor] = depth + 1
                queue.append((neighbor, depth + 1))

    del visited[source_agent]   # don't include the source itself in results
    return visited

def signal_strenght(influenced_agents:dict,product_id:str):
    signals = {}
    for agent_id, depth in influenced_agents.items():
        strength = 1 / (2 ** depth)
        signals[agent_id] = (product_id, strength)
    return signals



    