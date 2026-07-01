import torch 
import torch.nn as nn
from torch.distributions import Categorical


def reward(graph:dict,prev_item:str,chosen_item:str):

    if prev_item not in graph:
        return 0.0
    
    neighbours=graph[prev_item]

    total_transitions=sum(neighbours.values())

    if total_transitions ==0:
        return 0.0
    
    weight=neighbours.get(chosen_item,0)
    return weight / total_transitions


def select_action(model,state_tensor):


    logits=model(state_tensor)
    distribution=Categorical(logits=logits)

    action= distribution.sample()

    log_prob = distribution.log_prob(action)#given the action that got sampled, this returns the LOG of how likely that specific outcome was.

    return action ,log_prob
def ppo_update(model,optimizer,states,actions,old_log_probs, rewards,epsilon=0.2):
    logits=model(states)
    distribution = Categorical(logits=logits)
    
    new_log_probs=distribution.log_prob(actions)
    ratio = torch.exp(new_log_probs - old_log_probs)

    unclipped = ratio * rewards
    clipped = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * rewards

    loss = -torch.min(unclipped, clipped).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def select_action_with_social(model, state_tensor, agent, item_to_idx, temperature=1.5):
    """
    Like select_action but boosts products currently trending in the
    agent's social_signal — simulating "I saw this on social media"
    effect. This activates the BFS social contagion mechanism properly.
    """
    with torch.no_grad():
        logits = model(state_tensor).clone() / temperature

        for product_id, strength in agent.social_signal[-5:]:
            if product_id in item_to_idx:
                idx = item_to_idx[product_id]
                logits[0][idx] += strength * 2.0

        distribution = Categorical(logits=logits)
        action = distribution.sample()
        log_prob = distribution.log_prob(action)
    return action, log_prob


def build_reverse_vocab(item_to_idx: dict) -> dict:

    return {idx: item for item, idx in item_to_idx.items()}

def collect_experience(model, state_tensor, graph, idx_to_item):
    states=[]
    actions =[]
    old_log_probs=[]
    rewards=[]

    for i in range(state_tensor.size(0)):
        state= state_tensor[i].unsqueeze(0)
        action,log_prob=select_action(model, state)

        prev_item=idx_to_item[state[0,-1].item()]
        chosen_item = idx_to_item[action.item()]
        reward_value = reward(graph, prev_item, chosen_item)

        states.append(state.squeeze(0))
        actions.append(action.squeeze(0))
        old_log_probs.append(log_prob.squeeze(0))
        rewards.append(reward_value)

    return (
        torch.stack(states),
        torch.stack(actions),
        torch.stack(old_log_probs),
        torch.tensor(rewards)
    )


def train_rl(model,optimizer,state_tensor,item_to_idx,graph,episodes=10):
    idx_to_item=build_reverse_vocab(item_to_idx)
    for episode in range(episodes):
        states, actions, old_log_probs, rewards = collect_experience(
            model, state_tensor, graph, idx_to_item
        )

        loss = ppo_update(model, optimizer, states, actions, old_log_probs, rewards)

        avg_reward = rewards.mean().item()
        print(f"Episode {episode+1}/{episodes} — avg reward: {avg_reward:.4f} — loss: {loss:.4f}")





