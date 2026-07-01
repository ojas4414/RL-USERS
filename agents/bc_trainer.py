import torch
import torch.nn as nn


class policyNetwork(nn.Module):

    def __init__(self, vocab_size: int, embedding_dim: int = 64, window_size: int = 3):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim
        )
        self.window_size = window_size

        self.position_emb = nn.Embedding(
            num_embeddings=window_size,
            embedding_dim=embedding_dim
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=4,
            dim_feedforward=128,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.output = nn.Linear(embedding_dim, vocab_size)

    def forward(self, state_ids):
        batch_size = state_ids.size(0)

        item_embeds = self.embedding(state_ids)

        positions = torch.arange(self.window_size).unsqueeze(0).expand(batch_size, -1)
        pos_embeds = self.position_emb(positions)

        combined = item_embeds + pos_embeds

        encoded = self.transformer(combined)

        summary = encoded[:, -1, :]

        logits = self.output(summary)

        return logits


def build_vocab(pairs: list) -> dict:
    unique_items = set()
    for p in pairs:
        unique_items.update(p["state"])
        unique_items.add(p["action"])

    item_to_idx = {item: idx for idx, item in enumerate(sorted(unique_items))}
    return item_to_idx


def tensor(pairs: list, item_to_idx: dict):
    states = []
    actions = []

    for p in pairs:
        state_ids = [item_to_idx[item] for item in p["state"]]
        action_id = item_to_idx[p["action"]]
        states.append(state_ids)
        actions.append(action_id)

    state_tensor = torch.tensor(states)
    action_tensor = torch.tensor(actions)
    return state_tensor, action_tensor


def train(model, state_tensor, action_tensor, epoch=10, batch_size=64, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    samples = state_tensor.size(0)

    for e in range(epoch):
        permutation = torch.randperm(samples)
        total_loss = 0.0
        for i in range(0, samples, batch_size):
            indices = permutation[i:i + batch_size]
            batch_states = state_tensor[indices]
            batch_actions = action_tensor[indices]

            optimizer.zero_grad()
            logits = model(batch_states)
            loss = criterion(logits, batch_actions)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print(f"Epoch {e+1}/{epoch} — loss: {total_loss:.4f}")
