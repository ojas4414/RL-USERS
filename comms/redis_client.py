import json
import os

import redis


def get_redis() -> redis.Redis:
    host = os.getenv("REDIS_HOST", "localhost")
    return redis.Redis(host=host, port=6379, db=0, decode_responses=True)


def set_agent_state(client: redis.Redis, agent_id: int, state: dict, ttl: int = 1800) -> None:
    client.setex(f"agent:{agent_id}", ttl, json.dumps(state))


def get_agent_state(client: redis.Redis, agent_id: int) -> dict:
    raw = client.get(f"agent:{agent_id}")
    return json.loads(raw) if raw else {}
