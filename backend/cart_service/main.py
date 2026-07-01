import json
import os

import redis
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

_redis = redis.Redis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=6379,
    db=0,
    decode_responses=True,
)


class CartItem(BaseModel):
    agent_id: int
    product_id: str


def _key(agent_id: int) -> str:
    return f"cart:{agent_id}"


@app.get("/health")
def health():
    return {"status": "alive"}


@app.get("/cart/{agent_id}")
def get_cart(agent_id: int):
    raw = _redis.get(_key(agent_id))
    return json.loads(raw) if raw else []


@app.post("/cart")
def add_to_cart(item: CartItem):
    key = _key(item.agent_id)
    raw = _redis.get(key)
    cart: list = json.loads(raw) if raw else []
    cart.append(item.product_id)
    _redis.setex(key, 1800, json.dumps(cart))
    return {"agent_id": item.agent_id, "cart": cart}


@app.delete("/cart/{agent_id}")
def clear_cart(agent_id: int):
    _redis.delete(_key(agent_id))
    return {"agent_id": agent_id, "cart": []}
