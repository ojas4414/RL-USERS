import datetime
import json
import os

import redis
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

_redis = redis.Redis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=6379,
    db=0,
    decode_responses=True,
)


class SessionStart(BaseModel):
    agent_id: int
    persona: str


class SessionAction(BaseModel):
    agent_id: int
    action: str
    product_id: str


def _key(agent_id: int) -> str:
    return f"session:{agent_id}"


@app.get("/health")
def health():
    return {"status": "alive"}


@app.post("/session/start")
def start_session(req: SessionStart):
    session = {
        "agent_id": req.agent_id,
        "persona": req.persona,
        "started_at": datetime.datetime.utcnow().isoformat(),
        "actions": [],
    }
    _redis.setex(_key(req.agent_id), 3600, json.dumps(session))
    return session


@app.post("/session/action")
def add_action(req: SessionAction):
    key = _key(req.agent_id)
    raw = _redis.get(key)
    if not raw:
        raise HTTPException(status_code=404, detail="Session not found")
    session = json.loads(raw)
    session["actions"].append({"action": req.action, "product_id": req.product_id})
    ttl = _redis.ttl(key)
    _redis.setex(key, ttl if ttl > 0 else 3600, json.dumps(session))
    return session


@app.get("/session/{agent_id}")
def get_session(agent_id: int):
    raw = _redis.get(_key(agent_id))
    if not raw:
        raise HTTPException(status_code=404, detail="Session not found")
    return json.loads(raw)
