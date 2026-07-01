import datetime
import os
import time
from contextlib import asynccontextmanager

import httpx
import redis.asyncio as aioredis
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from comms.kafka_client import get_producer, publish_event

_producer = None
REDIS_HOST = os.environ.get("REDIS_HOST", "127.0.0.1")
LIVE_CHANNEL = "rl_users_live"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _producer
    if os.environ.get("KAFKA_HOST"):
        for attempt in range(6):
            try:
                _producer = get_producer()
                break
            except Exception:
                if attempt < 5:
                    time.sleep(5)
    yield


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SERVICES = {
    "product": "http://product_service:8001",
    "cart": "http://cart_service:8002",
    "order": "http://order_service:8003",
    "session": "http://session_service:8004",
}


async def _forward(method: str, url: str, body: dict | None = None) -> JSONResponse:
    async with httpx.AsyncClient() as client:
        resp = await client.request(method, url, json=body)
    return JSONResponse(content=resp.json(), status_code=resp.status_code)


def _publish(body: dict, action: str) -> None:
    if _producer is None:
        return
    publish_event(_producer, "agent_actions", {
        "agent_id": body.get("agent_id"),
        "action": action,
        "product_id": body.get("product_id"),
        "timestamp": datetime.datetime.utcnow().isoformat(),
    })


@app.get("/health")
async def health():
    return {"status": "alive"}


@app.get("/products")
async def get_products(request: Request):
    response = await _forward("GET", f"{SERVICES['product']}/products")
    _publish({}, "get_products")
    return response


@app.post("/cart")
async def post_cart(request: Request):
    body = await request.json()
    response = await _forward("POST", f"{SERVICES['cart']}/cart", body)
    _publish(body, "add_to_cart")
    return response


@app.post("/order")
async def post_order(request: Request):
    body = await request.json()
    response = await _forward("POST", f"{SERVICES['order']}/order", body)
    _publish(body, "place_order")
    return response


@app.post("/session/start")
async def post_session_start(request: Request):
    body = await request.json()
    response = await _forward("POST", f"{SERVICES['session']}/session/start", body)
    _publish(body, "session_start")
    return response


@app.websocket("/ws/live")
async def live_simulation(websocket: WebSocket):
    await websocket.accept()
    r = aioredis.Redis(host=REDIS_HOST, port=6379, db=0)
    pubsub = r.pubsub()
    await pubsub.subscribe(LIVE_CHANNEL)
    try:
        async for message in pubsub.listen():
            if message["type"] == "message":
                await websocket.send_text(message["data"].decode())
    except WebSocketDisconnect:
        pass
    finally:
        await pubsub.unsubscribe(LIVE_CHANNEL)
        await r.aclose()
