import datetime
import os
import time
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from comms.kafka_client import get_producer, publish_event
from live.runner import LiveRunner

_producer = None

# One runner per process. It owns the simulation thread and the bus every
# websocket reads from; a cold start builds a new one from scratch, which is
# exactly what we want on a free instance that sleeps between visitors.
live = LiveRunner()


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
    # Starts the simulation in the background. Loading the policy takes a second
    # or two, and doing it here rather than on first request means the feed is
    # already flowing by the time anyone opens the dashboard.
    await live.start()
    try:
        yield
    finally:
        await live.stop()


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


@app.get("/live/status")
async def live_status():
    """What the simulation is doing, and why, without opening a websocket."""
    return live.snapshot()


@app.websocket("/ws/live")
async def live_simulation(websocket: WebSocket):
    """Stream simulation snapshots.

    This used to subscribe to Redis directly, which meant that with no Redis
    reachable the handler raised on connect and the socket closed immediately —
    the dashboard then fell back to polling a file that does not exist in the
    deployed image. It now reads the in-process bus, which the runner feeds from
    either the local simulation or a Redis relay.
    """
    await websocket.accept()
    queue = live.bus.subscribe()
    try:
        while True:
            await websocket.send_text(await queue.get())
    except WebSocketDisconnect:
        pass
    finally:
        live.bus.unsubscribe(queue)


if os.path.isdir("visuals"):
    app.mount("/dashboard", StaticFiles(directory="visuals", html=True), name="dashboard")

    @app.get("/")
    async def root():
        return RedirectResponse(url="/dashboard/dashboard.html")
