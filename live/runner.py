"""
Starts the live simulation alongside the web server and keeps the feed fed.

The simulation is synchronous CPU-bound Python, so it runs on a daemon thread
rather than as an asyncio task — an `async def` wrapper around this loop would
block the event loop between awaits and stall the very websockets it feeds.
Payloads cross back onto the loop through ``call_soon_threadsafe``.

Everything here is built to survive a Render free-tier cold start: state lives
in the thread, nothing is persisted, and a fresh process simply starts a fresh
simulation. There is no resume path to get wrong.

Mode is chosen by FF_LIVE_MODE:

    auto       (default) run in-process; if the model cannot load, bridge from
               Redis instead; if neither works, report why to the dashboard
    inprocess  always run the simulation here
    external   never run it here — subscribe to Redis and relay whatever
               simulate_live.py publishes (the docker-compose topology)
    off        no live feed at all
"""
import asyncio
import datetime
import json
import os
import threading
import time
import traceback

from live.bus import LiveBus
from live.engine import LiveSimulation

LIVE_CHANNEL = os.environ.get("FF_LIVE_CHANNEL", "rl_users_live")

# How fast agents act. The simulation can run thousands of events per second,
# which is neither readable on a dashboard nor kind to a 0.1-CPU instance.
EVENTS_PER_SECOND = float(os.environ.get("FF_LIVE_EVENTS_PER_SECOND", "8"))
# How often the browser gets a new snapshot. Unchanged from simulate_live.py.
BROADCAST_INTERVAL = float(os.environ.get("FF_BROADCAST_INTERVAL", "0.5"))
MODE = os.environ.get("FF_LIVE_MODE", "auto").strip().lower()


class LiveRunner:
    """Owns the simulation thread (or the Redis bridge) for one process."""

    def __init__(self, bus=None, mode=MODE):
        self.bus = bus or LiveBus()
        self.mode = mode
        self.status = "starting"
        self.detail = ""
        self._loop = None
        self._thread = None
        self._redis_task = None
        self._stop = threading.Event()

    # ── lifecycle ─────────────────────────────────────────────────────────────
    async def start(self):
        self._loop = asyncio.get_running_loop()

        if self.mode == "off":
            self._set_status("disabled", "live feed disabled by FF_LIVE_MODE=off")
            return

        if self.mode == "external":
            self._start_redis_bridge()
            return

        self._thread = threading.Thread(
            target=self._run_simulation, name="live-simulation", daemon=True
        )
        self._thread.start()

    async def stop(self):
        self._stop.set()
        if self._redis_task is not None:
            self._redis_task.cancel()
            try:
                await self._redis_task
            except (asyncio.CancelledError, Exception):
                pass
        if self._thread is not None:
            # Daemon thread, so a hung join must not hold up shutdown; the pacing
            # sleep means one broadcast interval is plenty in practice.
            self._thread.join(timeout=2.0)

    # ── status ────────────────────────────────────────────────────────────────
    def _set_status(self, status, detail=""):
        self.status = status
        self.detail = detail
        self._publish({"status": status, "detail": detail, "events": [], "stats": {}})

    def snapshot(self):
        """Status for the /live/status endpoint."""
        return {
            "status": self.status,
            "detail": self.detail,
            "mode": self.mode,
            "subscribers": self.bus.subscriber_count,
            "events_per_second": EVENTS_PER_SECOND,
        }

    # ── publishing ────────────────────────────────────────────────────────────
    def _publish(self, payload):
        """Encode once here, then fan the string out; websockets just send it."""
        text = payload if isinstance(payload, str) else json.dumps(payload)
        if self._loop is None:
            return
        try:
            self._loop.call_soon_threadsafe(self.bus.publish, text)
        except RuntimeError:
            pass    # loop already closed — process is shutting down

    # ── in-process simulation ─────────────────────────────────────────────────
    def _run_simulation(self):
        try:
            simulation = LiveSimulation.build()
        except Exception as exc:
            traceback.print_exc()
            if self.mode == "auto":
                # The model could not load here, but another process may still be
                # publishing to Redis (docker-compose runs simulate_live.py in its
                # own container). Try that before giving up on the feed.
                self._loop.call_soon_threadsafe(self._start_redis_bridge)
                return
            self._set_status("unavailable", f"simulation could not start: {exc}")
            return

        sink = _EventSink(LIVE_CHANNEL)
        self.status = "running"
        self.detail = f"in-process simulation, {simulation.policy.backend} backend"
        print(f"[live] {self.detail}, {EVENTS_PER_SECOND:g} events/s", flush=True)

        min_interval = 1.0 / EVENTS_PER_SECOND if EVENTS_PER_SECOND > 0 else 0.0
        last_broadcast = 0.0

        for event in simulation.iter_events():
            if self._stop.is_set():
                break

            sink.forward(event)

            now = time.monotonic()
            if now - last_broadcast >= BROADCAST_INTERVAL:
                last_broadcast = now
                payload = json.dumps(simulation.payload())
                self._publish(payload)
                sink.mirror(payload)

            if min_interval:
                time.sleep(min_interval)

        sink.close()

    # ── redis bridge (someone else is simulating) ─────────────────────────────
    def _start_redis_bridge(self):
        self._redis_task = asyncio.ensure_future(self._bridge_redis())

    async def _bridge_redis(self):
        host = os.environ.get("REDIS_HOST")
        if not host:
            self._set_status(
                "unavailable",
                "no in-process simulation and REDIS_HOST is unset — nothing to relay",
            )
            return

        try:
            import redis.asyncio as aioredis
        except ImportError:
            self._set_status("unavailable", "redis package not installed")
            return

        client = aioredis.Redis(host=host, port=6379, db=0)
        try:
            pubsub = client.pubsub()
            await pubsub.subscribe(LIVE_CHANNEL)
        except Exception as exc:
            self._set_status("unavailable", f"redis unreachable at {host}:6379 ({exc})")
            await client.aclose()
            return

        self.status = "running"
        self.detail = f"relaying {LIVE_CHANNEL} from redis://{host}:6379"
        print(f"[live] {self.detail}", flush=True)
        try:
            async for message in pubsub.listen():
                if message["type"] == "message":
                    self.bus.publish(message["data"].decode())
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._set_status("unavailable", f"redis relay stopped: {exc}")
        finally:
            try:
                await pubsub.unsubscribe(LIVE_CHANNEL)
            except Exception:
                pass
            await client.aclose()


class _EventSink:
    """Optional Redis/Kafka mirroring — present when configured, silent when not.

    Both are strictly opt-in: the deployed single-service topology has neither,
    and a missing broker must never slow down or kill the simulation. Note the
    original script connected to Kafka unconditionally, which cost a connection
    timeout on every start outside docker-compose.
    """

    def __init__(self, channel):
        self.channel = channel
        self.redis = self._connect_redis()
        self.kafka = self._connect_kafka()
        self.topic = os.environ.get("FF_KAFKA_TOPIC", "agent_actions")

    @staticmethod
    def _connect_redis():
        host = os.environ.get("REDIS_HOST")
        if not host:
            return None
        try:
            import redis

            client = redis.Redis(host=host, port=6379, db=0, socket_timeout=1)
            client.ping()
            print(f"[live] mirroring payloads to redis://{host}:6379", flush=True)
            return client
        except Exception as exc:
            print(f"[live] redis mirror disabled ({exc})", flush=True)
            return None

    @staticmethod
    def _connect_kafka():
        if not (os.environ.get("KAFKA_HOST") or os.environ.get("KAFKA_BOOTSTRAP_SERVERS")):
            return None
        try:
            from comms.kafka_client import get_producer

            producer = get_producer()
            print("[live] mirroring events to kafka", flush=True)
            return producer
        except Exception as exc:
            print(f"[live] kafka mirror disabled ({exc})", flush=True)
            return None

    def forward(self, event):
        """Per-event Kafka publish — the analytics stream, not the dashboard path."""
        if self.kafka is None:
            return
        try:
            # Same wire format simulate_live.py has always published, so existing
            # consumers of `agent_actions` keep working.
            self.kafka.send(self.topic, value={
                "agent_id": event["agent_id"],
                "action": event["action"],
                "product_id": event["product_id"],
                "timestamp": datetime.datetime.utcnow().isoformat(),
            })
        except Exception:
            self.kafka = None   # broker died; stop trying rather than log per event

    def mirror(self, payload):
        """Per-broadcast Redis publish, so other subscribers still see the feed."""
        if self.redis is None:
            return
        try:
            self.redis.publish(self.channel, payload)
        except Exception:
            self.redis = None

    def close(self):
        for client in (self.redis, self.kafka):
            try:
                if client is not None:
                    client.close()
            except Exception:
                pass
