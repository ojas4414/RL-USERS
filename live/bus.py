"""
In-process fan-out for live simulation payloads.

The original pipeline was simulate_live.py -> Redis pub/sub -> gateway
websocket. That works under docker-compose, where a Redis container exists, but
on a single Render web service there is no Redis to publish into and the
websocket handler died on connect. This bus is the fallback the gateway falls
back to: same payloads, same websocket, no broker.

Redis is still used when it is actually reachable (see live/runner.py) — the
point is that its absence degrades to a working feed instead of an empty panel.
"""
import asyncio

# A slow websocket client should never grow the server's memory or stall the
# simulation thread. Each subscriber gets a small queue; when it fills, the
# oldest frame is dropped. Payloads are whole-state snapshots rather than
# deltas, so a dropped frame costs a client nothing but freshness.
QUEUE_DEPTH = 4


class LiveBus:
    """Broadcasts the latest payload to every subscribed websocket."""

    def __init__(self):
        self._subscribers = set()
        self._latest = None

    @property
    def latest(self):
        """Most recent payload, or None if the simulation has not produced one."""
        return self._latest

    def subscribe(self):
        queue = asyncio.Queue(maxsize=QUEUE_DEPTH)
        self._subscribers.add(queue)
        # Hand the newcomer the current state immediately so the dashboard can
        # leave its "connecting" state without waiting for the next broadcast.
        if self._latest is not None:
            queue.put_nowait(self._latest)
        return queue

    def unsubscribe(self, queue):
        self._subscribers.discard(queue)

    def publish(self, payload):
        """Fan a payload out. Must be called on the event loop thread."""
        self._latest = payload
        for queue in self._subscribers:
            if queue.full():
                try:
                    queue.get_nowait()      # drop the stalest frame
                except asyncio.QueueEmpty:
                    pass
            try:
                queue.put_nowait(payload)
            except asyncio.QueueFull:        # raced another consumer; skip it
                pass

    @property
    def subscriber_count(self):
        return len(self._subscribers)
