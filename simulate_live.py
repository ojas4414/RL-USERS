"""
simulate_live.py — continuous agent simulation, as a standalone process.

Agents run in an infinite loop and every action is written to
visuals/live_events.json and published to Redis, so a dashboard served from
anywhere can pick up the feed.

The simulation itself now lives in live/engine.py. Keeping it there is what lets
the FastAPI gateway run the same loop in-process (live/runner.py) instead of
requiring someone to start this script by hand — which is why the deployed
dashboard used to sit on "No live simulation running" forever. This file is the
standalone/docker-compose entrypoint; behaviour is unchanged.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, ".")

from live.engine import LiveSimulation

LIVE_FILE = os.environ.get("FF_LIVE_FILE", "visuals/live_events.json")
LIVE_CHANNEL = os.environ.get("FF_LIVE_CHANNEL", "rl_users_live")
KAFKA_TOPIC = os.environ.get("FF_KAFKA_TOPIC", "agent_actions")
BROADCAST_INTERVAL = 0.5   # seconds between dashboard updates — keeps the feed readable


def connect_redis():
    """Optional. A missing or dead Redis must never stop the simulation."""
    host = os.environ.get("REDIS_HOST", "127.0.0.1")
    try:
        import redis

        client = redis.Redis(host=host, port=6379, db=0, socket_timeout=1)
        client.ping()
        print(f"  Publishing to redis://{host}:6379 ({LIVE_CHANNEL})")
        return client
    except Exception as exc:
        print(f"  Redis unavailable at {host}:6379 ({exc}) — file output only")
        return None


def connect_kafka():
    """Opt-in: only attempt a broker when one is actually configured.

    This used to connect unconditionally against localhost:29092, which cost a
    connection timeout on every start outside docker-compose.
    """
    if not (os.environ.get("KAFKA_HOST") or os.environ.get("KAFKA_BOOTSTRAP_SERVERS")):
        return None
    try:
        from comms.kafka_client import get_producer

        producer = get_producer()
        print(f"  Publishing events to kafka topic {KAFKA_TOPIC}")
        return producer
    except Exception as exc:
        print(f"  Kafka unavailable ({exc}) — continuing without it")
        return None


def write_live(payload):
    tmp_path = LIVE_FILE + ".tmp"
    with open(tmp_path, "w") as handle:
        json.dump(payload, handle)
    os.replace(tmp_path, LIVE_FILE)   # atomic swap — the browser never reads a half-written file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--events-per-second",
        type=float,
        default=0.0,
        help="throttle the simulation (0 = as fast as possible, the default)",
    )
    args = parser.parse_args()

    print("Loading policy and product vocabulary...")
    simulation = LiveSimulation.build()
    print(f"  {simulation.policy.backend} backend, "
          f"{len(simulation.item_to_idx):,} products, {simulation.n_agents} agents")

    os.makedirs(os.path.dirname(LIVE_FILE) or ".", exist_ok=True)
    redis_client = connect_redis()
    kafka_producer = connect_kafka()

    print(f"\nLive simulation started — writing {LIVE_FILE}")
    print("Press Ctrl+C to stop.\n")

    min_interval = 1.0 / args.events_per_second if args.events_per_second > 0 else 0.0
    last_broadcast = 0.0
    last_round = 0

    try:
        for event in simulation.iter_events():
            if kafka_producer is not None:
                try:
                    kafka_producer.send(KAFKA_TOPIC, value={
                        "agent_id": event["agent_id"],
                        "action": event["action"],
                        "product_id": event["product_id"],
                        "timestamp": event["time"],
                    })
                except Exception:
                    kafka_producer = None   # broker died; stop retrying per event

            now = time.monotonic()
            if now - last_broadcast >= BROADCAST_INTERVAL:
                last_broadcast = now
                payload = simulation.payload()
                write_live(payload)
                if redis_client is not None:
                    try:
                        redis_client.publish(LIVE_CHANNEL, json.dumps(payload))
                    except Exception:
                        redis_client = None

            if simulation.rounds != last_round:
                last_round = simulation.rounds
                print(f"Round {last_round} — {simulation.stats['total_events']} events, "
                      f"{simulation.stats['total_purchases']} purchases so far")

            if min_interval:
                time.sleep(min_interval)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
