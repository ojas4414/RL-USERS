import json
import os

from kafka import KafkaProducer


def get_producer() -> KafkaProducer:
    bootstrap = os.getenv("KAFKA_BOOTSTRAP_SERVERS")
    if not bootstrap:
        host = os.getenv("KAFKA_HOST", "localhost")
        bootstrap = f"{host}:9092"
    return KafkaProducer(
        bootstrap_servers=bootstrap,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )


def publish_event(producer: KafkaProducer, topic: str, payload: dict) -> None:
    producer.send(topic, value=payload)
    producer.flush()
