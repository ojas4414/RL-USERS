import json
import os

# kafka-python is imported inside get_producer rather than at module scope.
# Kafka is optional in every deployment that isn't docker-compose — the single
# Render web service has no broker — and a top-level import made merely
# importing this module fail when the package was absent, which took the whole
# gateway down with it.


def get_producer():
    from kafka import KafkaProducer

    bootstrap = os.getenv("KAFKA_BOOTSTRAP_SERVERS")
    if not bootstrap:
        host = os.getenv("KAFKA_HOST", "localhost")
        bootstrap = f"{host}:9092"
    return KafkaProducer(
        bootstrap_servers=bootstrap,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )


def publish_event(producer, topic: str, payload: dict) -> None:
    producer.send(topic, value=payload)
    producer.flush()
