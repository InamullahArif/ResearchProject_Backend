"""
Optional Kafka bridge demo (proposal: streaming integration).

Requires: pip install kafka-python
Start a broker (e.g. Docker) and topic `fraud_tx`, or edit BOOTSTRAP / TOPIC below.

This script only prints messages — wire your own handler to POST JSON to the FastAPI /score endpoint.
"""
from __future__ import annotations

import json
import sys

BOOTSTRAP = "localhost:9092"
TOPIC = "fraud_tx"


def main():
    try:
        from kafka import KafkaConsumer
    except ImportError:
        print("Install kafka-python: pip install kafka-python", file=sys.stderr)
        sys.exit(1)

    consumer = KafkaConsumer(
        TOPIC,
        bootstrap_servers=[BOOTSTRAP],
        value_deserializer=lambda b: json.loads(b.decode("utf-8")),
        auto_offset_reset="earliest",
        enable_auto_commit=True,
        group_id="fraud-demo-consumer",
    )
    print(f"Listening on {TOPIC} @ {BOOTSTRAP} (Ctrl+C to stop)")
    for msg in consumer:
        print(msg.value)


if __name__ == "__main__":
    main()
