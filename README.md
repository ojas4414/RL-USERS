<div align="center">

# RL-USERS

**Multi-agent reinforcement learning simulation of Amazon shopper behavior**

50 virtual shoppers, trained on 701K real Amazon reviews, browse a live product catalog, influence each other through a social graph, and are validated against real e-commerce benchmarks — end to end, in real time.

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-BC%20%2B%20PPO-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![C++](https://img.shields.io/badge/C%2B%2B17-pybind11-00599C?logo=cplusplus&logoColor=white)](https://isocpp.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-microservices-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Redis](https://img.shields.io/badge/Redis-pub%2Fsub-DC382D?logo=redis&logoColor=white)](https://redis.io/)
[![Kafka](https://img.shields.io/badge/Apache%20Kafka-event%20stream-231F20?logo=apachekafka&logoColor=white)](https://kafka.apache.org/)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![React](https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=black)](https://react.dev/)
[![D3.js](https://img.shields.io/badge/D3.js-force%20graph-F9A03C?logo=d3dotjs&logoColor=white)](https://d3js.org/)

</div>

---

## What this is

RL-USERS simulates a shopping population, not a shopping cart. Each of the 50 agents has a persona (power buyer, average buyer, browser), a wallet, and a reinforcement-learned policy for what to look at next — trained first by behavioral cloning on real purchase sequences, then fine-tuned with PPO. Agents move through a funnel (`browse → product_detail → cart → checkout`), influence their neighbors on a social graph when they buy something, and their *aggregate behavior* — not their individual product choices — is validated against published e-commerce benchmarks.

The simulation streams live: every agent action is published to Redis and Kafka as it happens, a FastAPI gateway forwards it over a WebSocket, and the dashboard updates round by round instead of reading a static report file.

## Architecture

```
                     ┌─────────────────────────┐
                     │   Amazon Reviews (701K)  │
                     │   All_Beauty category    │
                     └────────────┬─────────────┘
                                  │
                     ┌────────────▼─────────────┐
                     │  Behavioral Cloning → PPO │   agents/bc_trainer.py
                     │  policy_bc.pt → policy_rl.pt│  agents/rl_trainer.py
                     └────────────┬─────────────┘
                                  │
   ┌──────────────────────────────▼──────────────────────────────┐
   │                    simulate_live.py                          │
   │  50 agents · heap scheduler · BFS social propagation ·       │
   │  topological funnel gate · persona-driven budgets/quit rates │
   │  (C++ / pybind11 accelerated core: agents/cpp/)               │
   └───────┬──────────────────────────────────────────┬──────────┘
           │ publish per event                          │ publish per event
   ┌───────▼────────┐                           ┌───────▼────────┐
   │      Redis      │                           │  Apache Kafka   │
   │    pub/sub       │                           │  agent_actions  │
   └───────┬────────┘                           └────────────────┘
           │ subscribe
   ┌───────▼────────────────────┐        ┌─────────────────────────┐
   │  FastAPI gateway /ws/live   │───────▶│   Live dashboard (D3)    │
   │  + microservice cluster     │  WS    │   ● LIVE / ○ LAST RUN    │
   └─────────────────────────────┘        └─────────────────────────┘
```

- **Training** — `agents/bc_trainer.py` imitation-learns from real user sequences; `agents/rl_trainer.py` fine-tunes with PPO, boosting product logits for agents that just received a social signal.
- **Simulation core** — a min-heap event scheduler, an Erdős–Rényi social graph with BFS influence propagation (`1 / 2^depth` decay), and a topologically-sorted funnel gate, all implemented in Python with C++/pybind11 accelerated equivalents in `agents/cpp/`.
- **Live infrastructure** — every agent action publishes to Redis (dashboard feed, throttled to ~2 updates/sec for readability) and Kafka (`agent_actions`, full-fidelity event log, retention-capped so a long-running dev session can't fill the disk). Kafka is currently write-only: no consumer exists yet, but the stream is there for future subscribers — e.g. an inventory tracker, an audit log, or another agent reacting to the first 50.
- **Backend** — a FastAPI gateway fronting `product_service`, `cart_service`, `order_service`, and `session_service`, each with its own Redis-backed state.
- **Validation** — session length, conversion rate, cart abandonment, and social influence concentration, each checked against real industry benchmarks with a PASS/FAIL verdict.

## Features

- 🧠 **Real RL policy inference**, not scripted behavior — every product choice is a forward pass through a PPO-trained model, with social signals live-boosting the logits of trending products.
- 🕸️ **Social contagion you can watch happen** — click any node in the dashboard's social graph to see a BFS influence pulse propagate outward in real time, with edges highlighting along the actual propagation path.
- 📡 **Genuinely live dashboard** — WebSocket-driven, with automatic fallback to the last saved report when no simulation is running.
- ✅ **Behavioral validation against real data**, not synthetic targets — session length, conversion rate, abandonment, and trend concentration are all benchmarked against published e-commerce numbers.
- 🐳 **Fully containerized backend** — Redis, Kafka, and four FastAPI microservices behind a gateway, orchestrated with a single `docker-compose up`.

## Getting started

### Prerequisites

- Python 3.11+
- Docker Desktop (for Redis, Kafka, and the microservice cluster)
- Node.js 18+ (for the dashboard, optional — a static HTML dashboard is also included)

### 1. Install Python dependencies

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
pip install torch pandas fastapi uvicorn kafka-python   # not pinned in requirements.txt yet
```

### 2. Get the dataset

The training data (701K Amazon "All Beauty" reviews, ~310MB) isn't checked into this repo — grab it from the [Amazon Reviews 2023 dataset](https://amazon-reviews-2023.github.io/) and place it at:

```
data/raw/All_Beauty.jsonl
```

### 3. Bring up the backend

```bash
docker-compose up -d --build
```

This starts Redis, Zookeeper + Kafka, and the four FastAPI microservices behind a gateway on `localhost:8000`.

### 4. Train (optional — a pretrained policy is already checked in)

```bash
python train.py
```

### 5. Run the live simulation

```bash
python simulate_live.py
```

### 6. Open the dashboard

```bash
python -m http.server 8080
```

Visit `http://localhost:8080/visuals/dashboard.html`. It shows `○ LAST RUN` with the last saved report until the simulation is running, then flips to `● LIVE` and updates in real time.

## Validation results

| Metric | Industry benchmark | Simulated | Verdict |
|---|---|---|---|
| Session length | ~8 items/session | 9.5 ± 10.8 | ✅ PASS |
| Conversion rate | 2–20% | ~16% | ✅ PASS |
| Cart abandonment | 60–95% | ~84% | ✅ PASS |
| Social influence (top-5 concentration) | 20–40% | ~3–15% | ⚠️ Below target — catalog diversity (2,985 products) spreads purchases more than a single trend event would |

## Repo layout

```
agents/          persona/agent model, BFS social graph, funnel gate, scheduler, BC + PPO trainers, C++ core
backend/         FastAPI gateway + product/cart/order/session microservices
comms/           Redis and Kafka client helpers
data/            data loading, persona clustering, social graph builder
validation/      behavioral metrics + benchmark verdicts
visuals/         live dashboard (static HTML) and Vite/React dashboard
simulate.py      one-shot batch simulation → validation_report.json
simulate_live.py continuous simulation → Redis/Kafka live stream
train.py         BC + PPO training entry point
docker-compose.yml   Redis, Kafka, and the microservice cluster
```
