# E-Commerce Chatbot FastAPI Project

This project implements a practical e-commerce chatbot with routing, policy RAG, product search, order lookup, authorization checks, guardrails, API, tests, Docker, and deployment notes.

## What It Teaches

- Product search.
- Policy question answering.
- Order tracking.
- User authorization before private order lookup.
- Human review flag for risky actions.
- FastAPI deployment.
- Docker containerization.

## Run Locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python cli.py "recommend running shoe"
python cli.py "what is the refund policy"
python cli.py "track order ORD1001" --user-id user_1
pytest
```

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python cli.py "recommend running shoe"
python cli.py "what is the refund policy"
python cli.py "track order ORD1001" --user-id user_1
pytest
```

## Run API

```bash
uvicorn app.main:app --reload --port 8001
```

Open:

```text
http://localhost:8001/docs
```

## Docker

```bash
docker build -t ecommerce-chatbot .
docker run -p 8001:8000 ecommerce-chatbot
```

## Production Upgrades

- Replace file data with Postgres/MongoDB.
- Replace policy keyword search with vector database RAG.
- Add login and JWT authentication.
- Add order service API integration.
- Add human approval workflow for refunds/cancellations.
- Add monitoring and audit logs.

