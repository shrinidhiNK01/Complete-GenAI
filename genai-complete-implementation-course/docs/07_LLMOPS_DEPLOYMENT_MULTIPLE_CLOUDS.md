# 07. LLMOps, Testing, Guardrails, and Multiple Deployment Steps

## LLMOps Goal

LLMOps makes LLM applications reliable, measurable, secure, and deployable.

Important parts:

- Config.
- Logging.
- Tests.
- Evaluation.
- Guardrails.
- API.
- Docker.
- CI/CD.
- Monitoring.
- Rollback.

## Project Structure

The project uses:

```text
app/
  core/
  ingestion/
  rag/
  agents/
  api/
  evaluation/
tests/
data/
Dockerfile
docker-compose.yml
.github/workflows/ci.yml
```

This structure separates concerns. Each folder has one responsibility.

## Unit Tests

Unit tests check small deterministic functions.

Examples:

- Chunk overlap works.
- Calculator is safe.
- Retriever finds expected chunk.

Run:

```bash
pytest
```

## Integration Tests

Integration tests should check:

- Ingest documents.
- Build index.
- Ask question.
- Receive answer with source.

Add this next:

```python
def test_full_rag_pipeline():
    pipeline = RAGPipeline(load_settings())
    result = pipeline.ask("What is the refund window?")
    assert "30 days" in result.answer.lower()
```

## Evaluation

Unit tests check code correctness. Evals check AI quality.

The project includes:

```bash
python cli.py eval
```

Production evals should include:

- 50 to 500 real questions.
- Expected facts.
- Source documents.
- Failure cases.
- Prompt injection examples.
- Permission boundary tests.

## Guardrails

Guardrails happen at multiple layers.

Input:

- Validate file type.
- Validate question length.
- Detect prompt injection.

Retrieval:

- Filter by user permission.
- Filter by tenant.
- Filter by freshness.

Generation:

- Require citations.
- Allow refusal.
- Validate JSON schema.

Tools:

- Allowlist tools.
- Validate arguments.
- Human approval for irreversible actions.

## Docker Deployment

Build:

```bash
docker build -t advanced-rag-llmops .
```

Run:

```bash
docker run -p 8000:8000 advanced-rag-llmops
```

Open:

```text
http://localhost:8000/docs
```

## Docker Compose

Run:

```bash
docker compose up --build
```

Use Compose when you add:

- Postgres.
- Redis.
- Qdrant.
- Elasticsearch.
- Monitoring services.

## GitHub Actions

The project includes:

```text
.github/workflows/ci.yml
```

It:

1. Checks out code.
2. Sets up Python.
3. Installs dependencies.
4. Runs syntax check.
5. Runs tests.

## AWS ECS/Fargate Deployment

Steps:

1. Build Docker image.
2. Create ECR repository.
3. Push image to ECR.
4. Create ECS cluster.
5. Create task definition.
6. Add CPU/memory.
7. Add environment variables.
8. Add secrets from AWS Secrets Manager.
9. Create Fargate service.
10. Attach Application Load Balancer.
11. Configure `/health` health check.
12. Enable CloudWatch logs.
13. Add autoscaling.
14. Add CI/CD deploy step.

## AWS EKS Deployment

Steps:

1. Build and push image to ECR.
2. Create EKS cluster.
3. Create Kubernetes Deployment.
4. Create Kubernetes Service.
5. Create Ingress.
6. Add secrets.
7. Add config map.
8. Apply manifests.
9. Monitor pods.
10. Add horizontal pod autoscaler.

## Azure Deployment

App Service for Containers:

1. Push image to Azure Container Registry.
2. Create App Service.
3. Select container image.
4. Add environment variables.
5. Add Key Vault secrets.
6. Enable logs.
7. Configure health check.

AKS:

1. Push image to ACR.
2. Create AKS.
3. Deploy Kubernetes manifests.
4. Add ingress.
5. Add monitoring.

## GCP Deployment

Cloud Run:

1. Build image.
2. Push to Artifact Registry.
3. Deploy to Cloud Run.
4. Add environment variables.
5. Add Secret Manager secrets.
6. Configure scaling.
7. Monitor with Cloud Logging.

## Jenkins Pipeline

Stages:

1. Checkout.
2. Install dependencies.
3. Run tests.
4. Build Docker image.
5. Push image.
6. Deploy.
7. Smoke test `/health`.

## Production Checklist

- Auth.
- Tenant isolation.
- Permissions before retrieval.
- Secret manager.
- Prompt injection tests.
- Eval dataset.
- Logging.
- Metrics.
- Rate limits.
- Cost alerts.
- Rollback.
- Data deletion process.
- Human approval for risky tools.

