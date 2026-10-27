# 13. Deployment Cookbook for Multiple Projects

This cookbook applies to:

- `advanced_rag_llmops`
- `ecommerce_chatbot_fastapi`
- Future voice, multimodal, and agent projects.

## Local Development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest
uvicorn app.main:app --reload --port 8000
```

Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pytest
uvicorn app.main:app --reload --port 8000
```

## Docker

```bash
docker build -t my-genai-app .
docker run -p 8000:8000 my-genai-app
```

## Docker Compose

Use Compose when you need multiple services:

```yaml
services:
  api:
    build: .
    ports:
      - "8000:8000"
  vector-db:
    image: qdrant/qdrant
    ports:
      - "6333:6333"
```

## AWS ECS/Fargate

1. Create ECR repository.
2. Build Docker image.
3. Push image to ECR.
4. Create ECS cluster.
5. Create task definition.
6. Configure environment variables.
7. Configure secrets.
8. Create service on Fargate.
9. Attach load balancer.
10. Configure health check.
11. Enable CloudWatch logs.
12. Add autoscaling.
13. Add CI/CD deployment.

## AWS EKS

1. Create EKS cluster.
2. Push image to ECR.
3. Create Deployment YAML.
4. Create Service YAML.
5. Create Ingress YAML.
6. Add ConfigMap.
7. Add Secret.
8. Apply manifests.
9. Add HPA autoscaling.
10. Add monitoring.

## Azure App Service

1. Push image to Azure Container Registry.
2. Create App Service for Containers.
3. Select container image.
4. Configure app settings.
5. Connect Key Vault secrets.
6. Enable logs.
7. Configure health check.

## Azure AKS

1. Push image to ACR.
2. Create AKS cluster.
3. Apply Kubernetes manifests.
4. Add ingress controller.
5. Add Key Vault integration.
6. Add Azure Monitor.

## GCP Cloud Run

1. Build image.
2. Push image to Artifact Registry.
3. Deploy to Cloud Run.
4. Add environment variables.
5. Add Secret Manager values.
6. Configure concurrency.
7. Configure min/max instances.
8. Monitor logs.

## Jenkins

Pipeline stages:

1. Checkout.
2. Install dependencies.
3. Run tests.
4. Build Docker image.
5. Push Docker image.
6. Deploy.
7. Smoke test `/health`.

## GitHub Actions

Basic CI:

```yaml
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install -r requirements.txt
      - run: pytest
```

## Production Environment Variables

Common:

- `APP_ENV`
- `LOG_LEVEL`
- `OPENAI_API_KEY`
- `OPENAI_MODEL`
- `VECTOR_DB_URL`
- `DATABASE_URL`
- `REDIS_URL`
- `AUTH_ISSUER`
- `SENTRY_DSN`

## Pre-Deployment Checklist

- Tests pass.
- Evals pass.
- Secrets not in code.
- Auth enabled.
- Tenant filtering enabled.
- Prompt injection tested.
- Rate limits configured.
- Logs configured.
- Health endpoint works.
- Rollback plan exists.

