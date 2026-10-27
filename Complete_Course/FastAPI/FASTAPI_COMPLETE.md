# FastAPI: The Complete Masterclass (Beginner to Advanced)

Welcome to the ultimate guide to API development with FastAPI. This document explains every major API concept, database integration pattern, security mechanism, testing strategy, and deployment architecture in simple English, followed by step-by-step implementation details.

---

## Table of Contents
1. **Introduction to FastAPI**
2. **HTTP API Basics & Request Validation (Pydantic)**
3. **Database Integration: SQLAlchemy ORM**
4. **Database Migrations: Alembic**
5. **Security & Authentication: JWT & OAuth2**
6. **Project Architecture & APIRouter**
7. **Testing APIs with Pytest**
8. **Production Deployment: Docker & Nginx**

---

## 1. Introduction to FastAPI

FastAPI is a modern, high-performance web framework for building APIs with Python. It is based on **Starlette** (for web routing) and **Pydantic** (for data validation).

### Why choose FastAPI?
*   **Speed:** As fast as NodeJS and Go, thanks to Python's async capabilities (`async`/`await`).
*   **Automatic Documentation:** Instantly generates interactive API docs (Swagger UI at `/docs` and ReDoc at `/redoc`) based on your code.
*   **Data Validation:** Automatically validates incoming request data using Python type hints. Invalid requests get blocked with a descriptive `422 Unprocessable Entity` error.
*   **Editor Support:** Auto-completion works everywhere, reducing development bugs.

---

## 2. HTTP API Basics & Request Validation (Pydantic)

### HTTP Requests
FastAPI maps HTTP methods to path operations:
*   `@app.get("/items")`: Read items.
*   `@app.post("/items")`: Create items.
*   `@app.put("/items/{id}")`: Replace item.
*   `@app.delete("/items/{id}")`: Delete item.

### Parameters
1.  **Path Parameters:** Declared in the URL path.
    ```python
    @app.get("/items/{item_id}")
    def read_item(item_id: int):
        return {"item_id": item_id}
    ```
2.  **Query Parameters:** Not declared in the path. Appended to URL (e.g., `/items?skip=0&limit=10`).
    ```python
    @app.get("/items")
    def read_items(skip: int = 0, limit: int = 10):
        return {"skip": skip, "limit": limit}
    ```

### Request Body & Pydantic
A Pydantic model defines the structure of data expected in the request body.

```python
from pydantic import BaseModel, Field

class ItemCreate(BaseModel):
    name: str = Field(..., min_length=3, max_length=50)
    price: float = Field(..., gt=0.0)
    is_offer: bool | None = None
```

FastAPI automatically reads the JSON request body, validates it against `ItemCreate`, and injects it into your function:

```python
@app.post("/items")
def create_item(item: ItemCreate):
    return {"message": f"Created {item.name} costing ${item.price}"}
```

---

## 3. Database Integration: SQLAlchemy ORM

An **Object Relational Mapper (ORM)** lets you interact with SQL databases using Python objects instead of writing raw SQL queries.

```text
Python Object (user.name = "Alice") ===> [ SQLAlchemy ] ===> SQL Command: INSERT INTO users ...
```

### The Setup File: `database.py`
To configure a database connection:
1.  **Create Engine:** Handles connection pools.
2.  **Create SessionLocal:** Active database session factory.
3.  **Define Base:** Class that database models inherit from to register tables.

```python
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

SQLALCHEMY_DATABASE_URL = "sqlite:///./sql_app.db"

engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()
```

### Dependency Injection: `get_db`
We use dependency injection to yield a database session for a request and close it when the request is complete:

```python
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
```

### Defining a Model
```python
from sqlalchemy import Column, Integer, String, Boolean
from .database import Base

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    is_active = Column(Boolean, default=True)
```

---

## 4. Database Migrations: Alembic

When you update your SQLAlchemy models (e.g. adding a new column `phone_number` to `User`), SQLite/Postgres tables don't update automatically. **Alembic** manages these database schema updates step-by-step.

### The Alembic Workflow
1.  **Initialize:** Create configuration folder.
    ```bash
    alembic init alembic
    ```
2.  **Configure:** In `alembic/env.py`, point Alembic to your model metadata:
    ```python
    from app.database import Base
    target_metadata = Base.metadata
    ```
    And edit `alembic.ini` (or `env.py` dynamically) to set the database URL.
3.  **Generate Migration:** Alembic compares your models against the current database and drafts a migration file.
    ```bash
    alembic revision --autogenerate -m "add phone number to user"
    ```
4.  **Apply Migration:** Run the updates on the actual database.
    ```bash
    alembic upgrade head
    ```

---

## 5. Security & Authentication: JWT & OAuth2

API security standardizes authentication using **JWT (JSON Web Tokens)**.

```text
1. Client logs in with Password ===> [ FastAPI ] ===> Signs JWT Token using Secret Key
2. Client sends Token in Header (Authorization: Bearer <token>) ===> [ FastAPI ] ===> Verifies Signature
```

### What is in a JWT?
A token contains three parts separated by dots: `header.payload.signature`.
*   **Header:** Specifies token type and signing algorithm (e.g. HS256).
*   **Payload:** Data being sent (e.g. `{"user_id": 42, "exp": 1718000000}`).
*   **Signature:** A hash computed using the header, payload, and a secret key known *only* to the backend. This prevents clients from tampering with the payload.

### Setup Steps
1.  **Hash Passwords:** Store passwords hashed using `bcrypt` (never store raw text).
    ```python
    from passlib.context import CryptContext
    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
    
    def hash_password(password: str):
        return pwd_context.hash(password)
    ```
2.  **Verify Password:** Compare login password hash against database record.
    ```python
    def verify_password(plain_password, hashed_password):
        return pwd_context.verify(plain_password, hashed_password)
    ```
3.  **Create Access Token:** Use the `python-jose` library to sign a JWT.
    ```python
    from jose import jwt
    from datetime import datetime, timedelta
    
    def create_access_token(data: dict, expires_delta: timedelta):
        to_encode = data.copy()
        expire = datetime.utcnow() + expires_delta
        to_encode.update({"exp": expire})
        return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    ```

---

## 6. Project Architecture & APIRouter

For production, you should structure your project into clean directories:

```text
app/
├── main.py             # App initialization
├── config.py           # Environment variables
├── database.py         # DB connection & session
├── models.py           # SQLAlchemy database tables
├── schemas.py          # Pydantic schemas (validations)
├── routers/            # Splitting API paths
│   ├── auth.py         # Login routes
│   └── users.py        # User CRUD routes
```

### Using APIRouter
In `app/routers/users.py`:
```python
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from .. import schemas, database

router = APIRouter(prefix="/users", tags=["Users"])

@router.post("/", response_model=schemas.UserResponse)
def create_user(user: schemas.UserCreate, db: Session = Depends(database.get_db)):
    # logic here
    return new_user
```

In `app/main.py`, simply register the router:
```python
from fastapi import FastAPI
from .routers import users, auth

app = FastAPI()
app.include_router(users.router)
app.include_router(auth.router)
```

---

## 7. Testing APIs with Pytest

Testing ensures changes don't break existing routes. We use `pytest` and the FastAPI `TestClient` (built on top of `httpx`).

### Database Isolation Fixture
We don't want tests writing dummy data to our production database. We configure a test database fixture that runs and cleans up after each test:

```python
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.main import app
from app.database import Base, get_db

# Create separate test database engine
TEST_DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(TEST_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

@pytest.fixture()
def session():
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()

@pytest.fixture()
def client(session):
    def override_get_db():
        try:
            yield session
        finally:
            pass
    app.dependency_overrides[get_db] = override_get_db
    yield TestClient(app)
    app.dependency_overrides.clear()
```

### Writing a Test
```python
def test_create_user(client):
    response = client.post("/users/", json={"email": "test@example.com", "password": "password123"})
    assert response.status_code == 201
    assert response.json()["email"] == "test@example.com"
```

---

## 8. Production Deployment: Docker & Nginx

### Dockerfile
Use a multi-stage Docker build to build clean, lightweight production containers.

```dockerfile
FROM python:3.11-slim as builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY . .
ENV PATH=/root/.local/bin:$PATH
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

### Nginx Reverse Proxy
Nginx sits in front of your FastAPI app to handle incoming SSL (HTTPS) decryption, static files caching, and route protection.

```nginx
server {
    listen 80;
    server_name api.example.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```
 Gunicorn / Uvicorn manages the Python worker processes, and Nginx acts as the gatekeeper.
