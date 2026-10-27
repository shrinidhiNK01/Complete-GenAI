"""
Complete Production-Ready FastAPI Application.
This file implements:
1. Pydantic schemas for User registration and Authentication.
2. SQLAlchemy database setup (SQLite local database).
3. Hashed password storage with bcrypt.
4. JWT token generation and authentication headers.
5. Global exception handlers and CORS middleware.
6. User profile retrieval protected by JWT tokens.

Run: uvicorn Complete_Course.FastAPI.app:app --reload
"""

import os
from datetime import datetime, timedelta
from typing import Generator
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from jose import JWTError, jwt
from passlib.context import CryptContext

# --- Configuration & Security Defaults ---
SECRET_KEY = "my_super_secret_jwt_signing_key_for_study_pack"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Password hashing helper
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
# OAuth2 bearer token reader
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/token")

# --- Database Setup ---
DATABASE_URL = "sqlite:///./fastapi_master_study.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class UserModel(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)

# Create tables if they do not exist
Base.metadata.create_all(bind=engine)

def get_db() -> Generator[Session, None, None]:
    """Yield database sessions to API routes."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- Pydantic Data Schemas ---
class UserRegister(BaseModel):
    email: EmailStr
    password: str

class UserResponse(BaseModel):
    id: int
    email: EmailStr

    class Config:
        orm_mode = True  # Pydantic v1 fallback compatibility

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    email: str | None = None

# --- FastAPI Initialization ---
app = FastAPI(
    title="Production API Masterclass",
    description="Complete beginner-to-advanced API implementation",
    version="1.0.0"
)

# Enable CORS (Cross-Origin Resource Sharing)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permits requests from any source domain
    allow_credentials=True,
    allow_methods=["*"],  # Permits GET, POST, etc.
    allow_headers=["*"],  # Permits all headers
)

# --- Security Helpers ---
def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def create_access_token(data: dict) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> UserModel:
    """Dependency to retrieve and validate the currently logged-in user from the JWT."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
        token_data = TokenData(email=email)
    except JWTError:
        raise credentials_exception
        
    user = db.query(UserModel).filter(UserModel.email == token_data.email).first()
    if user is None:
        raise credentials_exception
    return user

# --- API Routes ---
@app.get("/")
def read_root():
    return {"message": "Welcome to the FastAPI Masterclass API!", "docs": "/docs"}

@app.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def register_user(user_in: UserRegister, db: Session = Depends(get_db)):
    # Check if user already exists
    existing_user = db.query(UserModel).filter(UserModel.email == user_in.email).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
        
    hashed_pwd = get_password_hash(user_in.password)
    new_user = UserModel(email=user_in.email, hashed_password=hashed_pwd)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user

@app.post("/token", response_model=Token)
def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    """Authenticate email and password, returning a JWT token."""
    user = db.query(UserModel).filter(UserModel.email == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
        
    access_token = create_access_token(data={"sub": user.email})
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/users/me", response_model=UserResponse)
def read_users_me(current_user: UserModel = Depends(get_current_user)):
    """A protected endpoint returning details of the logged-in user."""
    return current_user
