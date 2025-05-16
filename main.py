import os
import traceback
import re  # Add this to your imports
from datetime import datetime, timedelta
from typing import Optional, List, Dict

import uvicorn
from fastapi import FastAPI, Depends, HTTPException, status, Request, Form, WebSocket, WebSocketDisconnect
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.exception_handlers import http_exception_handler, request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, Boolean, Text, DateTime, ForeignKey
from sqlalchemy.orm import sessionmaker, Session, relationship
from sqlalchemy.ext.declarative import declarative_base
import json
from google import genai
from google.genai import types
import markdown
from bs4 import BeautifulSoup

# Create necessary directories if they don't exist
os.makedirs("static/css", exist_ok=True)
os.makedirs("static/js", exist_ok=True)
os.makedirs("templates", exist_ok=True)

# Configuration
SECRET_KEY = "YOUR_SECRET_KEY"  # Change this to a secure random key
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Gemini API configuration
GEMINI_API_KEY = "AIzaSyDptZ01TC5k8bcmmtmGXlcngvavBjEbkBI"
os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY
MODEL_NAME = "gemini-2.0-flash"

# Initialize Gemini client
from google import genai
from google.genai import types

# Create a client with the API key
client = genai.Client(api_key=GEMINI_API_KEY)

# System prompt to guide the model's behavior
SYSTEM_PROMPT = """
You are a helpful legal assistant that answers questions based on your knowledge of legal concepts.

IMPORTANT RULES:
1. NEVER provide generic overviews of legal systems unless explicitly asked.
2. ALWAYS address the specific question asked by the user.
3. If the question is vague, ask for clarification instead of giving a generic response.
4. Do not mention Nepal or any other specific country unless the user specifically asks about it.
5. Keep your answers focused on the exact topic of the question.
6. If you're not sure about something, be honest about your limitations.
7. Format your responses with clear structure using markdown.

Remember: You are providing educational information, not personalized legal advice. Always recommend consulting with a qualified legal professional for specific situations.
"""

# Define common legal topics and their specialized prompts
LEGAL_TOPICS = {
    "tenant rights": """
When discussing tenant rights, be sure to:
1. Explain the basic rights tenants have regarding habitability, privacy, and security deposits
2. Note that landlord-tenant laws vary significantly by location
3. Cover common issues like repairs, eviction processes, and lease terminations
4. Mention resources like tenant unions and legal aid organizations
""",
    "divorce": """
When discussing divorce, be sure to:
1. Explain the difference between contested and uncontested divorce
2. Cover key aspects like division of assets, child custody, and support
3. Outline the general process for filing for divorce
4. Emphasize the importance of legal representation
""",
    "business formation": """
When discussing business formation, be sure to:
1. Explain different business structures (sole proprietorship, LLC, corporation, etc.)
2. Cover basic registration requirements and tax implications
3. Mention important considerations like liability protection
4. Discuss the importance of business plans and operating agreements
""",
    "criminal law": """
When discussing criminal law, be sure to:
1. Distinguish between misdemeanors and felonies
2. Explain basic rights of the accused
3. Outline the general criminal justice process
4. Emphasize the importance of legal representation
""",
    "intellectual property": """
When discussing intellectual property, be sure to:
1. Explain different types of IP (patents, trademarks, copyrights, trade secrets)
2. Cover basic registration processes and protection durations
3. Discuss common infringement issues
4. Mention international considerations
"""
}

# Function to detect legal topics in a query
def detect_legal_topics(query: str) -> List[str]:
    query_lower = query.lower()
    detected_topics = []
    
    for topic in LEGAL_TOPICS:
        if topic in query_lower or any(keyword in query_lower for keyword in topic.split()):
            detected_topics.append(topic)
    
    return detected_topics

# Database setup
SQLALCHEMY_DATABASE_URL = "sqlite:///./legal_chatbot.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Models
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    is_active = Column(Boolean, default=True)
    full_name = Column(String, nullable=True)
    phone = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship with chat history
    chat_messages = relationship("ChatMessage", back_populates="user")

class ChatMessage(Base):
    __tablename__ = "chat_messages"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    is_user = Column(Boolean, default=True)  # True if message is from user, False if from bot
    message = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Relationship with user
    user = relationship("User", back_populates="chat_messages")

# Pydantic models
class UserCreate(BaseModel):
    username: str
    email: str
    password: str

class UserUpdate(BaseModel):
    full_name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None

class UserInDB(BaseModel):
    id: int
    username: str
    email: str
    is_active: bool
    full_name: Optional[str] = None
    phone: Optional[str] = None
    created_at: datetime

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

class ChatMessageCreate(BaseModel):
    message: str
    is_user: bool = True

class ChatMessageResponse(BaseModel):
    id: int
    message: str
    is_user: bool
    timestamp: datetime

# Create tables
Base.metadata.create_all(bind=engine)

# FastAPI app
app = FastAPI(title="Legal Chatbot")

# Add exception handlers
@app.exception_handler(StarletteHTTPException)
async def custom_http_exception_handler(request, exc):
    return await http_exception_handler(request, exc)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return await request_validation_exception_handler(request, exc)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    # Log the error
    print(f"Global exception: {exc}")
    import traceback
    traceback.print_exc()
    
    # Return a friendly error page
    return templates.TemplateResponse(
        "error.html", 
        {
            "request": request, 
            "error_message": str(exc),
            "status_code": 500
        },
        status_code=500
    )

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Helper functions
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def get_user(db: Session, username: str):
    return db.query(User).filter(User.username == username).first()

def authenticate_user(db: Session, username: str, password: str):
    user = get_user(db, username)
    if not user or not verify_password(password, user.hashed_password):
        return False
    return user

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    user = get_user(db, username=token_data.username)
    if user is None:
        raise credentials_exception
    return user

def get_cookie_user(request: Request, db: Session = Depends(get_db)):
    token = request.cookies.get("access_token")
    if not token:
        return None
    try:
        token = token.replace("Bearer ", "")
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            return None
        user = get_user(db, username=username)
        return user
    except JWTError:
        return None

# Updated function to get response from Gemini
def get_legal_response(query: str, chat_history: List[ChatMessage] = None) -> Dict:
    try:
        # Create prompt with system instruction and query
        prompt_text = f"{SYSTEM_PROMPT}\n\nUser query: {query}\n\nPlease provide a helpful response that directly addresses the question without mentioning Nepal or any specific country unless explicitly asked."
        
        # Generate content using the client's models.generate_content method
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt_text,
                config=types.GenerateContentConfig(
                    temperature=0.2,
                    top_p=0.8,
                    top_k=40,
                    max_output_tokens=1024,
                )
            )
            
            # Extract text from response
            if hasattr(response, 'text'):
                raw_answer = response.text
            else:
                raw_answer = str(response)
        except Exception as inner_error:
            print(f"Inner error: {str(inner_error)}")
            traceback.print_exc()
            raw_answer = f"I'm sorry, I encountered an error processing your request. Error: {str(inner_error)}"
        
        # Remove any code block markers
        raw_answer = raw_answer.replace("```html", "").replace("```", "")
        
        # Convert markdown to HTML
        html_answer = markdown.markdown(raw_answer)
        
        # Clean up the HTML to ensure it's safe and well-formatted
        soup = BeautifulSoup(html_answer, 'html.parser')
        
        # Add CSS classes for styling
        for tag in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
            tag['class'] = tag.get('class', []) + ['legal-heading']
        
        for tag in soup.find_all('p'):
            tag['class'] = tag.get('class', []) + ['legal-paragraph']
        
        for tag in soup.find_all(['ul', 'ol']):
            tag['class'] = tag.get('class', []) + ['legal-list']
        
        for tag in soup.find_all('li'):
            tag['class'] = tag.get('class', []) + ['legal-list-item']
        
        for tag in soup.find_all(['strong', 'b']):
            tag['class'] = tag.get('class', []) + ['legal-bold']
        
        for tag in soup.find_all(['em', 'i']):
            tag['class'] = tag.get('class', []) + ['legal-italic']
        
        formatted_answer = str(soup)
        
        return {
            "response": formatted_answer,
            "sources": []  # No sources in this implementation
        }
    except Exception as e:
        print(f"Error getting response from Gemini: {str(e)}")
        traceback.print_exc()
        return {
            "response": f"I'm sorry, I encountered an error processing your request. Error: {str(e)}",
            "sources": []
        }

# Routes
@app.get("/", response_class=HTMLResponse)
async def home(request: Request, db: Session = Depends(get_db)):
    user = get_cookie_user(request, db)
    return templates.TemplateResponse("home.html", {"request": request, "user": user})

@app.get("/about", response_class=HTMLResponse)
async def about(request: Request, db: Session = Depends(get_db)):
    user = get_cookie_user(request, db)
    return templates.TemplateResponse("about.html", {"request": request, "user": user})

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.get("/signup", response_class=HTMLResponse)
async def signup_page(request: Request):
    return templates.TemplateResponse("signup.html", {"request": request})

@app.post("/token")
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/login", response_class=HTMLResponse)
async def login(request: Request, username: str = Form(...), password: str = Form(...), db: Session = Depends(get_db)):
    user = authenticate_user(db, username, password)
    if not user:
        return templates.TemplateResponse("login.html", {"request": request, "error": "Invalid username or password"})
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    
    response = RedirectResponse(url="/chatbot", status_code=status.HTTP_303_SEE_OTHER)
    response.set_cookie(key="access_token", value=f"Bearer {access_token}", httponly=True)
    return response

@app.post("/signup", response_class=HTMLResponse)
async def signup(
    request: Request,
    username: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db)
):
    # Check if username already exists
    db_user = get_user(db, username)
    if db_user:
        return templates.TemplateResponse(
            "signup.html", 
            {"request": request, "error": "Username already registered"}
        )
    
    # Create new user
    hashed_password = get_password_hash(password)
    new_user = User(username=username, email=email, hashed_password=hashed_password)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    
    # Auto-login after signup
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": new_user.username}, expires_delta=access_token_expires
    )
    
    response = RedirectResponse(url="/chatbot", status_code=status.HTTP_303_SEE_OTHER)
    response.set_cookie(key="access_token", value=f"Bearer {access_token}", httponly=True)
    return response

@app.get("/profile", response_class=HTMLResponse)
async def profile(request: Request, db: Session = Depends(get_db)):
    user = get_cookie_user(request, db)
    if not user:
        return RedirectResponse(url="/login", status_code=status.HTTP_303_SEE_OTHER)
    
    # Get chat history for the user
    chat_history = db.query(ChatMessage).filter(ChatMessage.user_id == user.id).order_by(ChatMessage.timestamp).all()
    return templates.TemplateResponse("profile.html", {"request": request, "user": user, "chat_history": chat_history})

@app.post("/update-profile", response_class=HTMLResponse)
async def update_profile(
    request: Request,
    full_name: str = Form(None),
    email: str = Form(None),
    phone: str = Form(None),
    db: Session = Depends(get_db)
):
    user = get_cookie_user(request, db)
    if not user:
        return RedirectResponse(url="/login", status_code=status.HTTP_303_SEE_OTHER)
    
    # Update user profile
    if full_name:
        user.full_name = full_name
    if email:
        user.email = email
    if phone:
        user.phone = phone
    
    db.commit()
    db.refresh(user)
    
    return RedirectResponse(url="/profile", status_code=status.HTTP_303_SEE_OTHER)

@app.get("/logout")
async def logout():
    response = RedirectResponse(url="/", status_code=status.HTTP_303_SEE_OTHER)
    response.delete_cookie(key="access_token")
    return response

@app.get("/chatbot", response_class=HTMLResponse)
async def chatbot_page(request: Request, db: Session = Depends(get_db)):
    user = get_cookie_user(request, db)
    if not user:
        return RedirectResponse(url="/login", status_code=status.HTTP_303_SEE_OTHER)
    
    # Get chat history for the user
    chat_history = db.query(ChatMessage).filter(ChatMessage.user_id == user.id).order_by(ChatMessage.timestamp).all()
    
    return templates.TemplateResponse("chatbot.html", {
        "request": request, 
        "user": user, 
        "chat_history": chat_history,
        "now": datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
    })

@app.post("/query")
async def query(request: Request, db: Session = Depends(get_db)):
    user = get_cookie_user(request, db)
    if not user:
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={"error": "Authentication required"}
        )
    
    try:
        # Parse the request body
        data = await request.json()
        user_message = data.get("query", "")
        
        if not user_message:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"error": "Query is required"}
            )
        
        print(f"Processing query: {user_message}")
        
        # Get chat history for context
        chat_history = db.query(ChatMessage).filter(ChatMessage.user_id == user.id).order_by(ChatMessage.timestamp).all()
        
        # Save user message to history
        user_chat_message = ChatMessage(user_id=user.id, message=user_message, is_user=True)
        db.add(user_chat_message)
        db.commit()
        
        # Get response from the Gemini model
        response_data = get_legal_response(user_message, chat_history)
        
        # Save bot response to history
        bot_chat_message = ChatMessage(user_id=user.id, message=response_data["response"], is_user=False)
        db.add(bot_chat_message)
        db.commit()
        
        return {
            "response": response_data["response"],
            "sources": response_data["sources"]
        }
    except Exception as e:
        print(f"Error in query endpoint: {str(e)}")
        traceback.print_exc()
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": str(e)}
        )

@app.get("/chat-history")
async def get_chat_history(request: Request, db: Session = Depends(get_db)):
    user = get_cookie_user(request, db)
    if not user:
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={"error": "Authentication required"}
        )
    
    chat_history = db.query(ChatMessage).filter(ChatMessage.user_id == user.id).order_by(ChatMessage.timestamp).all()
    return [{"id": msg.id, "message": msg.message, "is_user": msg.is_user, "timestamp": msg.timestamp} for msg in chat_history]

@app.delete("/clear-history")
async def clear_chat_history(request: Request, db: Session = Depends(get_db)):
    user = get_cookie_user(request, db)
    if not user:
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={"error": "Authentication required"}
        )
    
    try:
        db.query(ChatMessage).filter(ChatMessage.user_id == user.id).delete()
        db.commit()
        return {"message": "Chat history cleared successfully"}
    except Exception as e:
        print(f"Error clearing chat history: {str(e)}")
        traceback.print_exc()
        db.rollback()
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": f"Failed to clear history: {str(e)}"}
        )

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
