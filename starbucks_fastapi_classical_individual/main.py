from fastapi import FastAPI, Request, Form, HTTPException, Depends
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse, PlainTextResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import uvicorn
from model_utils import load_model_and_scaler, prepare_input_for_inference, predict_n_days
import pandas as pd
import os
from pymongo import MongoClient
from passlib.context import CryptContext
from datetime import datetime, timedelta
from typing import Optional

# Try to import a JWT implementation. Prefer PyJWT, fall back to python-jose.
try:
    import jwt  # PyJWT
    JWT_LIB = 'pyjwt'
except Exception:
    try:
        from jose import jwt  # python-jose
        JWT_LIB = 'jose'
    except Exception:
        raise ImportError(
            "Missing a JWT library. Install PyJWT (`pip install PyJWT`) or python-jose (`pip install python-jose[cryptography]`)"
        )

# Initialize FastAPI app
app = FastAPI()
# Mount static and template directories with absolute paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")

# MongoDB setup
client = MongoClient('mongodb://localhost:27017/')
db = client['starbucks_stock']
users_collection = db['users']

# Security setup
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()
SECRET_KEY = "your-secret-key"  # Change this to a secure secret key
ALGORITHM = "HS256"

COMPANIES = ["SBUX"]

model, scaler = load_model_and_scaler()
class StockData(BaseModel):
    recent_closes: list[float]

class UserLogin(BaseModel):
    username: str
    password: str

class UserCreate(BaseModel):
    username: str
    password: str
    
    @property
    def safe_password(self) -> str:
        """Ensure password meets bcrypt requirements (max 72 bytes)"""
        return self.password.encode('utf-8')[:72].decode('utf-8')

# Helper functions
def verify_password(plain_password, hashed_password):
    # Truncate password to bcrypt max length
    safe_password = plain_password.encode('utf-8')[:72].decode('utf-8')
    return pwd_context.verify(safe_password, hashed_password)

def get_password_hash(password):
    # Truncate password to bcrypt max length
    safe_password = password.encode('utf-8')[:72].decode('utf-8')
    return pwd_context.hash(safe_password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    request: Request = None
):
    # Accept token from Authorization header (Bearer) or from cookie
    token = None
    if credentials and getattr(credentials, 'credentials', None):
        token = credentials.credentials
    elif request is not None:
        token = request.cookies.get('access_token')

    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid authentication credentials")
        return username
    except Exception as e:
        err_name = e.__class__.__name__
        if 'Expired' in err_name:
            raise HTTPException(status_code=401, detail="Token has expired")
        raise HTTPException(status_code=401, detail="Could not validate credentials")

@app.get("/")
async def form(request: Request, company: str = "SBUX"):
    # Check if user is authenticated via cookie
    token = request.cookies.get("access_token")
    if not token:
        # Redirect to login if no token
        return RedirectResponse(url="/login", status_code=303)
    try:
        # Verify token
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if not username:
            return RedirectResponse(url="/login?error=invalid_token", status_code=303)
    except Exception as e:
        print(f"Token validation error: {str(e)}")  # Debug print
        return RedirectResponse(url="/login?error=invalid_token", status_code=303)
        
    return templates.TemplateResponse("index.html", {
        "request": request,
        "companies": COMPANIES,
        "company": company,
        "username": username
    })

@app.get("/login")
async def login_page(request: Request):
    print("Accessing login page") # Debug print
    try:
        user_count = users_collection.count_documents({})
        print(f"Found {user_count} users") # Debug print
    except Exception as e:
        print(f"MongoDB error: {str(e)}") # Debug print
        user_count = 0
    show_register = user_count == 0
    return templates.TemplateResponse(
        "login.html",
        {
            "request": request,
            "show_register": show_register,
            "error": request.query_params.get("error")
        }
    )


@app.get("/login-test", response_class=PlainTextResponse)
def login_test():
    """Simple test endpoint to verify the server is running and routes are reachable."""
    return "login route reachable"

@app.post("/api/register")
async def register(user: UserCreate):
    if users_collection.find_one({"username": user.username}):
        raise HTTPException(status_code=400, detail="Username already registered")
    
    hashed_password = get_password_hash(user.password)
    user_data = {
        "username": user.username,
        "hashed_password": hashed_password,
        "created_at": datetime.utcnow()
    }
    
    users_collection.insert_one(user_data)
    return {"message": "User created successfully"}

@app.post("/api/login")
async def login(user: UserLogin):
    db_user = users_collection.find_one({"username": user.username})
    if not db_user or not verify_password(user.password, db_user["hashed_password"]):
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    
    access_token = create_access_token(
        data={"sub": user.username},
        expires_delta=timedelta(minutes=30)
    )
    # Return token and set it as a secure httpOnly cookie so form posts send it automatically
    response = JSONResponse(content={"token": access_token})
    response.set_cookie(key="access_token", value=access_token, httponly=True)
    return response


@app.post("/api/logout")
async def logout():
    response = JSONResponse(content={"message": "Logged out successfully"})
    response.delete_cookie(key="access_token")
    return response


@app.post("/predict_form")
async def predict_form(
    request: Request,
    company: str = Form("SBUX"),
    days: int = Form(7)
):
    # Check authentication
    token = request.cookies.get("access_token")
    if not token:
        return RedirectResponse(url="/login", status_code=303)
    try:
        # Verify token
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        current_user = payload.get("sub")
        if not current_user:
            return RedirectResponse(url="/login?error=invalid_token", status_code=303)
    except Exception as e:
        print(f"Token validation error: {str(e)}")  # Debug print
        return RedirectResponse(url="/login?error=invalid_token", status_code=303)
    try:
        csv_path = os.path.join(os.path.dirname(__file__), f"{company}.csv")
        if not os.path.exists(csv_path):
            return templates.TemplateResponse("index.html", {
                "request": request,
                "error": f"CSV file not found at {csv_path}",
                "companies": COMPANIES,
                "company": company
            })

        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        if 'Close' not in df.columns:
            return templates.TemplateResponse("index.html", {
                "request": request,
                "error": "CSV does not contain 'Close' column.",
                "companies": COMPANIES,
                "company": company
            })

        closes = df['Close'].dropna().tolist()
        window_size = 1000
        if len(closes) < window_size:
            return templates.TemplateResponse("index.html", {
                "request": request,
                "error": f"Insufficient data in CSV: need at least {window_size} closes.",
                "companies": COMPANIES,
                "company": company
            })

        recent = closes[-window_size:]
        inv_preds = predict_n_days(
            model=model,
            scaler=scaler,
            recent_data=recent,
            window_size=window_size,
            n_days=days
        )
        preds_rounded = [f"{x:.2f}" for x in inv_preds]
        
        # Log prediction in MongoDB
        prediction_log = {
            "user": current_user,
            "company": company,
            "prediction_date": datetime.utcnow(),
            "days_predicted": days,
            "predictions": preds_rounded
        }
        db['prediction_logs'].insert_one(prediction_log)
        
        return templates.TemplateResponse("index.html", {
            "request": request,
            "predictions": preds_rounded,
            "companies": COMPANIES,
            "company": company
        })
    except Exception as e:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "error": str(e),
            "companies": COMPANIES,
            "company": company
        })


if __name__ == "__main__":
    
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="127.0.0.1", port=port)
