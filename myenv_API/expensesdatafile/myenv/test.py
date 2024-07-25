from fastapi import FastAPI,Request, HTTPException, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import jwt
from datetime import datetime, timedelta
import json
import base64
from dotenv import load_dotenv
import os
import uvicorn
from app import query_pdf

load_dotenv()

app = FastAPI()

# JWT Authentication
security = HTTPBearer()

# Setting secret key for JWT encryption
app.secret_key = 'test123'

# Token Expiration time
Token_Expiration_time = 30

USERS_FILE_PATH = 'C:/genaiCheck/expensesdatafile/myenv/user_info_json'

# CORS configuration (add your CORS settings if needed)

# Loading Master credentials
def load_master_credentials():
    try:
        with open(USERS_FILE_PATH, "r") as f:
            master_credentials = json.load(f)
    except FileNotFoundError:
        master_credentials = {}
        with open(USERS_FILE_PATH, "w") as f:
            json.dump(master_credentials, f)
    return master_credentials

master_credentials = load_master_credentials()

# Documentation
@app.get("/")
async def get_docs():
    return {"message": "Welcome to the API Docs"}

# Encode and decode functions for passwords
def encode_password(password):
    return base64.b64encode(password.encode()).decode()

def decode_password(encoded_password):
    return base64.b64decode(encoded_password).decode()

# Request Model for Login
class UserLogin(BaseModel):
    username: str
    password: str

# Endpoint to register a new user
@app.post('/register')
async def register(user: UserLogin):
    if user.username in master_credentials:
        raise HTTPException(status_code=400, detail='User already exists')
    
    master_credentials[user.username] = encode_password(user.password)
    with open(USERS_FILE_PATH, "w") as f:
        json.dump(master_credentials, f)
    
    return {"message": "User registered successfully"}

# Endpoint to login
@app.post('/login')
async def login(user: UserLogin):
    if user.username in master_credentials and decode_password(master_credentials[user.username]) == user.password:
        # Set Token Expiration Time
        expiration_time = datetime.utcnow() + timedelta(minutes=Token_Expiration_time)
        
        # Generate JWT Token with Expiration Time
        token = jwt.encode({"username": user.username, 'exp': expiration_time}, app.secret_key, algorithm='HS256')
        
        response = {
            'response_code': 200,
            'data': {
                'access_token': token,
                'expires_in': Token_Expiration_time * 60000
            }
        }
        return response
    
    raise HTTPException(status_code=401, detail='Invalid Userid or Password')

## QueryRequest
class SearchRequest(BaseModel):
    query_prompt: str
    #temperature: float =0.5

## Function to generate response

## API to retrieve

@app.post('/search')
async def search(
    request: SearchRequest,
    credentials : HTTPAuthorizationCredentials = Depends(HTTPBearer())
):
    token=credentials.credentials
    try:
        payload = jwt.decode(token, app.secret_key, algorithms=['HS256'])

        # payload =await request.json()
        # query_request=SearchRequest(**payload)
        # query_prompt=query_request.query_prompt
        #temperature=query_request.temperature

            # Prompt for Query Search
        prompt =f"User:{request.query_prompt}\nAI:"

        response =query_pdf(prompt)

        ## Prepare the response JSON
        response_json={
            'response_code': 200,
            'llm_response':response,
        }
        return response_json

    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail='Token Expired')
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail='Invalid Token')


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8080)


    
    


    