from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import uvicorn
from typing import List, Dict, Any
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from fastapi.staticfiles import StaticFiles




app = FastAPI(title="Admission Prediction API", 
              description="API for predicting admission chances and displaying model metrics",
              version="1.0.0")

# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Load and process data
def load_data():
    data = pd.read_csv("admission_predict_system.csv")
    data.drop('Serial No.', axis=1, inplace=True)
    return data

# Model variables
data = load_data()
X = data.drop('Chance of Admit ', axis=1)
y = data['Chance of Admit ']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Standardization of data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model training (default polynomial degree=2)
poly = PolynomialFeatures(degree=2)
X_poly_train = poly.fit_transform(X_train_scaled)
X_poly_test = poly.transform(X_test_scaled)
model = LinearRegression()
model.fit(X_poly_train, y_train)

# Input validation class
class AdmissionFeatures(BaseModel):
    gre: float
    toefl: float
    university_rating: float
    sop: float
    lor: float
    cgpa: float
    research: int
    polynomial_degree: int = 2

# Results class
class PredictionResults(BaseModel):
    prediction: float
    model_metrics: Dict[str, float]

# @app.get("/")
# def root():
#     return {"message": "Welcome to the Admission Prediction API"}

# Mount the static folder
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/api/model-metrics")
def get_model_metrics(polynomial_degree: int = 2):
    """
    Get the evaluation metrics for the model with the specified polynomial degree
    """
    try:
        # Create polynomial features with the specified degree
        poly = PolynomialFeatures(degree=polynomial_degree)
        X_poly_train = poly.fit_transform(X_train_scaled)
        X_poly_test = poly.transform(X_test_scaled)
        
        # Train the model
        model = LinearRegression()
        model.fit(X_poly_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_poly_test)
        
        # Calculate metrics
        r2 = r2_score(y_test, y_pred)
        adj_r2 = 1 - (1 - r2) * (len(y_test) - 1) / (len(y_test) - X_poly_test.shape[1] - 1)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        
        return {
            "r2_score": round(r2, 4),
            "adjusted_r2": round(adj_r2, 4),
            "mean_absolute_error": round(mae, 4),
            "mean_squared_error": round(mse, 4),
            "root_mean_squared_error": round(rmse, 4),
            "polynomial_degree": polynomial_degree
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating metrics: {str(e)}")

@app.post("/api/predict-admission", response_model=PredictionResults)
def predict_admission(features: AdmissionFeatures):
    """
    Predict admission chance based on input features and return model metrics
    """
    try:
        # Convert input features to DataFrame with correct column names
        input_df = pd.DataFrame({
            'GRE Score': [features.gre],
            'TOEFL Score': [features.toefl],
            'University Rating': [features.university_rating],
            'SOP': [features.sop],
            'LOR ': [features.lor],
            'CGPA': [features.cgpa],
            'Research': [features.research]
        })
        
        # Scale the input
        input_scaled = scaler.transform(input_df)
        
        # Create polynomial features with the specified degree
        poly_degree = features.polynomial_degree
        poly = PolynomialFeatures(degree=poly_degree)
        X_poly_train = poly.fit_transform(X_train_scaled)
        X_poly_test = poly.transform(X_test_scaled)
        
        # Train the model
        model = LinearRegression()
        model.fit(X_poly_train, y_train)
        
        # Transform input for prediction
        input_poly = poly.transform(input_scaled)
        
        # Make prediction
        prediction = model.predict(input_poly)[0]
        
        # Get model metrics
        y_pred = model.predict(X_poly_test)
        r2 = r2_score(y_test, y_pred)
        adj_r2 = 1 - (1 - r2) * (len(y_test) - 1) / (len(y_test) - X_poly_test.shape[1] - 1)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        
        return PredictionResults(
            prediction=round(float(prediction), 4),
            model_metrics={
                "r2_score": round(r2, 4),
                "adjusted_r2": round(adj_r2, 4),
                "mean_absolute_error": round(mae, 4),
                "mean_squared_error": round(mse, 4),
                "root_mean_squared_error": round(rmse, 4),
                "polynomial_degree": poly_degree
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error making prediction: {str(e)}")
    

templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("ii.html", {"request": request})


def evaluate_model(model, name):
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    r2 = r2_score(y_test, y_pred)
    adj_r2 = 1 - (1 - r2) * (len(y_test) - 1) / (len(y_test) - X_test_scaled.shape[1] - 1)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    return {
        'name': name,
        'r2': r2,
        'adj_r2': adj_r2,
        'mae': mae,
        'mse': mse,
        'rmse': rmse
    }


@app.get("/api/evaluate-all-models")
def evaluate_all_models():
    models = [
        (LinearRegression(), "Linear Regression"),
        (DecisionTreeRegressor(random_state=0), "Decision Tree Regressor"),
        (RandomForestRegressor(random_state=0), "Random Forest Regressor")
    ]
    results = [evaluate_model(model, name) for model, name in models]
    
    models_df = pd.DataFrame({
        'Model Name': [result['name'] for result in results],
        'R Square Value': [round(result['r2'], 4) for result in results],
        'Adjusted R square value': [round(result['adj_r2'], 4) for result in results],
        'Mean Absolute Error': [round(result['mae'], 2) for result in results],
        'Mean Square Error': [round(result['mse'], 2) for result in results],
        'Root Mean Square Error': [round(result['rmse'], 2) for result in results]
    })
    
    return models_df.to_dict(orient="records")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8081)
    # uvicorn.run(app, host="0.0.0.0", port=8000)

# if __name__ == "__main__":
#     # app.run(host="0.0.0.0", port=8081, reload=True)
#     uvicorn.run(host="0.0.0.0", port=8001)

# To run the API server:
# python api.py
#
# API Documentation will be available at:
# http://localhost:8000/docs
#
# Example API calls:
# - GET /api/model-metrics?polynomial_degree=2
# - POST /api/predict-admission with JSON body:
#   {
#     "gre": 337,
#     "toefl": 118,
#     "university_rating": 4,
#     "sop": 4.5,
#     "lor": 4.5,
#     "cgpa": 9.65,
#     "research": 1,
#     "polynomial_degree": 2
#   }