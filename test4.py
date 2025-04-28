import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import streamlit as st
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.ensemble import ExtraTreesRegressor

warnings.filterwarnings('ignore')

# Function to load and process data
def load_data():
    data = pd.read_csv("admission_predict_system.csv")
    data.drop('Serial No.', axis=1, inplace=True)
    return data

# Streamlit App
st.set_page_config(page_title="Admission Prediction", page_icon="🎓", layout="wide")

# Custom CSS for styling
st.markdown(
    """
    <style>
    .main {
        background-color: #f0f2f5;
    }
    .header {
        text-align: center;
        color: #2c3e50;
        font-size: 2.5em;
        font-weight: bold;
    }
    .subheader {
        color: #2980b9;
        font-size: 1.5em;
        font-weight: bold;
    }
    .stButton>button {
        background-color: #3498db;
        color: white;
        border-radius: 5px;
        height: 3em;
        width: 10em;
        font-size: 1em;
    }
    .stButton>button:hover {
        background-color: #2980b9;
    }
    </style>
    """, unsafe_allow_html=True
)

st.title("🎓 Admission Prediction Data Analysis & Regression Models")

# Load Data
data = load_data()

# Independent and Dependent Features
X = data.drop('Chance of Admit ', axis=1)
y = data['Chance of Admit ']

# Data Analytics Header
st.header("Data Analytics")

# Data Description
if st.button("Display Data Description", key='desc', help="Show descriptive statistics."):
    st.subheader("Descriptive Statistics")
    st.dataframe(data.describe().T)

# Unique Values
if st.button("Display Unique Values", key='unique', help="Show unique values in each column."):
    st.subheader("Unique Values in Each Column")
    st.dataframe(data.nunique())

# Column Names
if st.button("Display Column Names", key='columns', help="Show column names."):
    st.subheader("Column Names")
    st.write(data.columns.tolist())

# Data Distribution for Each Column
if st.button("Display Data Distribution", key='distribution', help="Show data distribution for each column."):
    st.subheader("Data Distribution for Each Column")
    fig, axes = plt.subplots(5, 2, figsize=(12, 25))
    columnnumber = 1

    for column in data.columns:
        if columnnumber <= 10:
            sns.histplot(data[column], kde=True, ax=axes[(columnnumber-1)//2, (columnnumber-1)%2])
            axes[(columnnumber-1)//2, (columnnumber-1)%2].set_xlabel(column, fontsize=20)
        columnnumber += 1

    st.pyplot(fig)

# Correlation Heatmap
if st.button("Display Correlation Heatmap", key='heatmap', help="Show correlation heatmap of the data."):
    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(data.corr(), annot=True, cmap='icefire', ax=ax)
    st.pyplot(fig)

# Splitting the dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Displaying X_train and y_train
if st.button("Display number of rows in  X_train and y_train ", key='train_data', help="Show training dataset."):
    st.subheader("X_train Values")
    st.dataframe(X_train)
    st.subheader("y_train Values")
    st.dataframe(y_train)

# Standardization of data
scaler = StandardScaler()

# Scaling of train and test data
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model Evaluation Summary Header
st.header("Model Evaluation Summary")

# Define and evaluate models in a reusable function
def evaluate_model(model, model_name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    adj_r2 = 1 - (1 - r2) * (len(y_test) - 1) / (len(y_test) - X_test.shape[1] - 1)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    return {
        'name': model_name,
        'r2': r2,
        'adj_r2': adj_r2,
        'mae': mae,
        'mse': mse,
        'rmse': rmse
    }

# Evaluate all models
models = [
    (LinearRegression(), "Linear Regression"),
    (Ridge(), "Ridge Regression"),
    (SVR(), "Support Vector Regression"),
    (DecisionTreeRegressor(random_state=0), "Decision Tree Regressor")
]

results = []
for model, name in models:
    results.append(evaluate_model(model, name))

# Displaying the Model Evaluation Summary
if st.button("Display Model Evaluation Summary", key='eval_summary', help="Show model evaluation metrics."):
    for result in results:
        st.subheader(result['name'])
        st.write(f"R²: {round(result['r2'] * 100, 2)}%")
        st.write(f"Adjusted R²: {round(result['adj_r2'] * 100, 2)}%")
        st.write(f"MAE: {result['mae']}")
        st.write(f"MSE: {result['mse']}")
        st.write(f"RMSE: {result['rmse']}")

## ElasticNet Regression
## Training of Model
elasticnet = ElasticNet()
elasticnet.fit(X_train, y_train)
elasticnet_test = elasticnet.score(X_test, y_test)
elasticnet_train =elasticnet.score(X_train,y_train)
model_metrics(elasticnet)
## Prediction from the test data
elasticnet_pred=elasticnet.predict(X_test)
print(elasticnet_pred)
## residuals
residuals=y_test-elasticnet_pred
residuals
## computing R Square 
from sklearn.metrics import r2_score
elasticnet_r2=r2_score(y_test,elasticnet_pred)
Adjust_r_EN=1 - (1-rsquare_ln)*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1)
MAE_EN=(metrics.mean_absolute_error(y_test,elasticnet.predict(X_test)))
MSE_EN =(metrics.mean_squared_error(y_test,elasticnet.predict(X_test)))
RMSE_EN=(np.sqrt(metrics.mean_squared_error(y_test,elasticnet.predict(X_test))))


## Random Forest Regressor
## Training of Model
rf = RandomForestRegressor()
rf.fit(X_train,y_train)
rf_test = rf.score(X_test, y_test)
rf_train =rf.score(X_train,y_train)
model_metrics(rf)

## Prediction of Test Data
y_rf = rf.predict(X_test)
y_rf

## R square of Random Forest Regressor
rsquare_rf=metrics.r2_score(y_test, y_rf)
rsquare_rf

Adjust_r_htrf=1 - (1-rsquare_htrf)*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1)
MAE_htrf=(metrics.mean_absolute_error(y_test,rf_random.predict(X_test)))
MSE_htrf =(metrics.mean_squared_error(y_test,rf_random.predict(X_test)))
RMSE_htrf=(np.sqrt(metrics.mean_squared_error(y_test,rf_random.predict(X_test))))

