import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import streamlit as st
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn import metrics
##  Importing RandomizedSearchCV
from sklearn.model_selection import RandomizedSearchCV

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
    (ElasticNet(), "ElasticNet Regression"),
    (DecisionTreeRegressor(random_state=0), "Decision Tree Regressor"),
    (RandomForestRegressor(), "Random Forest Regressor")
]

results = []
for model, name in models:
    results.append(evaluate_model(model, name))

# Create DataFrame for all models' metrics
models_df = pd.DataFrame({
    'Model Name': [result['name'] for result in results],
    'R Square Value': [round(result['r2'], 4) for result in results],
    'Adjusted R square value': [round(result['adj_r2'], 4) for result in results],
    'Mean Absolute Error': [round(result['mae'], 2) for result in results],
    'Mean Square Error': [round(result['mse'], 2) for result in results],
    'Root Mean Square Error': [round(result['rmse'], 2) for result in results]
})

# Displaying the Model Evaluation Summary
if st.button("Display Model Evaluation Summary", key='eval_summary', help="Show model evaluation metrics."):
    st.subheader("Model Evaluation Metrics")
    st.dataframe(models_df)

## Randomized Search CV

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10, 15, 100]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 5, 10]

# Random search of parameters, using 5 fold cross validation, 
# search across 100 different combinations
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid,scoring='neg_mean_squared_error', n_iter = 10, cv = 5, verbose=2, random_state=42, n_jobs = 1)

## Training of Model
rf_random.fit(X_train,y_train)
## The best estimator across ALL searched params
rf_random.best_estimator_

## The best parameters across ALL searched params
rf_random.best_params_

## Printing predicted Test Data
prediction = rf_random.predict(X_test)
prediction

## R Square value of Random Forest Regressor Model After Hyperparameter Tunning
rsquare_htrf=metrics.r2_score(y_test, prediction)
rsquare_htrf

## Errors of Random Forest Regressor Model After Hyperparameter Tunning
print("MAE is :" , metrics.mean_absolute_error(y_test,prediction))
print("MSE is :" , metrics.mean_squared_error(y_test,prediction))
print("RMSE is :" , np.sqrt(metrics.mean_squared_error(y_test,prediction)))

## Dataframe of Predicted Price Vs Actual Price
data_test_df = pd.DataFrame({"Predicted Price" : rf.predict(X_test),"Actual Price" : y_test}).reset_index(drop = True)
data_test_df

## Prediction
y_predict = rf_flightfare_prediction.predict(X_test)
y_predict

## R Square 
metrics.r2_score(y_test, y_predict)

Adjust_r_htrf=1 - (1-rsquare_htrf)*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1)
MAE_htrf=(metrics.mean_absolute_error(y_test,rf_random.predict(X_test)))
MSE_htrf =(metrics.mean_squared_error(y_test,rf_random.predict(X_test)))
RMSE_htrf=(np.sqrt(metrics.mean_squared_error(y_test,rf_random.predict(X_test))))