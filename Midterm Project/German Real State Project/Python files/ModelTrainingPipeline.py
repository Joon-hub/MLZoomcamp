# Import libraries
import os
import pandas as pd
import numpy as np
from sklearn.svm import SVR
import logging
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score

from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression,Lasso, Ridge, ElasticNet
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline

import logging

# Initialize logging
logging.basicConfig(filename="model_training.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Check the current working directory
current_dir = os.getcwd()
logging.info(f"Current working directory: {current_dir}")

# Check if the file exists
file_path = '/Users/sudhirjoon/Library/Mobile Documents/com~apple~CloudDocs/Uni_Mannheim/Sem2/Machine Learning zoomcamp/MLZoomcamp/Midterm Project/German Real State Project/Data/preprocessed_data.csv'
if os.path.exists(file_path):
    df = pd.read_csv(file_path)
    logging.info("Dataset loaded successfully. Shape: %s", df.shape)
else:
    logging.error("File not found: %s", file_path)
    raise FileNotFoundError(f"File '{file_path}' not found in the current directory")

# Check for NA values and remove them
initial_shape = df.shape
df.dropna(inplace=True)
final_shape = df.shape

# Log the number of rows removed due to NA values
logging.info("Number of rows removed due to NA values: %s", initial_shape[0] - final_shape[0])

# Split the data into features (X) and target (y)
X = df.drop(['totalRent', 'numerical__num_pipeline__totalRentPerSquareMeter'], axis=1)
y = df['totalRent']

# Split the data into training, validation, and testing sets
df_full_train, df_test, y_full_train,y_test = train_test_split(X,y, test_size=0.2, random_state=1)
df_train, df_val,y_train, y_val = train_test_split(df_full_train,y_full_train, test_size= 0.25, random_state=1)

# Convert data to dictionaries for DictVectorizer
X_train_dict = df_train.to_dict(orient='records')
X_val_dict = df_val.to_dict(orient='records')
X_test_dict = df_test.to_dict(orient='records')

# feature extraction
dv = DictVectorizer()
X_train = dv.fit_transform(X_train_dict)
X_val = dv.transform(X_val_dict)
X_test = dv.transform(X_test_dict)

# Log the shapes of the resulting datasets
logging.info(f"Data split completed:")
logging.info(f"X_train shape: {X_train.shape}")
logging.info(f"X_val shape: {X_val.shape}")
logging.info(f"X_test shape: {X_test.shape}")

# Define parameter grids for RandomizedSearchCV
print("Defining parameter grids for RandomizedSearchCV")
param_grids = {
    'linear': {},
    
    "svr": {
        "C": [0.1, 1, 10, 100],  
        "epsilon": [0.01, 0.1, 0.2, 0.5], 
        "kernel": ["linear", "poly", "rbf", "sigmoid"],  
        "gamma": ["scale", "auto"],  
        "max_iter": [1000, 5000, 10000]  
    },
    
    "mlp": {
        'hidden_layer_sizes': [(50,), (100,), (50, 25), (100, 50)],
        'activation': ['relu', 'tanh'],
        'solver': ['adam'],
        'alpha': [0.001, 0.01, 0.1, 1],
        'learning_rate': ['constant', 'adaptive'],
        'max_iter': [1000, 2000, 3000]
        },

    "dt": {
        'max_depth': [None, 5, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None],
        'splitter': ['best', 'random']
        },

    "rf":{
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 5, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None]
        },

    "xgb": {
        'n_estimators': [100, 200, 500],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 6, 10],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0],
        'alpha': [0, 0.1, 0.5],
        'lambda': [0, 0.1, 1.0],
        'gamma': [0, 0.1, 0.5]
        }
}


# Models to tune
print("\n================================")
print("Models to tune:")
models = {
    "linear": LinearRegression(),
    "svr":SVR(),
    "mlp": MLPRegressor(max_iter=1000),
    "dt": DecisionTreeRegressor(),
    "rf": RandomForestRegressor(),
    "xgb": XGBRegressor(objective='reg:squarederror', random_state=42),
}

# Training Models with RandomizedSearchCV 
tuned_models = {}
cv_results = {}

for name, model in models.items():
    logging.info(f"Fitting model {name} with RandomizedSearchCV...")
    if name in param_grids:
        search = RandomizedSearchCV(
            model,
            param_distributions=param_grids[name],
            n_iter=20,
            cv=3,
            scoring='neg_mean_squared_error',
            random_state=42
        )
        search.fit(X_train, y_train)
        tuned_models[name] = search.best_estimator_
        cv_results[name] = search.cv_results_
        logging.info("Best parameters for %s: %s" % (name, search.best_params_))
        logging.info("Best score for %s: %s" % (name, -search.best_score_))
        logging.info("-------------------------------------------------------")
        logging.info("")
    else:
        logging.info(f"Model {name} does not have parameters to tune.")
        best_model = model.fit(X_train, y_train)

    # Store the tuned model
    tuned_models[name] = best_model

    # Cross-validation scores
    cv_scores = cross_val_score(best_model, X_train, y_train, cv= 5, scoring = 'neg_mean_squared_error')
    cv_results[name] = np.mean(-cv_scores)
    logging.info(f"Cross-validation score for {name}: {-cv_scores.mean().round(2)}")
    logging.info("")

    # Make predictions on the training set
    y_pred_train = best_model.predict(X_train)
    score = np.sqrt(mean_squared_error(y_train, y_pred_train))
    logging.info(f"RMSE error on Traing Set {name}: {score.round(2)}")
    logging.info("")

    # Make predictions on the validation set
    y_pred_val = best_model.predict(X_val)

    # Calculate RMSE for validation
    score = np.sqrt(mean_squared_error(y_val, y_pred_val))
    logging.info(f"RMSE error on Validation Set {name}: {score.round(2)}")
    logging.info("")

    # Make predictions on the test set
    y_pred_test = best_model.predict(X_test)

    # Calculate RMSE for test
    score = np.sqrt(mean_squared_error(y_test, y_pred_test))
    logging.info(f"RMSE error for {name} on test set: {score.round(2)}")
    logging.info("")
    logging.info("-------------------------------------------------------")
    

