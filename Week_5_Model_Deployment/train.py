# Import necessary libraries
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
import pickle
import warnings
import argparse

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# adding parsers to add the parameters from command line arguments
parser = argparse.ArgumentParser(description='Train a logistic regression model.')
parser.add_argument('--C', type=float, default=1, help='Regularization strength (default: %(default)s)')
parser.add_argument('--n_splits', type=int, default=5, help='Number of splits for K-Fold cross-validation (default: 5)')
args = parser.parse_args()

# Training parameters
C = args.C  # Regularization strength
n_splits = args.n_splits  # Number of splits for K-Fold cross-validation

# Load dataset
df = pd.read_csv("churn_dataset.csv")
print(f'Dataset has been successfully loaded: {df.shape}\n')

# Standardize column names: lowercase and replace spaces with underscores
df.columns = df.columns.str.lower().str.replace(' ', '_')

# Identify categorical and numerical column names
categorical = df.select_dtypes(include='object').columns.to_list()
numerical = df.select_dtypes(exclude='object').columns.to_list()

print(f'Categorical columns: {categorical}')
print(f'Numerical columns: {numerical}')

# Preprocess categorical columns: convert to lowercase and replace spaces
for c in categorical:
    df[c] = df[c].str.lower().str.replace(' ', '_')

# Convert 'totalcharges' to numeric, handling errors and filling NaNs with 0
df.totalcharges = pd.to_numeric(df.totalcharges, errors='coerce').fillna(0)

# Map 'churn' values to integers: 'yes' to 1 and 'no' to 0
df.churn = df.churn.map({'yes': 1, 'no': 0}).fillna(0).astype(int)

# Split dataset into training and testing sets
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
print(f'\nShape of training dataset (df_full_train): {df_full_train.shape}')
print(f'Shape of testing dataset (df_test): {df_test.shape}\n')

# Define numerical and categorical features for model training
numerical = ['tenure', 'monthlycharges', 'totalcharges']
categorical = [
    'gender', 'seniorcitizen', 'partner', 'dependents', 'phoneservice',
    'multiplelines', 'internetservice', 'onlinesecurity', 'onlinebackup',
    'deviceprotection', 'techsupport', 'streamingtv', 'streamingmovies',
    'contract', 'paperlessbilling', 'paymentmethod',
]

# Function to train the model
def train(df_train, y_train, C=1.0):
    # Convert DataFrame to a list of dictionaries
    dicts = df_train[categorical + numerical].to_dict(orient='records')

    # Use DictVectorizer to handle categorical features
    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)

    # Fit the Logistic Regression model
    model = LogisticRegression(C=C, max_iter=1000)
    model.fit(X_train, y_train)
    
    return dv, model

# Function to make predictions
def predict(df, dv, model):
    # Convert DataFrame to a list of dictionaries
    dicts = df[categorical + numerical].to_dict(orient='records')

    # Transform data and predict probabilities
    X = dv.transform(dicts)
    y_pred = model.predict_proba(X)[:, 1]

    return y_pred

# K-Fold Cross Validation
print(f'Running K-Fold Cross Validation with C = {C} and n_splits = {n_splits}\n')
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1)
scores = []

for train_idx, val_idx in kfold.split(df_full_train):
    df_train = df_full_train.iloc[train_idx]
    df_val = df_full_train.iloc[val_idx]

    y_train = df_train.churn.values
    y_val = df_val.churn.values

    # Train the model
    dv, model = train(df_train, y_train, C=C)

    # Make predictions
    y_pred = predict(df_val, dv, model)

    # Calculate AUC score
    auc = roc_auc_score(y_val, y_pred)
    scores.append(auc)

# Print mean and standard deviation of AUC scores
print(f'C: {C}, Mean AUC on {n_splits} fold is: {np.mean(scores)}, AUC Standard Deviation: {np.std(scores)}\n')
print(f'AUC Scores for each fold: {scores} \n')

# Export the trained model
output_file = f'model_C={C}_splits={n_splits}.bin'
with open(output_file, 'wb') as f_out:
    pickle.dump((dv, model), f_out)
print(f'Model has been saved as: {output_file}\n')
