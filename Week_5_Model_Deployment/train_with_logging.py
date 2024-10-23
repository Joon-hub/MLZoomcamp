import pandas as pd 
import numpy as np
import logging
import argparse
import pickle
import sys
from sklearn.model_selection import train_test_split, KFold
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

def parse_args():
    parser = argparse.ArgumentParser(description="Train a Logistic Regression model for customer churn prediction")
    parser.add_argument("--C", type=float, default=1.0, help="Regularization strength for Logistic Regression (default: 1.0)")
    parser.add_argument("--n_splits", type=int, default=5, help="Number of splits for K-Fold cross-validation (default: 5)")
    return parser.parse_args()

def load_dataset():
    logger.info("Loading dataset...")
    df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
    
    # standardize column names: lowercase and replaces spaces with underscore
    df.columns = df.columns.str.lower().str.replace(' ', '_')

    # preprocess categorical columns
    categorical_cols = df.select_dtypes(include='object').columns
    for col in categorical_cols:
        df[col] = df[col].str.lower().str.replace(' ', '_')

    # convert 'totalcharges' to numeric, handling errors and filling NaNs with 0
    df['totalcharges'] = pd.to_numeric(df['totalcharges'], errors='coerce').fillna(0)

    # map 'churn' values to integers: 'yes' to 1 and 'no' to 0
    df['churn'] = df['churn'].map({'yes': 1, 'no': 0}).fillna(0).astype(int)

    logger.info(f"Dataset has been successfully loaded and preprocessed: {df.shape}")
    return df

def split_dataset(df):
    logger.info("Splitting dataset into training and testing sets...")
    df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
    logger.info(f"Shape of training dataset: {df_full_train.shape}")
    logger.info(f"Shape of testing dataset: {df_test.shape}")
    return df_full_train, df_test

# Define features
categorical = [
    'gender', 'seniorcitizen', 'partner', 'dependents', 'phoneservice',
    'multiplelines', 'internetservice', 'onlinesecurity', 'onlinebackup',
    'deviceprotection', 'techsupport', 'streamingtv', 'streamingmovies',
    'contract', 'paperlessbilling', 'paymentmethod'
]

numerical = ['tenure', 'monthlycharges', 'totalcharges']

def train(df_train, y_train, C=1.0):
    logger.info("Starting model training...")
    # Convert DataFrame to a list of dictionaries
    dicts = df_train[categorical + numerical].to_dict(orient='records')

    # Use DictVectorizer to handle categorical features
    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)

    # Fit the Logistic Regression model
    model = LogisticRegression(C=C, max_iter=1000)
    model.fit(X_train, y_train)
    
    logger.info("Model training completed")
    return dv, model

def predict(df, dv, model):
    dicts = df[categorical + numerical].to_dict(orient='records')
    X = dv.transform(dicts)
    y_pred = model.predict_proba(X)[:, 1]
    return y_pred

def evaluate_model(df_train, y_train, C, n_splits):
    logger.info("Evaluating model with cross-validation...")
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=1)
    scores = []

    for train_idx, val_idx in kf.split(df_train):
        df_fold_train = df_train.iloc[train_idx]
        df_fold_val = df_train.iloc[val_idx]
        
        y_fold_train = y_train.iloc[train_idx]
        y_fold_val = y_train.iloc[val_idx]
        
        dv, model = train(df_fold_train, y_fold_train, C)
        y_pred = predict(df_fold_val, dv, model)
        
        auc = roc_auc_score(y_fold_val, y_pred)
        scores.append(auc)

    mean_score = np.mean(scores)
    std_score = np.std(scores)
    logger.info(f"Cross-validation completed. Mean AUC: {mean_score:.3f} (±{std_score:.3f})")
    return mean_score, std_score

def main():
    args = parse_args()

    # Load and preprocess dataset
    df = load_dataset()

    # Split dataset
    df_full_train, df_test = split_dataset(df)
    y_full_train = df_full_train.churn
    
    # Evaluate model with cross-validation
    mean_score, std_score = evaluate_model(df_full_train, y_full_train, args.C, args.n_splits)

    # Train final model on full training data
    logger.info("Training final model on full training dataset...")
    dv, model = train(df_full_train, y_full_train, args.C)

    # Save the trained model and DictVectorizer
    with open('model.pkl', 'wb') as f:
        pickle.dump((dv, model), f)
    logger.info("Trained model and DictVectorizer have been saved as model.pkl")

if __name__ == "__main__":
    try:
        main()
        logger.info("Training completed successfully!")
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        sys.exit(1)
    sys.exit(0)