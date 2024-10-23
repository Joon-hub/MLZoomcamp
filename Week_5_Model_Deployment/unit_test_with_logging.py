import unittest
import pandas as pd
import numpy as np
import pickle
import logging
from pathlib import Path
import warnings
import sys

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('model_test.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TestChurnPredictor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Load the model and vectorizer once for all tests"""
        try:
            model_path = Path('model_C=1.0.bin')
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found at {model_path}")
            
            with open(model_path, 'rb') as f_in:
                cls.dv, cls.model = pickle.load(f_in)
            logger.info("Model and vectorizer loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def predict_single(self, customer):
        """Helper method to make predictions for a single customer"""
        try:
            X = self.dv.transform([customer])
            y_pred = self.model.predict_proba(X)[0, 1]
            return y_pred
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise

    def test_basic_prediction(self):
        """Test basic prediction functionality"""
        customer = {
            'gender': 'female',
            'seniorcitizen': 0,
            'partner': 'yes',
            'dependents': 'no',
            'phoneservice': 'no',
            'multiplelines': 'no_phone_service',
            'internetservice': 'dsl',
            'onlinesecurity': 'no',
            'onlinebackup': 'yes',
            'deviceprotection': 'no',
            'techsupport': 'no',
            'streamingtv': 'no',
            'streamingmovies': 'no',
            'contract': 'month-to-month',
            'paperlessbilling': 'yes',
            'paymentmethod': 'electronic_check',
            'tenure': 1,
            'monthlycharges': 29.85,
            'totalcharges': 29.85
        }
        
        prediction = self.predict_single(customer)
        self.assertIsInstance(prediction, float)
        self.assertTrue(0 <= prediction <= 1)
        logger.info(f"Basic prediction test passed. Prediction: {prediction:.4f}")

    def test_missing_features(self):
        """Test model behavior with missing features"""
        customer_missing_features = {
            'gender': 'female',
            'seniorcitizen': 0,
            'partner': 'yes',
            # Missing some features intentionally
            'internetservice': 'dsl',
            'contract': 'month-to-month',
            'tenure': 1,
            'monthlycharges': 29.85,
            'totalcharges': 29.85
        }
        
        with self.assertRaises(Exception):
            self.predict_single(customer_missing_features)
        logger.info("Missing features test passed")

    def test_invalid_values(self):
        """Test model behavior with invalid categorical values"""
        customer_invalid = {
            'gender': 'INVALID',
            'seniorcitizen': 0,
            'partner': 'yes',
            'dependents': 'no',
            'phoneservice': 'no',
            'multiplelines': 'no_phone_service',
            'internetservice': 'dsl',
            'onlinesecurity': 'no',
            'onlinebackup': 'yes',
            'deviceprotection': 'no',
            'techsupport': 'no',
            'streamingtv': 'no',
            'streamingmovies': 'no',
            'contract': 'month-to-month',
            'paperlessbilling': 'yes',
            'paymentmethod': 'electronic_check',
            'tenure': 1,
            'monthlycharges': 29.85,
            'totalcharges': 29.85
        }
        
        prediction = self.predict_single(customer_invalid)
        self.assertIsInstance(prediction, float)
        self.assertTrue(0 <= prediction <= 1)
        logger.info(f"Invalid values test passed. Prediction: {prediction:.4f}")

    def test_numerical_boundaries(self):
        """Test model behavior with extreme numerical values"""
        customer_extreme = {
            'gender': 'female',
            'seniorcitizen': 0,
            'partner': 'yes',
            'dependents': 'no',
            'phoneservice': 'no',
            'multiplelines': 'no_phone_service',
            'internetservice': 'dsl',
            'onlinesecurity': 'no',
            'onlinebackup': 'yes',
            'deviceprotection': 'no',
            'techsupport': 'no',
            'streamingtv': 'no',
            'streamingmovies': 'no',
            'contract': 'month-to-month',
            'paperlessbilling': 'yes',
            'paymentmethod': 'electronic_check',
            'tenure': 1000,  # Extreme value
            'monthlycharges': 9999.99,  # Extreme value
            'totalcharges': 999999.99  # Extreme value
        }
        
        prediction = self.predict_single(customer_extreme)
        self.assertIsInstance(prediction, float)
        self.assertTrue(0 <= prediction <= 1)
        logger.info(f"Numerical boundaries test passed. Prediction: {prediction:.4f}")

    def test_batch_prediction(self):
        """Test model behavior with batch predictions"""
        customers = [
            {
                'gender': 'female',
                'seniorcitizen': 0,
                'partner': 'yes',
                'dependents': 'no',
                'phoneservice': 'no',
                'multiplelines': 'no_phone_service',
                'internetservice': 'dsl',
                'onlinesecurity': 'no',
                'onlinebackup': 'yes',
                'deviceprotection': 'no',
                'techsupport': 'no',
                'streamingtv': 'no',
                'streamingmovies': 'no',
                'contract': 'month-to-month',
                'paperlessbilling': 'yes',
                'paymentmethod': 'electronic_check',
                'tenure': 1,
                'monthlycharges': 29.85,
                'totalcharges': 29.85
            }
        ] * 3  # Create 3 identical customers for batch testing
        
        X = self.dv.transform(customers)
        y_pred = self.model.predict_proba(X)[:, 1]
        
        self.assertEqual(len(y_pred), 3)
        self.assertTrue(all(0 <= p <= 1 for p in y_pred))
        logger.info("Batch prediction test passed")

def run_tests():
    """Run all tests and generate a report"""
    logger.info("Starting model tests...")
    
    # Create a test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestChurnPredictor)
    
    # Run the tests and capture the results
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Log the results
    logger.info(f"""
    Test Results:
    - Tests run: {result.testsRun}
    - Failures: {len(result.failures)}
    - Errors: {len(result.errors)}
    - Skipped: {len(result.skipped)}
    """)
    
    return result.wasSuccessful()

if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)