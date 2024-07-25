import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
import pickle
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRegressor

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(project_root)
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, load_object

@dataclass
class ModelTrainerConfig:
    model_path: str = os.path.join('artifacts', 'xgboost_model.pkl')
    preprocessor_path: str = os.path.join('artifacts', 'preprocessor.pkl')

class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Loading preprocessor")
            preprocessor = load_object(self.config.preprocessor_path)
            train_array = train_array.sample(5000, random_state = 42)
            test_array = test_array.sample(2000, random_state = 42)
            logging.info(f"train_array shape: {train_array.shape}")
            logging.info(f"test_array shape: {test_array.shape}")

            # Ensure the arrays are in the expected shape
            if len(train_array.shape) != 2 or len(test_array.shape) != 2:
                raise CustomException("Train and test arrays must be 2-dimensional", sys)

            # Split train and test arrays into features and target
            X_train = train_array.drop(columns=['Fare'])
            y_train = train_array['Fare']
            X_test = test_array.drop(columns=['Fare'])
            y_test = test_array['Fare']
            
            # Define parameter grid
            param_grid = {
                'regressor__n_estimators': [300, 400, 500],
                'regressor__learning_rate': [0.5],
                'regressor__max_depth': [10, 12],
                'regressor__subsample': [0.8, 0.9],
                'regressor__colsample_bytree': [0.8, 0.9],
                'regressor__min_child_weight': [15, 20],
                'regressor__reg_alpha': [0, 0.1, 0.5],
                'regressor__reg_lambda': [0, 0.1, 0.5],
            }

            # Define XGBoost regressor
            xgb_regressor = XGBRegressor(tree_method = 'gpu_hist', gpu_id = 0, random_state=73)

            # Create pipeline for XGBoost
            xgb_pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('regressor', xgb_regressor)
            ])

            # Define and fit RandomizedSearchCV object
            logging.info("Starting RandomizedSearchCV")
            xgb_search = RandomizedSearchCV(xgb_pipeline, n_iter=20, cv=5, scoring='neg_mean_squared_error', 
                                            random_state=42, n_jobs=-1, error_score='raise', param_distributions=param_grid)
            xgb_search.fit(X_train, y_train)

            # Get the best score and parameters
            best_score = np.sqrt(-xgb_search.best_score_)
            best_params_ = xgb_search.best_params_
            logging.info(f"XGBoost Best R^2 Score: {best_score}")
            logging.info(f"Best parameters for XGBoost: {best_params_}")

            # Retrieve the best model and predict on test data
            best_model_xgb = xgb_search.best_estimator_
            y_pred_test = best_model_xgb.predict(X_test)

            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred_test)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred_test)
            mae = mean_absolute_error(y_test, y_pred_test)
            logging.info(f"Mean Absolute Error: {mae}")
            logging.info(f"Mean Squared Error: {mse}")
            logging.info(f"Root Mean Squared Error: {rmse}")
            logging.info(f"R^2 Score: {r2}")

            # Get feature importances
            X_train_transformed = preprocessor.fit_transform(X_train)
            importances = best_model_xgb.named_steps['regressor'].feature_importances_
            feature_names = preprocessor.get_feature_names_out()
            feature_importances = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
            feature_importances = feature_importances.sort_values(by='Importance', ascending=False)
            logging.info("Feature Importances:")
            logging.info(feature_importances)

            # Plot actual vs predicted fares
            plt.figure(figsize=(10, 6))
            sns.scatterplot(x=y_test, y=y_pred_test, color='skyblue', alpha=0.5, s=50)
            plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='navy', linestyle='--')
            plt.xlabel("Actual Fare")
            plt.ylabel("Predicted Fare")
            plt.title("Actual vs. Predicted Fares")
            plt.show()

            # Save the pipeline to a file
            joblib.dump(xgb_search, self.config.model_path)
            logging.info(f"Model and preprocessor saved to {self.config.model_path}")

            return best_model_xgb

        except Exception as e:
            raise CustomException(e, sys)