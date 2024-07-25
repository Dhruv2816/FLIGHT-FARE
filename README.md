# **FLIGHT FARE PROJECT**

### **Components:**
- **data_ingestion.py**: Handles data loading and initial preprocessing steps.
- **data_transformation.py**: Responsible for transforming the data, including scaling and encoding categorical variables.
- **model_trainer.py**: Trains the model using XGBoost and performs hyperparameter tuning using RandomizedSearchCV.
- **exception.py**: Custom exception handling.
- **logger.py**: Configures logging for the project.
- **utils.py**: Utility functions, including saving and loading objects.

### **Technologies Used:**
- **Programming Language**: Python 3.7 or above
- **Data Analysis and Manipulation**: Pandas, NumPy
- **Machine Learning**: Scikit-learn, XGBoost
- **Hyperparameter Tuning**: RandomizedSearchCV
- **Data Preprocessing**: Scikit-learn transformers and pipelines
- **Logging**: Python's logging module
- **Serialization**: Joblib

### **Models Used:**
- **XGBoost**: An efficient and scalable implementation of the gradient boosting framework by Friedman, widely used by data scientists and Kaggle competition winners.

## Data Preprocessing and Feature Engineering

1. **Numerical Features**:
   - `Duration_in_hours`
   - `Days_left`
   - Scaling using `QuantileTransformer` and `RobustScaler`.

2. **Categorical Features**:
   - `Journey_day`, `Airline`, `Class`, `Source`, `Departure`, `Total_stops`, `Arrival`, `Destination`, `On_weekend`, `Daytime_departure`, `Daytime_arrival`
   - Encoding using `OneHotEncoder` with `handle_unknown='ignore'`.

3. **Feature Engineering**:
   - Conversion of `Date_of_journey` to datetime and extraction of `Journey_month`.
   - Creation of new features `On_weekend`, `Daytime_departure`, and `Daytime_arrival`.
   - Handling rare categories in `Airline` by grouping them into 'Other'.

## Getting Started

### Prerequisites

- Python 3.7 or above
- Required libraries listed in `requirements.txt`

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/flight-fare-prediction.git
   cd flight-fare-prediction
2. Create a virtual environment and activate it:
python -m venv venv
venv\Scripts\activate
3. Install the required packages:
pip install -r requirements.txt

### Running the Project
1. Place your training and testing data in the data/ directory as train.csv and test.csv.
2. Run the data ingestion, transformation, and model training scripts in sequence:
`python src/components/data_ingestion.py`
`python src/components/data_transformation.py`
`python src/components/model_trainer.py `
3. The preprocessor and model objects will be saved in the artifacts/ directory.



### **Results:**

- **Best parameters for XGBoost:** 
  - {'regressor__subsample': 0.9, 'regressor__reg_lambda': 0.5, 'regressor__reg_alpha': 0, 'regressor__n_estimators': 300, 'regressor__min_child_weight': 20, 'regressor__max_depth': 10, 'regressor__learning_rate': 0.5, 'regressor__colsample_bytree': 0.9}

- **Mean Absolute Error:** 4377.192530273437
- **Mean Squared Error:** 47889288.83133516
- **Root Mean Squared Error:** 6920.2087274398855
- **R² Score:** 0.8793434500694275

  
