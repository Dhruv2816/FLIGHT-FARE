import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler, QuantileTransformer

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(project_root)
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', "preprocessor.pkl")
    column_names_file_path: str = os.path.join('artifacts', "column_names.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation
        '''
        try:
            numerical_columns = ['Duration_in_hours', 'Days_left']
            categorical_columns = ['Journey_day', 'Airline', 'Class', 'Source', 'Departure', 'Total_stops', 
                                   'Arrival', 'Destination', 'On_weekend', 'Daytime_departure', 'Daytime_arrival']

            # Preprocess numerical columns with different scalers
            num_pipeline = Pipeline(steps=[
                ('quant', QuantileTransformer(output_distribution='normal', random_state=0)),
                ('robust', RobustScaler())
            ])

            # Preprocess categorical columns
            cat_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore',sparse_output= False))
            ])
            
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', num_pipeline, numerical_columns),
                    ('cat', cat_pipeline, categorical_columns)
                ],
                remainder='passthrough'
            )

            return preprocessor, numerical_columns, categorical_columns

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info("Read train and test data completed")

            # Drop the 'Flight_code' column if it exists
            if 'Flight_code' in train_df.columns:
                train_df = train_df.drop(columns=['Flight_code'])
            if 'Flight_code' in test_df.columns:
                test_df = test_df.drop(columns=['Flight_code'])

            # Convert to 'category' data type
            categorical_columns = ['Airline', 'Class', 'Source', 'Departure', 'Total_stops', 'Arrival', 'Destination', 'Journey_day']
            train_df[categorical_columns] = train_df[categorical_columns].astype('category')
            test_df[categorical_columns] = test_df[categorical_columns].astype('category')

            # Convert 'Date_of_journey' column to datetime data type and extract month
            train_df['Date_of_journey'] = pd.to_datetime(train_df['Date_of_journey'])
            test_df['Date_of_journey'] = pd.to_datetime(test_df['Date_of_journey'])
            train_df['Journey_month'] = train_df['Date_of_journey'].dt.month
            test_df['Journey_month'] = test_df['Date_of_journey'].dt.month

            # Drop the 'Date_of_journey' column
            train_df = train_df.drop(columns=['Date_of_journey'])
            test_df = test_df.drop(columns=['Date_of_journey'])

            # Replace rare Airline categories with 'Other'
            rare_airlines = ['AkasaAir', 'AllianceAir', 'StarAir', 'SpiceJet']
            train_df['Airline'] = train_df['Airline'].apply(lambda x: 'Other' if x in rare_airlines else x)
            test_df['Airline'] = test_df['Airline'].apply(lambda x: 'Other' if x in rare_airlines else x)

            # Create new column 'On_weekend'
            train_df['On_weekend'] = train_df['Journey_day'].isin(['Saturday', 'Sunday'])
            test_df['On_weekend'] = test_df['Journey_day'].isin(['Saturday', 'Sunday'])

            # Create new columns 'Daytime_departure' and 'Daytime_arrival'
            overnight = ['After 6 PM', 'Before 6 AM']
            train_df['Daytime_departure'] = ~train_df['Departure'].isin(overnight)
            test_df['Daytime_departure'] = ~test_df['Departure'].isin(overnight)
            train_df['Daytime_arrival'] = ~train_df['Arrival'].isin(overnight)
            test_df['Daytime_arrival'] = ~test_df['Arrival'].isin(overnight)

            logging.info("Obtaining preprocessing object")
            preprocessing_obj, numerical_columns, categorical_columns = self.get_data_transformer_object()

            target_column_name = "Fare"

            logging.info("Applying preprocessing object on training and testing dataframes")

            # Apply preprocessing to input features
            input_feature_train_df = train_df.drop(columns=["Fare"])
            input_feature_test_df = test_df.drop(columns=["Fare"])
            target_feature_train_df = train_df["Fare"]
            target_feature_test_df = test_df["Fare"]

            # Apply the preprocessing pipeline to the input features
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # Retrieve one-hot encoded column names
            one_hot_encoded_columns = preprocessing_obj.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_columns).tolist()

            # Combine all column names
            all_column_names = numerical_columns + one_hot_encoded_columns + list(input_feature_train_df.columns.difference(numerical_columns + categorical_columns))

            # Save column names
            save_object(
                file_path=self.data_transformation_config.column_names_file_path,
                obj=all_column_names
            )

            target_feature_train_arr = np.array(target_feature_train_df).reshape(-1, 1)
            target_feature_test_arr = np.array(target_feature_test_df).reshape(-1, 1)
            # Horizontally stack the input features and the reshaped target features
            train_arr = np.hstack((input_feature_train_arr, target_feature_train_arr))
            print(train_arr.shape)
            test_arr = np.hstack((input_feature_test_arr, target_feature_test_arr))
            print(test_arr.shape)
            
            # Save DataFrames to CSV files
            train_file_path = r'C:\Users\dream\OneDrive\Desktop\Flight_Flare_New\SAVE\train_array.csv'
            test_file_path = r'C:\Users\dream\OneDrive\Desktop\Flight_Flare_New\SAVE\test_array.csv'
            pd.DataFrame(train_arr, columns=all_column_names + [target_column_name]).to_csv(train_file_path, index=False)
            pd.DataFrame(test_arr, columns=all_column_names + [target_column_name]).to_csv(test_file_path, index=False)

            logging.info("Saved preprocessing object.")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_df,
                test_df,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e, sys)