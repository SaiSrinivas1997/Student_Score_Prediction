import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    pre_processor_obj_filepath = os.path.join("artifacts",'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.DataTransformation_config = DataTransformationConfig()
    def get_data_transformer_obj(self):
        try:

            numerical_features = [ 'reading_score', 'writing_score']
            categorical_features = ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']

            numerical_pipeline = Pipeline(steps = [("Imputer", SimpleImputer(strategy = "median"))])
            logging.info(f"Numerical features: {numerical_features}")

            categorical_pipeline = Pipeline(steps = [("Imputer", SimpleImputer(strategy = "most_frequent")),
                                                     ("one_hot_encoder", OneHotEncoder(handle_unknown = 'ignore'))])
            logging.info(f"Categorical features: {categorical_features}")

            logging.info("Data Transformation Started")
            preprocessor = ColumnTransformer([("numerical_pipeline", numerical_pipeline, numerical_features),
                                              ("categorical_pipeline", categorical_pipeline, categorical_features)])
            logging.info("Data Transformation completed")
            
            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)

    def Initiate_Data_Transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessor object")
            preprocessing_obj = self.get_data_transformer_obj()

            target_column_name = "math_score"

            input_feature_train_df = train_df.drop(columns = [target_column_name], axis = 1)
            target_feature_train_df = train_df[target_column_name]
        
            input_feature_test_df = test_df.drop(columns = [target_column_name], axis = 1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing on training and testing data frame")
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_object(file_path = self.DataTransformation_config.pre_processor_obj_filepath,
                        obj = preprocessing_obj)
            
            logging.info(f"Saved Preprocessing Object")

            return (train_arr, test_arr, self.DataTransformation_config.pre_processor_obj_filepath)
        except Exception as e:
            raise CustomException(e, sys)