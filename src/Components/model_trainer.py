import os
import sys
from dataclasses import dataclass

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import (AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor)

from sklearn.metrics import r2_score

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object
from src.utils import evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            logging.info("Split trian and test input data")
            x_train, y_train, x_test, y_test = (train_arr[:,:-1],
                                                train_arr[:,-1],
                                                test_arr[:,:-1],
                                                test_arr[:,-1])
            
            models = {"Random Forest":RandomForestRegressor(),
                      "Decision Tree": DecisionTreeRegressor(),
                      "Gradient Boosting": GradientBoostingRegressor(),
                      "Linear Regression": LinearRegression(),
                      "K-Neighbors Regressor": KNeighborsRegressor(),
                      "XGB Regressor": XGBRegressor(),
                      "Cat Boosting Regressor": CatBoostRegressor(),
                      "Ada Boost Regressor": AdaBoostRegressor()}
            
            model_report:dict = evaluate_model(x_train,
                                               x_test,
                                               y_train,
                                               y_test,
                                               models)
            
            # to get best model score from dict 
            best_model_score = max(sorted(model_report.values()))
            # to get best model name from dict
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            
            best_model = models[best_model_name]
            if best_model_score<0.6:
                raise CustomException("No best model found")
            
            logging.info("Best found model on training and testing dataset")
            save_object(file_path = self.model_trainer_config.trained_model_file_path,
                        obj = best_model)
            
            y_pred = best_model.predict(x_test)
            r2_square = r2_score(y_test, y_pred)
            return r2_square
        except Exception as e:
            raise CustomException(e, sys)