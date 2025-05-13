#Note:- this is a regression probleum because the total score is a contineous value.
#Here we are training a model for each and every algorithm we need to check which r2 score will get good accuracy then we will select that model.
import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_model

#for every component we need to create a config file for that every time we will see w.r.to data injection and data transformation initially we are able to define config and we also gave a path for pkl file and data files.

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join('artifacts',"model.pkl")

#which will be responsible for my model training
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            #now create a tupils
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            #now create a dictionary of models
            models={
                "RandomForestRegressor" : RandomForestRegressor(),
                "DecisionTreeRegressor" : DecisionTreeRegressor(),
                "GradientBoostingRegressor" : GradientBoostingRegressor(),
                "LinearRegression" : LinearRegression(),
                "KNeighborsRegressor" : KNeighborsRegressor(),
                "XGBRegressor" : XGBRegressor(),
                "CatBoostRegressor" : CatBoostRegressor(verbose=False),
                "AdaBoostRegressor" : AdaBoostRegressor(),

            }
            params = {
                "DecisionTreeRegressor": {
                    "criterion": ["squared_error", "friedman_mse","absolute_error","poisson"],
                    #"splitter": ["best", "random"]
                    #"max_features":["sqrt","log2"]
                },
                "RandomForestRegressor": {
                    #"criterion": ["squared_error", "friedman_mse","absolute_error","poisson"],
                    #"max_features":["sqrt","log2",None],
                    "n_estimators": [8,16,32,64,128,256]  
                },
                "GradientBoostingRegressor": {
                    #"loss": ["squared_error", "huber","absolute_error","quantile"],
                    #"criterion": ["squared_error", "friedman_mse"],
                    #"max_features":["auto","sqrt",log2"],
                    "n_estimators": [8,16,32,64,128,256],
                    "learning_rate": [.1,.01,.05,.001],
                    "subsample":[0.6,0.7,0.75,0.8,0.85,0.9]
                },
                "LinearRegression": {},  # No hyperparameters to tune
                "KNeighborsRegressor": {
                    "n_neighbors": [5,7,9,11]
                    #"weights":["uniform","distance"],
                    #"algorithm":["ball_tree","kd_tree","brute"]
                },
                "XGBRegressor": {
                    "n_estimators": [8,16,32,64,128,256],
                    "learning_rate": [.1,.01,.05,.001],
                },
                "CatBoostRegressor": {
                    "learning_rate": [0.01,0.05,0.1],
                    "depth": [6,8,10],
                    "iterations":[30,50,100]
                },
                "AdaBoostRegressor": {
                    "n_estimators": [8,16,32,64,128,256],
                    #"loss":["linear","square","exponential"],
                    "learning_rate": [.1,.01,.05,.001]
                }
            }
            model_report:dict=evaluate_model(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models,param=params)

            # to get best model score from dict
            best_model_score=max(sorted(model_report.values()))

            # to get best model name from dict
            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model=models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)

            r2_square=r2_score(y_test,predicted)
            return r2_square
        
        except Exception as e:
            raise CustomException(e,sys)
