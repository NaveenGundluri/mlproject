#if i want to save this project into the cloud so that i need to write code here.
#suppose if i want to read a dataset into a database here i want to create sql or moangodb client overhere.
#in the utils code i will try to call it inside the components.
import os
import sys

import numpy as np
import pandas as pd
import dill 
import pickle

from sklearn.model_selection import GridSearchCV
from src.exception import CustomException

from sklearn.metrics import r2_score

def save_object(file_path, obj):
    try:
        dir_path=os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj,file_obj)

    except Exception as e:
        raise CustomException(e,sys)
    
def evaluate_model(X_train,y_train,X_test,y_test,models,param):
    try:
        report={}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para = param.get(list(models.keys())[i], {})  # safely get parameters

            #para=param[list(models.keys())[i]]

            gs = GridSearchCV(model, para, cv=3)
            gs.fit(X_train, y_train)
            
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)
            #model.fit(X_train, y_train) # Train model

            # Make predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
    
            # Evaluate Train and Test dataset
            train_model_score = r2_score(y_train, y_train_pred)

            test_model_score= r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]]=test_model_score

        return report
    except Exception as e:
        raise CustomException(e,sys)
    
#here iam create my own definetion called a load_object. it just doing it will open the filepath and read it by readbyte(( "rb" mode), it will be treated as an uploaded file) mode and it is loading the pkl file using this "dill" .
#inshot this load object is responsible in loading the pkl file.that is reason we have written in utils.py
#reason in the written in utils.py this is common functionality through out the project.
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)

