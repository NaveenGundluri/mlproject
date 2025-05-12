import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer # this ColumnTransformer use to create the pipeline like onehot encoding, standard scaling...
from sklearn.impute import SimpleImputer # for missing values
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object

@dataclass
class DataTransformationConfig:  # this DataTransformationConfig it will give me any path that it will requiring any inputs, i may probably required from data and transformation component itself.
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl") # pkl : A PKL file is a file saved in the Python pickle format, which contains serialized Python objects. These files are typically used to store machine learning models, data pre-processing objects, or any Python objects that need to be saved for later use.

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
    
    def get_data_transformer_object(self):
        # this function is a responsible for the data transformation based on the different types of the data
        try:
            numerical_columns=['writing_score','reading_score']
            categorical_columns=[
                'gender',
                'race_ethnicity',
                'parental_level_of_education',
                'lunch',
                'test_preparation_course',
            ]

            #Create a numerical pipeline (for handling a missing values)
            #this pipeline is run on the training dataset.fit_transform in the training dataset and we just do transform on the test dataset.
            num_pipeline=Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="median")), #imputer : to handle missing values, median : to handle outliers
                ("scaler",StandardScaler()) # this is basically doing the standerd scaling
                ]
            )

            #Create a categorical pipeline (for handling a  missing values)
            #All categorical features will be converted into numerical features 
            cat_pipeline=Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")), # most_frequent : this is basically means iam trying to replace all the missing values with the help of mode.
                ("one_hot_encoder",OneHotEncoder()),
                ("scaler",StandardScaler(with_mean=False))

                ]

            )
            logging.info(f"Numerical columns : {numerical_columns}")
            logging.info(f"Categorical columns : {categorical_columns}")

            #now this combine numerical pi(peline and categorical pipeline together
            preprocessor=ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,numerical_columns),
                    ("cat_pipeline",cat_pipeline,categorical_columns)
                ]
            )
            return preprocessor #we returning all things which we have done till now 
        
        except Exception as e:
            raise CustomException(e,sys)

    # initiating data transformation    
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")
            logging.info("Obtaining preprocessing object")
            
            #what are you created this get_data_transformer_object we will try to call this
            preprocessing_obj=self.get_data_transformer_object()
            #create target column name
            target_column_name="math_score"
            numerical_columns=['writing_score','reading_score']

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info(f"Applying preprocessing object on training dataframe and testing dataframe")

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_arr=np.c_[
                input_feature_train_arr,np.array(target_feature_train_df)
            ]

            test_arr=np.c_[
                input_feature_test_arr,np.array(target_feature_test_df)
            ]

            logging.info(f"Saved preprocessing object")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            # now where do we write this function save object. we are going to use utils, utils is one place inside the src(utils we have all the common things, that we are going to be import are use)
            # we are saving this pickle name in the harddisk
            )

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            raise CustomException(e,sys)
            