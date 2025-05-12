import os
import sys
#import which you already created custom exception
from src.exception import CustomException
#logging for this data injection part
from src.logger import logging

import pandas as pd
from sklearn.model_selection import train_test_split #we need to perfoem train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig
#dataclass : It's most useful when you want to create classes that are primarily used to store data (i.e., data containers), without having to manually write boilerplate code.

#In data injection whenver we are performing the data injection component there should be some inputs that may be required in this data injection component.
#Inputs like : where have to probably save the train data, test data, and raw data these are variables i will be creating in another class called "DataInjection class"

#In my DataInjection component any input that is specifically required i will probably give through this DataInjectionConfig and probably you will do this in datatransformation also 
#Output w.r.to datainjection that will anything like numpy arrey,files,folders and multiple things.

@dataclass #use decorator which is called a dataclass
#inside a class to define a class variable we will use __init__ but if you try this dataclass you will be able to define your class variable.
#The @dataclass decorator in Python (from the dataclasses module) is used to automatically generate special methods for a class, such as:
#__init__() – constructor
#__repr__() – string representation
#__eq__() – equality comparison
#And others like __hash__() or ordering methods if specified
#It's most useful when you want to create classes that are primarily used to store data (i.e., data containers), without having to manually write boilerplate code.
#Without @dataclass, you'd have to write all of that manually.
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts',"train.csv")#here it is string type and we are creating a path which i called artifacts, whichever we will get an output that file will be saved in inside the artifacts folder.
    test_data_path: str=os.path.join('artifacts',"test.csv")
    raw_data_path: str=os.path.join('artifacts',"data.csv")

#if you are only defining variables you just use dataclass, suppose if you have other functions in our class so that you will go ahead with that functions like __init__ ......
class DataInjection:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("entered the daata ingestion method or component")
        try:
            df=pd.read_csv('notebook\data\stud.csv')
            logging.info('Read the dataset as dataframe')
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            logging.info('Train test split initiated')
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            logging.info('Injection of the data is completed')

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e,sys)
           
if __name__=="__main__":
    obj=DataInjection()
    train_data,test_data=obj.initiate_data_ingestion()

    data_transformation=DataTransformation()
    data_transformation.initiate_data_transformation(train_data,test_data)
    



        
       