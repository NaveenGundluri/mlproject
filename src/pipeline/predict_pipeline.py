#i need to create a web application which will be interacting with this kind of pkl files w.r.to any input data that we give.
#lets say i our web application we will have a form where will be giving all out input data that is required for predicting thee student performance and then we just click on submit button. internelly that backend capture the data and that data needs to probably interact with this preprocessor.pkl, model.pkl and then we will try to see that how weather we will able to get the prediction.
#go and create app.py file (here we will be using Flask app (Flask Framework))
#now as soon as we create app.py file remember the folder sturcture what are all we did, now this will be making sure that again follow the pipeline structure the moduler programming structure and then we try to develope it.

#now here create a prediction pipeline 
import os
import sys # for my exception handling
import pandas as pd
from src.exception import CustomException
from src.utils import load_object #here i want to import one function is called load_object (just to load a pkl file)

#create predction pipeline class
class PredictPipeline:
    def __init__(self): #initial thing over here __init__ function
        pass

    ##create prediction pipeline here
    #this predict is just like my model prediction. it means basically doing the prediction right. cheack currently we have two pkl files that is model.pkl file and another one is preprocessor.pkl file
    def predict(self,features):
        try:
            model_path='artifacts\model.pkl'
            preprocessor_path='artifacts\preprocessor.pkl'
            print("Before Loading")
            model=load_object(file_path=model_path) #here basically calling load_object function. this lod_object is nothing but import the pkl and probably it just load the pkl file inshot.
            preprocessor=load_object(file_path=preprocessor_path) #once you load this scale your data
            print("After Loading")
            data_scaled=preprocessor.transform(features) #it will transform the features. after transforming the features my model will just do the prediction.
            preds=model.predict(data_scaled) #it will do prediction by using scaled data(data_scaled)
            return preds #returning this prediction
        
        except Exception as e:
            raise CustomException(e,sys)

#create customdata class
#this customdata class will be responsible in mapping all the inputs that we are giving in the html to the backend w.r.to the perticuler  values.
class CustomData:
    def __init__(  self,
        gender: str,
        race_ethnicity: str,
        parental_level_of_education,
        lunch: str,
        test_preparation_course: str,
        reading_score: int,
        writing_score: int):

        #now this definetion hint iam writing it over here . i will be doing creating a variable using self. once you assingn this values, values are basically comes over web application. see home.html i used gender same name getting mapped over there suppose if you select male or female that value will go ahead in the specific variable right. similarly all the things also will do the same actions.
        self.gender = gender

        self.race_ethnicity = race_ethnicity

        self.parental_level_of_education = parental_level_of_education

        self.lunch = lunch

        self.test_preparation_course = test_preparation_course

        self.reading_score = reading_score

        self.writing_score = writing_score

    def get_data_as_data_frame(self): #this get_data_as_data_frame function will do, it will just return all my input as a form of data because we trained our model in the form of a daraframe. now create variable on try ,catch and i will create a dictionary.
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }

            return pd.DataFrame(custom_data_input_dict) #iam returning the  pd.DataFrame as a dataframe
        #inshot what is basically going to happen right.. from my web application whatever my inputs are giving that same inputs will get assign will get mapped with "gender": [self.gender],"race_ethnicity": [self.race_ethnicity],"parental_level_of_education": [self.parental_level_of_education],"lunch": [self.lunch],"test_preparation_course": [self.test_preparation_course],"reading_score": [self.reading_score],"writing_score": [self.writing_score], this perticular value.

        except Exception as e:
            raise CustomException(e, sys)