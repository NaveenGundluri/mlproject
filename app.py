#now as soon as we create app.py file remember the folder sturcture what are all we did, now this will be making sure that again follow the pipeline structure the moduler programming structure and then we try to develope it.
from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from src.pipeline.predict_pipeline import CustomData,PredictPipeline
from sklearn.preprocessing import StandardScaler #the reason why iam doing this StandardScaler because i need to use it that pkl files.
#now first create a predicton pipeline in the predict_pipeline.py

#create a application name here
#create a Flask app (Flask)
#__name__ which gives you the entry point where we will need to executed
application=Flask(__name__)

#create a variable and assign it to the app
app=application

#route for a homepage
#create ('/') just like a home page
@app.route('/')
#define my index
def index():
    return render_template('index.html') #here just i return render_templete to index.html so done this is my index.

#now test it weather it is working fine 
#before that when we use render_template('index.html') it just go and search for templates folder.so now go and create templete folder.
#inside the templates folder create index.html file. inside a index.html just create welcome message -->welcome to the homepage.

@app.route('/predictdata',methods=['GET','POST']) #create some prediction data and here menthods are only two menthod going to support.
def predict_datapoint():
    #inside this predict_datapoint i will be doing everything probably getting my data and then probably doing my prediction and everything over here.
    if request.method=='GET': #if my request.method=='GET'. if it is GET method i will return render_template and i will return a default 'home.html' page.
        return render_template('home.html') #index.html is a home and this home.html is having inside a file simple input data fields that we really need to provide a model  to do the prediction.
    else:
        #instead 'GET' if it is not 'GET' then it will be 'POST' right
        #pass #right now i will pass here because i will entire thing over here. because in the post part i will be probably capture the data and i will do the standerdscaling and feature scaling and then i will go to the prediction.
        #inside your templete folder create another file that is home.html

        #suppose it is POST request you start creating on a data. for creating a data first of all i will create a my own custom class. now this is where your inside src folder you have pipeline folder inside this we have predict_pipeline. that same function or class will actually be created over here(predict_pipeline).
        
        #Now here only iam going to call the customdata, now import customdata from src.pipeline.predict_pipeline
        data=CustomData(
            gender=request.form.get('gender'), #here iam doing iam reading all the gender by using request.form.get('gender'). when we do the post this request will have the entire information.
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('writing_score')),
            writing_score=float(request.form.get('reading_score'))

        )
        #i probably convert above data into a dataframe
        pred_df=data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        #create predict_pipeline and define it
        predict_pipeline=PredictPipeline()
        print("Mid Prediction")
        results=predict_pipeline.predict(pred_df)
        print("after Prediction")
        return render_template('home.html',results=results[0]) #we are reading this results value in the home.html
    

if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True) #this basically going to map it under 127.0.1 with debug=True(http://127.0.0.1:5000/),(http://127.0.0.1:5000/predictdata)
