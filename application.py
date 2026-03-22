from flask import Flask,request,jsonify,render_template
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
application =Flask(__name__)
app=application

## importing model
with open('models/model_ridge.pkl',"rb") as file:
    ridge_model=pickle.load(file)
    
with open('models/model_scaler.pkl',"rb") as file:
    scaler_model=pickle.load(file)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict_data",methods=["GET",'POST'])
def predict_datapoint():
    if request.method=="POST":
       
        Temperature = float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws')) 
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC')) 
        DMC = float(request.form.get('DMC')) 
        DC = float(request.form.get('DC')) 
        ISI = float(request.form.get('ISI')) 
        BUI = float(request.form.get('BUI')) 
        Classes = int(request.form.get('Classes'))
        Region = int(request.form.get('Region'))
        
        new_data=scaler_model.transform([[Temperature,Rain,RH,Ws,FFMC,DMC,DC,ISI,BUI]])
        result=ridge_model.predict(new_data)
        return render_template("home.html",result=result[0])
    else:
        return render_template("home.html")
    


if __name__=="__main__":
    app.run(debug=True)
    
    
    