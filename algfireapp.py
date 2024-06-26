from flask import Flask
from flask import render_template,request,jsonify

import pickle

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

app=Flask(__name__)

@app.route('/')
def home_page():
    return render_template("home.html")

standard_scaler=pickle.load(open('models/scaler.pkl','rb'))
ridge_model=pickle.load(open('models/ridge.pkl','rb'))

@app.route('/prediction',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='POST':
        Temp=float(request.form['Temperature'])
        rh=float(request.form['RH'])
        ws=float(request.form['Ws'])
        rain=float(request.form['Rain'])
        ffmc=float(request.form['FFMC'])
        dmc=float(request.form['DMC'])
        isi=float(request.form['ISI'])
        classes=float(request.form['Classes'])
        region=float(request.form['Region'])

        data=standard_scaler.transform([[Temp,rh,ws,rain,ffmc,dmc,isi,classes,region]])
        result=ridge_model.predict(data)

        return render_template("home.html",result=result[0])

if __name__==('__main__'):
    app.run("0.0.0.0")
