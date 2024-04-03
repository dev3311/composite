from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from joblib import dump, load
import seaborn as sns


app = Flask(__name__)

model = load('rock_poly_model.joblib') 


@app.route('/')
def home():
    return render_template('index.html')



@app.route('/predict',methods=['POST'])
def predict():
    
   
    try:

        in_value1 = float(request.form['input_value1'])
       
        
        input_feature = [[in_value1]]

      
        prediction = model.predict(input_feature)
        
        output=round(prediction[0],2)
        
    

        return render_template('index.html', prediction_text = 'Rock density is {}'.format(output))
     
    except Exception as e:
        return render_template('error.html', error=str(e))
    


    
if __name__ == '__main__':
    app.run()    
