from flask import Flask,request,render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.CreditCardDefaulters.pipelines.prediction_pipeline import CustomData,PredictPipeline

app=Flask(__name__)


## Route for a home page

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predictdefaulter',methods=['GET','POST'])
def predict_defaulter():
    if request.method=='GET':
        return render_template('home.html')
    else:
        print("printing..")
        print(request.form.get('EDUCATION'))
        data=CustomData(
            ID = (request.form.get('ID')),
            LIMIT_BAL = (request.form.get('LIMIT_BAL')),
            AGE = (request.form.get('AGE')),
            BILL_AMT1 = (request.form.get('BILL_AMT1')),
            BILL_AMT2 = (request.form.get('BILL_AMT2')),
            BILL_AMT3 = (request.form.get('BILL_AMT3')),
            BILL_AMT4 = (request.form.get('BILL_AMT4')),
            BILL_AMT5 = (request.form.get('BILL_AMT5')),
            BILL_AMT6 = (request.form.get('BILL_AMT6')),
            PAY_AMT1 = (request.form.get('PAY_AMT1')),
            PAY_AMT2 = (request.form.get('PAY_AMT2')),
            PAY_AMT3 = (request.form.get('PAY_AMT3')),
            PAY_AMT4 = (request.form.get('PAY_AMT4')),
            PAY_AMT5 = (request.form.get('PAY_AMT5')),
            PAY_AMT6 = (request.form.get('PAY_AMT6')),
            SEX = (request.form.get('SEX')),
            EDUCATION = (request.form.get('EDUCATION')),
            MARRIAGE = (request.form.get('MARRIAGE')),
            PAY_0 = (request.form.get('PAY_0')),
            PAY_2 = (request.form.get('PAY_2')),
            PAY_3 = (request.form.get('PAY_3')),
            PAY_4 = (request.form.get('PAY_4')),
            PAY_5 = (request.form.get('PAY_5')),
            PAY_6 = (request.form.get('PAY_6'))
        )


        pred_df=data.get_data_as_data_frame()
        pred_df.rename(columns={'PAY_0':'PAY_1'}, inplace=True)

        # pred_df["EDUCATION"]=pred_df["EDUCATION"].map({0:4,1:1,2:2,3:3,4:4,5:4,6:4})
        # pred_df["MARRIAGE"]=pred_df["MARRIAGE"].map({0:3,1:1,2:2,3:3})
        pred_df = pred_df.drop(columns="ID",axis=1)
        print(pred_df) 

        predict_pipeline=PredictPipeline()
        result=predict_pipeline.predict(pred_df)
        print("Result..")
        print(result)
        if result!=0.:
            result="The person is Defaulter"
        else:
            result="The person is Not Defaulter"

        return render_template("result.html",final_result=result)
    

if __name__=="__main__":
    app.run(host="0.0.0.0",port=5000)