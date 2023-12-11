import os
import sys
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
import numpy as np
import pickle
from src.CreditCardDefaulters.utils.utils import load_object
from sklearn.metrics import accuracy_score



class ModelEvaluation:
    def __init__(self):
        pass

    
    def eval_metrics(self,actual, pred):
        accuracyScore=accuracy_score(actual,pred)
        return accuracyScore


    def initiate_model_evaluation(self,train_array,test_array):
        try:
            X_test,y_test=(test_array[:,:-1], test_array[:,-1])

            model_path=os.path.join("artifacts","model.pkl")
            model=load_object(model_path)

        

            mlflow.set_registry_uri("https://dagshub.com/Pranaliyangandul/CreditCardDefaulters.mlflow")
            
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
            
            print(tracking_url_type_store)



            with mlflow.start_run():

                predicted_qualities = model.predict(X_test)

                score= self.eval_metrics(y_test, predicted_qualities)

                mlflow.log_metric("accuracy score", score)


                # Model registry does not work with file store
                if tracking_url_type_store != "file":

                    # Register the model
                    # There are other ways to use the Model Registry, which depends on the use case,
                    # please refer to the doc for more information:
                    # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                    mlflow.sklearn.log_model(model, "model", registered_model_name="ml_model")
                else:
                    mlflow.sklearn.log_model(model, "model")


                

            
        except Exception as e:
            raise e