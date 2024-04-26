import os
import sys
import mlflow
import pickle
import numpy as np
import mlflow.sklearn
from urllib.parse import urlparse
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from src.DiamondPricePrediction.utils.utils import load_object
from src.DiamondPricePrediction.utils.utils import load_params 
import yaml


class ModelEvaluation:
    def __init__(self):
        pass

    
    def eval_metrics(self,actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))# here is RMSE
        mae = mean_absolute_error(actual, pred)# here is MAE
        r2 = r2_score(actual, pred)# here is r3 value
        return rmse, mae, r2


    def initiate_model_evaluation(self,train_array,test_array):
        try:
            X_test,y_test=(test_array[:,:-1], test_array[:,-1])

            model_path=os.path.join("Artifacts","Model.pkl")

            model=load_object(model_path)
            
            mlflow.set_tracking_uri("http://127.0.0.1:5000")
                        
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
            
            print(tracking_url_type_store)
            
            # Get the directory of the current script
            script_dir = os.path.dirname(os.path.abspath(__file__))
            # Construct the path to the parameters.yaml file
            params_path = os.path.join(script_dir, '..' ,'..', '..', 'parameters.yaml')
            # Load parameters from YAML file
            params = load_params(params_path)

            with mlflow.start_run():

                predicted_qualities = model.predict(X_test)

                (rmse, mae, r2) = self.eval_metrics(y_test, predicted_qualities)

                mlflow.log_metric("rmse", rmse)
                mlflow.log_metric("r2", r2)
                mlflow.log_metric("mae", mae)
                
                # Log parameters (key-value pairs)
                for param_name, param_value in params.items():
                    mlflow.log_param(param_name, param_value)


                # Model registry does not work with file store
                if tracking_url_type_store != "file":

                    # Register the model
                    # There are other ways to use the Model Registry, which depends on the use case,
                    # please refer to the doc for more information:
                    # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                    mlflow.sklearn.log_model(model, "Model", registered_model_name="ml_model")
                else:
                    mlflow.sklearn.log_model(model, "Model")
            
        except Exception as e:
            raise e