
import pandas as pd
import numpy as np
import os
import sys
from src.DiamondPricePrediction.logger import logging
from src.DiamondPricePrediction.exception import customexception
from dataclasses import dataclass
from src.DiamondPricePrediction.utils.utils import save_object
from src.DiamondPricePrediction.utils.utils import evaluate_model
from src.DiamondPricePrediction.utils.utils import load_params 

from sklearn.neural_network import MLPRegressor
import yaml


@dataclass 
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')
    
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initate_model_training(self,train_array,test_array):
        try:
            logging.info('Splitting Dependent and Independent variables from train and test data')
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            
            # Load parameters from YAML file
            # Get the directory of the current script
            script_dir = os.path.dirname(os.path.abspath(__file__))
            # Construct the path to the parameters.yaml file
            params_path = os.path.join(script_dir, '..' ,'..', '..', 'parameters.yaml')
            
            # Load parameters from YAML file
            params = load_params(params_path)

            models={
            #'LinearRegression':LinearRegression(),
            #'Lasso':Lasso(),
            #'Ridge':Ridge(),
            #'Elasticnet':ElasticNet(),
            'NeuralNetwork':MLPRegressor(**params)
        }
            
            model_report:dict=evaluate_model(X_train,y_train,X_test,y_test,models)
            print(model_report)
            print('\n====================================================================================\n')
            logging.info(f'Model Report : {model_report}')

            # To get best model score from dictionary 
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            best_model = models[best_model_name]

            print(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')
            print('\n====================================================================================\n')
            logging.info(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')

            save_object(
                 file_path=self.model_trainer_config.trained_model_file_path,
                 obj=best_model
            )
          

        except Exception as e:
            logging.info('Exception occured at Model Training')
            raise customexception(e,sys)

        
    