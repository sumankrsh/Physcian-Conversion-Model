######import packages and dependencies#########

#Data manimulation libraries
import pandas as pd
import numpy as np
import boto3
from io import BytesIO
import seaborn as sns

#ML packages
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc
import xgboost as xgb
from urllib.parse import urlparse
import mlflow
from mlflow.tracking.client import MlflowClient
import joblib

#System and config packages
import sys
import os
import importlib.util #load utils and other scripts
import yaml #read config

#feature store using hopsworks
import hopsworks
from hsml.schema import Schema
from hsml.model_schema import ModelSchema

#warning ignore package
import warnings
warnings.filterwarnings('ignore')

#custom module
from utils import utils

class Trainmodel():
   
   # Load the YAML configuration from the file
    with open('./conf/training_pipeline.yml', 'r') as config_file:
        conf = yaml.safe_load(config_file)

    def __init__(self):
       
        self.bucket_name = self.conf['s3']['bucket_name']
        self.aws_region = self.conf['s3']['aws_region']
        self.api_key = os.environ.get("HOPSWORKS_API_KEY") #load keys from github secrets
        self.project_name = os.environ.get("HOPSWORKS_PROJECT_NAME")
        self.fs_table = self.conf['feature_store']['table_name']

    def model_train(self):
        
        #load model input dataframe and feature list for model build
        input_training_path = self.conf['s3']['df_training_set']
        df_input = utils.load_data_from_s3(self,self.bucket_name, self.aws_region, input_training_path)
        
        df_input = df_input.reset_index()
        df_input.drop(['index'], axis = 1, inplace = True, errors= 'ignore')

        #Clean column names
        df_input.columns = df_input.columns.str.strip()
        df_input.columns = df_input.columns.str.replace(' ', '_')
        df_input.columns = df_input.columns.str.lower()
        
        lookup_key = self.conf['feature_store']['lookup_key']
        utils.convert_columns_to_string(self, df_input, lookup_key)

        # load pickle feature lsit for model training
        file_select_features = self.conf['s3']['model_variable_list_file_path']
        model_features_list = utils.load_pickle_from_s3(self,self.bucket_name,self.aws_region, file_select_features)
        
        #login to hopsworks to load feature group
        project = hopsworks.login(
            api_key_value= self.api_key,
            project= self.project_name,
        )
        # Get features from Hopsworks
        fs = project.get_feature_store()
        features_df = fs.get_feature_group(self.fs_table , version=1)
        df_model = features_df.select(model_features_list).read()

         # Defining the features (X) and the target (y)
        X = df_input.drop("target", axis=1)
        y = df_input["target"]

        # Performing the train-test split to creat training df and inference set
        val_size = self.conf['train_model_parameters']['val_size']
        X_train, X_val, y_train, y_val = train_test_split(X, y,
                                                            test_size=val_size, 
                                                            random_state=42,
                                                                stratify= y)
        
        #logging model in mlflow
        mlflow.xgboost.autolog()
        mlflow.set_experiment(self.conf['mlflow']['experiment_name'])
        with mlflow.start_run():
            drop_id_col_list = self.conf['feature_transformation']['id_col_list']
            best_params = self.conf['train_model_parameters']['model_params']
                        
            # Train the final model with the best hyperparameters
            classifier = xgb.XGBClassifier(**best_params, random_state=42)
            classifier.fit(X_train.drop(drop_id_col_list, axis=1, errors='ignore'), y_train)
            
            # Evaluate the final model on a test dataset (X_val, y_val)
            train_score = classifier.score(X_train.drop(drop_id_col_list, axis=1, errors='ignore'), y_train)
            test_score = classifier.score(X_val.drop(drop_id_col_list, axis=1, errors='ignore'), y_val)
            
            # Log evaluation metric (e.g., accuracy)
            mlflow.log_param("train_data_shape", X_train.shape)
            mlflow.log_param("validation_data_shape", X_val.shape)
            mlflow.log_metric("train_accuracy", train_score)
            mlflow.log_metric("test_accuracy", test_score)
            
            # Log the trained model using MLflow's XGBoost log function
            mlflow.xgboost.log_model(classifier,artifact_path="usecase", registered_model_name="xgboost-model")
            
            #log confusion metrics
            utils.eval_cm(self,classifier, X_train, y_train, X_val,
                                            y_val,drop_id_col_list)
            
            # log roc curve
            utils.roc_curve(self,classifier, 
                            X_val,y_val,drop_id_col_list)
            
            #Log model evaluation metrics
            mlflow.log_metrics(utils.evaluation_metrics(self,
                classifier,
                X_train, y_train, 
                X_val, y_val,
                  drop_id_col_list))
            
            # root_path = "../.."
            #mlflow_path_cm_train = os.path.join("..","..","data","output", "mlflow_exp_logs","confusion_matrix_train.png")
            # mlflow_path_cm_val = os.path.join("..","..","data","output", "mlflow_exp_logs","confusion_matrix_validation.png")
            # mlflow_path_roc = os.path.join(root_path,"data","output", "mlflow_exp_logs","'roc_curve.png'")
            #mlflow.log_artifact("confusion_matrix_train.png")
            # mlflow.log_artifact(mlflow_path_cm_val)
            # mlflow.log_artifact(mlflow_path_roc)
            #print(os.getcwd())
        
        #logging model in Hopsworks
        input_schema = Schema(X_train.values)
        output_schema = Schema(y_train)
        model_schema = ModelSchema(input_schema=input_schema, output_schema=output_schema)
        model_schema.to_dict()

        model_dir= self.conf['feature_store']['model_directory']
        if os.path.isdir(model_dir) == False:
            os.mkdir(model_dir)
        
        joblib.dump(classifier, model_dir + '/xgboost_physician_classifier.pkl')

        # cm = utils.eval_cm(self,classifier, X_train, y_train, X_val,
        #                                     y_val,drop_id_col_list)
        # cm.savefig(model_dir + "/confusion_matrix.png") 

        # roc = utils.roc_curve(self,classifier, 
        #                     X_val,y_val,drop_id_col_list)
        # roc.savefig(model_dir + "/roc_curve.png") 


        mr = project.get_model_registry()

        model = mr.python.create_model(
            name="xgboost_physician_classifier", 
            metrics= utils.evaluation_metrics(self,classifier,X_train, y_train, X_val, y_val,
                  drop_id_col_list),
            model_schema=model_schema,
            input_example=X_train.sample(), 
            description="Physician conversion Predictor")

        model.save(model_dir)
        print('training Pipeline ran successfully')



if __name__ == '__main__':
    task = Trainmodel()
    task.model_train()