######import packages and dependencies#########

#Data manimulation libraries
import pandas as pd
import numpy as np
import boto3
from io import BytesIO

#ML packages
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.model_selection import train_test_split

#System and config packages
import sys
import os
import importlib.util #load utils and other scripts
import yaml #read config

#feature store using hopsworks
import hopsworks

#warning ignore package
import warnings
warnings.filterwarnings('ignore')

#custom module
from utils import utils


class DataPrep:

    def __init__(self):

        #Load data from s3
        self.bucket_name = self.conf['s3']['bucket_name']
        self.aws_region = self.conf['s3']['aws_region']

    # Load the YAML configuration from the file
    with open('./conf/feature_pipeline.yml', 'r') as config_file:
        conf = yaml.safe_load(config_file)

    def preprocess_data(self):
        """
        Perform data preprocessing steps as specified in the configuration.

        This method loads data from an S3 bucket, cleans the data, performs one-hot encoding,
        applies feature selection, and saves the preprocessed data back to S3.
        """

        # #Load data from s3
        file_path = self.conf['s3']['file_path']
        df_input = utils.load_data_from_s3(self, self.bucket_name,self.aws_region, file_path)

        #Clean column name and convert to lower case
        df_input.columns = df_input.columns.str.strip()
        df_input.columns = df_input.columns.str.replace(' ', '_')
        df_input.columns = df_input.columns.str.lower()
        
        #Drop unwanted column: "HCO Affiliation" - "Affiliation Type" is more valid column for us
        df_input.drop(self.conf['feature_transformation']['drop_column_list'], axis=1, inplace=True)

        #One hot encode categorical features
        df_input = pd.get_dummies(df_input, columns=self.conf['feature_transformation']['one_hot_encode_feature_list'], drop_first=True)
        df_input.columns = df_input.columns.str.replace(" ", "").str.replace("-", "")
        
        #Select variables for feature selection
        id_target_col_list = self.conf['feature_transformation']['id_target_col_list']
        col_for_feature_selection = df_input.columns.difference(id_target_col_list)
        
        #Variance threshold feature selection method
        var_thr = VarianceThreshold(threshold=self.conf['param_values']['variance_threshold_value'])
        var_thr.fit(df_input[col_for_feature_selection])

        df_input_subset = df_input[col_for_feature_selection]
        remove_col_list = [col for col in df_input_subset.columns if col not in df_input_subset.columns[var_thr.get_support()]]
        
        #remove above list column from master dataframe
        df_input.drop(remove_col_list, axis=1, inplace=True, errors='ignore')
        
        # segregate train and inference set 
        """ 
            Train set to be used furter for model building
            Inference set: As we don't have any Inference data segregating the same from
            entire data
        """
        target = self.conf['feature_transformation']['target_col']
        inference_size = self.conf['param_values']['inference_size']
        X = df_input.drop(target, axis=1)
        y = df_input[target]
        X_train_set, X_inference, y_train_set, y_inference = train_test_split(X, y,
                                                                       test_size=inference_size, 
                                                                       random_state=42,
                                                                         stratify= y)
        
        train_df = pd.concat([X_train_set, y_train_set], axis=1)
        inference_df = pd.concat([X_inference, y_inference], axis=1)
        
        df_feature_store = train_df.copy()
        
        # push data to s3 bucket
        train_path = self.conf['preprocessed']['train_df']
        inference_path = self.conf['preprocessed']['inference_df']
        utils.push_df_to_s3(self,train_df, self.bucket_name, self.aws_region, train_path)
        utils.push_df_to_s3(self,inference_df, self.bucket_name, self.aws_region, inference_path)
        
        
        #Feature Selection Using Select K Best
        id_col_list = self.conf['feature_transformation']['id_col_list']
        n = self.conf['param_values']['select_k_best_feature_num']

        df = df_input.drop(id_col_list, axis=1)
        target_col_var = df_input[target]
        top_n_col_list = utils.select_kbest_features(self,df, target_col_var, n)
        
        #Convert to list
        top_n_col_list = top_n_col_list.tolist()

        # Dump top_n_col_list to s3 bucket
        folder_path = self.conf['preprocessed']['model_variable_list_file_path']
        file_name = self.conf['preprocessed']['model_variable_list_file_name']

        utils.pickle_dump_list_to_s3(self,top_n_col_list, folder_path, file_name, self.bucket_name, self.aws_region)

        #column list for dataframe
        cols_for_model_df_list = id_col_list + top_n_col_list
        df_feature_eng_output = df_input[cols_for_model_df_list]
        # df_model_input = df_feature_eng_output.copy()

        #uploading data to Feature Store
        api_key = os.environ.get("HOPSWORKS_API_KEY") #load keys from github secrets
        project_name = os.environ.get("HOPSWORKS_PROJECT_NAME")
        
        #login to hopsworks
        project = hopsworks.login(
            api_key_value= api_key,
            project= project_name,
        )
        
        fs = project.get_feature_store() #initiate featurestore

        #Create feature group
        physician_conversion_feature_group = fs.get_or_create_feature_group(
            name=self.conf['feature_store']['table_name'],
            version=1,
            description=self.conf['feature_store']['description'],
            primary_key=self.conf['feature_store']['lookup_key']
        )
        
        #insert entire modeling data into featurestore
        col_list = self.conf['feature_store']['lookup_key']
        utils.convert_columns_to_string(self, df_feature_store,col_list)
        utils.convert_columns_to_int(self, df_feature_store, col_list)

        physician_conversion_feature_group.insert(df_feature_store)

        print('Feature Pipeline ran successfully')



if __name__ == '__main__':
    task = DataPrep()
    task.preprocess_data()

