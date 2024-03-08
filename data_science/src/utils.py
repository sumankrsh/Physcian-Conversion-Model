######import packages and dependencies#########

#Data manimulation libraries
import pandas as pd
import numpy as np
import boto3
from io import BytesIO

#ML packages
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, f1_score, accuracy_score
import pickle

#System and config packages
import sys
import os
import importlib.util #load utils and other scripts
import yaml #read config
import urllib

#fVizualization and charts
import matplotlib.pyplot as plt
import seaborn as sns

#warning ignore package
import warnings
warnings.filterwarnings('ignore')


class utils:
    def push_df_to_s3(self, df, bucket_name, aws_region, s3_object_key):
        """
        Push a DataFrame to an S3 bucket.

        Args:
            df (pd.DataFrame): The DataFrame to push to S3.
            bucket_name (str): The name of the S3 bucket.
            aws_region (str): The AWS region of the S3 bucket.
            s3_object_key (str): The key of the S3 object.

        Returns:
            #push the desired dataframe to s3 bucket

        Example:
            To push a DataFrame 'my_df' to an S3 bucket 'my-bucket' in the 'us-west-2' region
            with the object key 'data/my_data.csv', you can call the function as follows:
            
            >>> push_df_to_s3(my_df, 'my-bucket', 'us-west-2', 'data/my_data.csv')

        Note:
            - Ensure that AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY are set in your environment
              variables or through other means to authenticate with AWS.
            - The DataFrame will be converted to CSV format before being pushed to S3.
        """
        aws_access_key = os.environ.get("AWS_ACCESS_KEY_ID")
        aws_secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")

        csv_buffer = BytesIO()
        df.to_csv(csv_buffer, index=False)
        csv_content = csv_buffer.getvalue()

        s3 = boto3.resource("s3", aws_access_key_id=aws_access_key, aws_secret_access_key=aws_secret_key, region_name=aws_region)

        s3.Object(bucket_name, s3_object_key).put(Body=csv_content)

        return {"df_push_status": 'success'}
    

    def load_data_from_s3(self, bucket_name, aws_region, file_path):
        """
        Load data from an S3 bucket.

        Args:
            bucket_name (str): The name of the S3 bucket.
            aws_region (str): The AWS region of the S3 bucket.
            file_path (str): The path to the S3 file.

        Returns:
            pd.DataFrame: The loaded DataFrame.

        Example:
            To load data from an S3 bucket named 'my-bucket' in the 'us-west-2' region with a file path
            'data/my_data.csv', you can call the function as follows:

            >>> loaded_data = load_data_from_s3('my-bucket', 'us-west-2', 'data/my_data.csv')

        Note:
            - Ensure that AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY are set in your environment
              variables or through other means to authenticate with AWS.
            - The function assumes that the S3 object contains CSV data.
        """
        aws_access_key = os.environ.get("AWS_ACCESS_KEY_ID")
        aws_secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")

        s3 = boto3.resource("s3", aws_access_key_id=aws_access_key, aws_secret_access_key=aws_secret_key, region_name=aws_region)

        s3_object = s3.Object(bucket_name, file_path)
        csv_content = s3_object.get()['Body'].read()

        df_input = pd.read_csv(BytesIO(csv_content))

        return df_input
    

    def select_kbest_features(self, df, target_col, n):
        """
        Select the top n features from a DataFrame using the SelectKBest algorithm.

        Args:
            df (pd.DataFrame): The DataFrame to select features from.
            target_col (pd.Series): The target column for feature selection.
            n (int): The number of features to select.

        Returns:
            pd.Index: A list of the top n features.
        """
        # Initialize the SelectKBest selector with the specified score function and k value
        selector = SelectKBest(score_func=f_classif, k=n)

        # Fit the selector to the DataFrame and target column
        selected_features = selector.fit(df, target_col)

        # Get the mask of selected features
        mask = selector.get_support()

        # Get the names of the top n features
        top_n_features = df.columns[mask]

        return top_n_features
    
    def pickle_dump_list_to_s3(self, column_list, folder_path, file_name, bucket_name, aws_region):
        """
        Pickle dump a list of columns and upload it to an S3 bucket.

        Args:
            column_list (list): List of columns to pickle.
            folder_path (str): The path within the S3 bucket where the file will be stored.
            file_name (str): The name of the pickled file.
            bucket_name (str): The name of the S3 bucket.
            aws_region (str): The AWS region of the S3 bucket.
        """
        # Get AWS credentials from environment variables
        aws_access_key = os.environ.get("AWS_ACCESS_KEY_ID")
        aws_secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")

        # Pickle dump the list to a local file
        with open(file_name, 'wb') as file:
            pickle.dump(column_list, file)

        # Upload the pickled file to S3
        s3 = boto3.resource("s3", aws_access_key_id=aws_access_key, aws_secret_access_key=aws_secret_key, region_name=aws_region)
        s3.Bucket(bucket_name).upload_file(file_name, os.path.join(folder_path, file_name))


    def load_pickle_from_s3(self, bucket_name, aws_region, file_path):
        """
        Load data from an S3 bucket by deserializing a pickled file.

        Args:
            bucket_name (str): The name of the S3 bucket.
            aws_region (str): The AWS region of the S3 bucket.
            file_path (str): The path to the pickled file in the S3 bucket.

        Returns:
            Any: The Python object deserialized from the pickled file.
        """
        try:
            # Get AWS credentials from environment variables
            aws_access_key = os.environ.get("AWS_ACCESS_KEY_ID")
            aws_secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")

            # Create an S3 resource
            s3 = boto3.resource("s3", aws_access_key_id=aws_access_key, aws_secret_access_key=aws_secret_key, region_name=aws_region)

            # Get the S3 object
            s3_object = s3.Object(bucket_name, file_path)

            # Read the pickled file from the S3 response
            pickle_data = s3_object.get()['Body'].read()

            # Deserialize the pickle data to obtain the Python object
            loaded_object = pickle.loads(pickle_data)

            return loaded_object
        except Exception as e:
            print(f"Error: {str(e)}")
            return None
        
    
    def convert_columns_to_string(self, df, columns):
        """
        Convert specified columns in a DataFrame to string data type.

        Args:
            df (pd.DataFrame): The DataFrame containing the columns to be converted.
            columns (list): List of column names to convert to string.

        Returns:
            None
        """
        for col in columns:
            if col in df.columns:
                df[col] = df[col].astype(str)
            else:
                print(f"Column '{col}' not found in the DataFrame.")

    
    def convert_columns_to_int(self, df, columns_to_remove=[]):
        """
        Convert specified columns in a DataFrame to integer type after removing specified columns.

        Args:
            df (pd.DataFrame): The DataFrame containing the columns to be converted.
            columns_to_remove (list): A list of column names to be removed before conversion.

        Returns:
            pd.DataFrame: The DataFrame with specified columns converted to integer.

        Example:
        >>> df = pd.DataFrame({'A': ['1', '2', '3'], 'B': ['4', '5', '6'], 'C': ['7', '8', '9']})
        >>> columns_to_remove = ['C']
        >>> columns_to_convert = [col for col in df.columns if col not in columns_to_remove]
        >>> converted_df = convert_columns_to_int(df, columns_to_remove)
        >>> print(converted_df.dtypes)
        A    int64
        B    int64
        dtype: object
        """
        columns_to_convert = [col for col in df.columns if col not in columns_to_remove]
        
        for col in columns_to_convert:
            if col in df.columns:
                df[col] = df[col].astype(int)
            else:
                print(f"Column '{col}' not found in the DataFrame.")


    
    def eval_cm(self, model, X_train, y_train, X_val, y_val, drop_id_col_list):
        """
        Evaluate a classification model by plotting confusion matrices for both the training and validation sets.

        Args:
            model: The classification model to evaluate.
            X_train: The training data features.
            y_train: The training data labels.
            X_val: The validation data features.
            y_val: The validation data labels.
            drop_id_col_list: List of columns to drop before prediction.

        Returns:
            None
        """
        # Fit the model to the training data
        model.fit(X_train.drop(drop_id_col_list, axis=1, errors='ignore'), y_train)

        # Predict labels for the training and validation sets
        y_pred_train = model.predict(X_train.drop(drop_id_col_list, axis=1, errors='ignore'))
        y_pred_val = model.predict(X_val.drop(drop_id_col_list, axis=1, errors='ignore'))

        # Create and display confusion matrices
        plt.figure(figsize=(8, 6))
        
        # Confusion Matrix for Training Set
        cm_train = confusion_matrix(y_train, y_pred_train)
        plt.subplot(1, 2, 1)
        sns.heatmap(cm_train, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title('Confusion Matrix (Train)')
        
        # Confusion Matrix for Validation Set
        cm_val = confusion_matrix(y_val, y_pred_val)
        plt.subplot(1, 2, 2)
        sns.heatmap(cm_val, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title('Confusion Matrix (Validation)')
        
        # Save the confusion matrix plots
        plt.savefig('confusion_matrix.png')


    def roc_curve(self, model, X_val, y_val, drop_id_col_list):
        """
        Generate and save a ROC curve plot for a classification model.

        Args:
            model: The classification model for which to generate the ROC curve.
            X_val: The validation data features.
            y_val: The validation data labels.
            drop_id_col_list: List of columns to drop before prediction.

        Returns:
            None
        """
        # Predict probabilities for positive class
        y_pred = model.predict_proba(X_val.drop(drop_id_col_list, axis=1, errors='ignore'))[:, 1]

        # Calculate ROC curve values
        fpr, tpr, thresholds = roc_curve(y_val, y_pred)

        # Calculate AUC-ROC score
        roc_auc = roc_auc_score(y_val, y_pred)

        # Create and save the ROC curve plot
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        roc_curve_plot_path = "roc_curve.png"
        
        # Save the ROC curve plot
        plt.savefig(roc_curve_plot_path)


    def evaluation_metrics(self, model, X_train, y_train, X_val, y_val, drop_id_col_list):
        """
        Calculate and log F1-score and accuracy metrics for model evaluation.

        Args:
            model: The classification model to evaluate.
            X_train: The training data features.
            y_train: The training data labels.
            X_val: The validation data features.
            y_val: The validation data labels.
            drop_id_col_list: List of columns to drop before prediction.

        Returns:
            dict: A dictionary containing the training and validation F1-scores and accuracies.
        """
        # Fit the model on training data
        model.fit(X_train.drop(drop_id_col_list, axis=1, errors='ignore'), y_train)

        # Predict labels for training and validation data
        y_pred_train = model.predict(X_train.drop(drop_id_col_list, axis=1, errors='ignore'))
        y_pred_val = model.predict(X_val.drop(drop_id_col_list, axis=1, errors='ignore'))

        # Calculate F1-score and accuracy for training data
        f1_train = f1_score(y_train, y_pred_train)
        accuracy_train = accuracy_score(y_train, y_pred_train)

        # Calculate F1-score and accuracy for validation data
        f1_val = f1_score(y_val, y_pred_val)
        accuracy_val = accuracy_score(y_val, y_pred_val)

        # Return the metrics as a dictionary
        return {
            'Train_F1-score': round(f1_train, 2),
            'Validation_F1-score': round(f1_val, 2),
            'Train_Accuracy': round(accuracy_train, 2),
            'Validation_Accuracy': round(accuracy_val, 2)
        }
    

    def load_module(self, file_name, module_name):
        """
        Load a Python module from a specified file and return the loaded module.

        Args:
            file_name (str): The path to the Python file to load as a module.
            module_name (str): The name to assign to the loaded module.

        Returns:
            module: The loaded Python module.
        """
        # Create a module specification from the file location
        spec = importlib.util.spec_from_file_location(module_name, file_name)
        
        # Create an empty module with the specified name
        module = importlib.util.module_from_spec(spec)
        
        # Add the module to the system modules
        sys.modules[module_name] = module
        
        # Execute the module's code and populate it
        spec.loader.exec_module(module)
        
        # Return the loaded module
        return module
