import yaml
import os
import pandas as pd
from logging import Logger
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OrdinalEncoder

class TrainingPreprocessing:

    """
    Description: This class contains the methods used for the preprocessing of the training data

    Written By: Shivam Shinde

    Version: 1.0

    Revision: None
    """

    def __init__(self):
        self.params = yaml.safe_load(
        open(os.path.join("config", "params.yaml")))
    

    # def gettingDependentAndIndependentFeatures(self):

    #     """
    #     Description: This method is used to get the dependent and independent features 

    #     Written By: Shivam Shinde

    #     Version: 1.0

    #     Revision: None
    #     """

    #     path_of_fileFromDb = self.params['data_preparation']['Training_FileFromDB']
    #     filename = self.params['data_preparation']['master_csv']

    #     try:
    #         df = pd.read_csv(os.path.join(path_of_fileFromDb,filename))

    #         X = df.drop(columns=['class'])
    #         y = df['class']

    #         return X, y
        
    #     except Exception as e:
    #         raise e


    
    def replaceQuestionMark(self):


        """
        Description: This method is used to replace the unique value ? from the feature column named stalk_root with one of its other unique values

        Written By: Shivam Shinde

        Version: 1.0

        Revision: None
        """

        path_for_log = self.params['TrainingLogs']['preprocessingLogs']

        if not os.path.exists(path_for_log):
            os.makedirs(path_for_log)
        
        f = open(path_for_log,"a+")

        path_of_fileFromDb = self.params['data_preparation']['Training_FileFromDB']
        filename = self.params['data_preparation']['master_csv']

        try:
            # reading a file extracted from the database
            df = pd.read_csv(os.path.join(path_of_fileFromDb,filename))

            df[df['stalk-root'] == "?"]["stalk-root"] = "b"
            Logger().log(f, f"The ? value from the feature column named stalk-root replaced with 'b' successfully!")

        except Exception as e:
            Logger().log(f, f"Exception occured while replaceing ? value from the feature column named stalk-root 'b'. Exception: {str(e)}")
            pass


    def transformPipeline(self):

        """
        Description: This method is used to impute the missing values as well as to encode the categorical feature column values using scikit-learn ordinal-encoder

        Written By: Shivam Shinde

        Version: 1.0

        Revision: None
        """

        path_for_log = self.params['TrainingLogs']['preprocessingLogs']

        if not os.path.exists(path_for_log):
            os.makedirs(path_for_log)
        
        f = open(path_for_log,"a+")

        path_of_fileFromDb = self.params['data_preparation']['Training_FileFromDB']
        filename = self.params['data_preparation']['master_csv']

        
        try:
            # getting the dataframe
            df = pd.read_csv(os.path.join(path_of_fileFromDb,filename))

            # parameters for the knn imputer
            n_neighbors = self.params['data_preprocessing']['KNNImputer']['n_neighbors']
            weights = self.params['data_preprocessing']['KNNImputer']['weights']
            missing_values = self.params['data_preprocessing']['KNNImputer']['missing_values']


            # parameters for the ordianl encoding 
            handle_unknown = self.params['data_preprocessing']['OrdinalEncoder']['handle_unknown']
            unknown_value = self.params['data_preprocessing']['OrdinalEncoder']['unknown_value']

            # creating pipeline
            pipeline = Pipeline(
                ('imputor',KNNImputer(n_neighbors=n_neighbors, weights=weights, missing_values=missing_values)),
                ('encoder',OrdinalEncoder(handle_unknown=handle_unknown, unknown_value=unknown_value))
            )

            # transforming data using created pipeline
            df_transformed = pipeline.fit_transform(df)

            # creating a dataframe out of the array 
            df_transformed = pd.DataFrame(df_transformed, columns=df.columns)

            transformed_data_path = self.params['data_preprocessing']['preprocessed_data_dir']
            transformed_data_filename = self.params['data_preprocessing']['preprocessed_data_filename']

            # writing a transformed data to a csv file
            df_transformed.to_csv(os.path.join(transformed_data_path,transformed_data_filename), header=True, index=False)

            Logger().log(f, "Processing of the data completed successfully!")
        
        except Exception as e:
            Logger().log(f, f"Exception occurred while preprocessing a training data. Exception: {str(e)}")
            raise e



            


