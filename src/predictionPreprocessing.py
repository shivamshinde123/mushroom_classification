import yaml
import os
import pandas as pd
from performLogging import Logger
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
import pathlib
import pickle

class PredictionPreprocessing:

    """
    Description: This class contains the methods used for the preprocessing of the prediction data

    Written By: Shivam Shinde

    Version: 1.0

    Revision: None
    """

    def __init__(self):
        pass

    # def gettingDependentAndIndependentFeatures(self):

    #     """
    #     Description: This method is used to get the dependent and independent features 

    #     Written By: Shivam Shinde

    #     Version: 1.0

    #     Revision: None
    #     """

    #     with open(os.path.join("config", "params.yaml")) as p:
    #         params = yaml.safe_load(p)

    #     main_log_dir = params['PredictionLogs']['main_log_dir']
    #     filename = params['PredictionLogs']['preprocessingLogs']

    #     if not os.path.exists(main_log_dir):
    #         os.makedirs(main_log_dir)

    #     whole_path = os.path.join(main_log_dir, filename)
    #     whole_path = pathlib.Path(whole_path)

    #     f = open(whole_path, 'a+')

    #     try:
    #         path_of_fileFromDb = params['pred_data_preparation']['Prediction_FileFromDB']
    #         filename = params['pred_data_preparation']['master_csv']

    #         df = pd.read_csv(os.path.join(path_of_fileFromDb,filename))

    #         X = df.drop(columns=['class'])
    #         y = df['class']

    #         return X, y
        
    #     except Exception as e:
    #         raise e
        
    #     finally:
    #         f.close()


    
    def replaceQuestionMark(self):


        """
        Description: This method is used to replace the unique value ? from the feature column named stalk_root with one of its other unique values

        Written By: Shivam Shinde

        Version: 1.0

        Revision: None
        """

        with open(os.path.join("config", "params.yaml")) as p:
            params = yaml.safe_load(p)

        main_log_dir = params['PredictionLogs']['main_log_dir']
        filename = params['PredictionLogs']['preprocessingLogs']

        if not os.path.exists(main_log_dir):
            os.makedirs(main_log_dir)

        whole_path = os.path.join(main_log_dir, filename)
        whole_path = pathlib.Path(whole_path)

        f = open(whole_path, 'a+')

        path_of_fileFromDb = params['pred_data_preparation']['Prediction_FileFromDB']
        filename = params['pred_data_preparation']['master_csv']

        try:
            # reading a file extracted from the database
            df = pd.read_csv(os.path.join(path_of_fileFromDb,filename))

            dict = {
                '?':'b',
                'c':'c',
                'b':'b',
                'e':'e',
                'r':'r'
            }

            df['stalk_root'] = df['stalk_root'].map(dict)
            Logger().log(f, f"The ? value from the feature column named stalk-root replaced with 'b' successfully!")

        except Exception as e:
            Logger().log(f, f"Exception occured while replaceing ? value from the feature column named stalk-root 'b'. Exception: {str(e)}")
            raise e
        
        finally:
            f.close()


    def transformPipeline(self):

        """
        Description: This method is used to impute the missing values as well as to encode the categorical feature column values using scikit-learn ordinal-encoder

        Written By: Shivam Shinde

        Version: 1.0

        Revision: None
        """

        with open(os.path.join("config", "params.yaml")) as p:
            params = yaml.safe_load(p)

        main_log_dir = params['PredictionLogs']['main_log_dir']
        filename = params['PredictionLogs']['preprocessingLogs']

        if not os.path.exists(main_log_dir):
            os.makedirs(main_log_dir)

        whole_path = os.path.join(main_log_dir, filename)
        whole_path = pathlib.Path(whole_path)

        f = open(whole_path, 'a+')

        path_of_fileFromDb = params['pred_data_preparation']['Prediction_FileFromDB']
        filename = params['pred_data_preparation']['master_csv']

        
        try:
            # getting the dataframe
            df = pd.read_csv(os.path.join(path_of_fileFromDb,filename))

            # # parameters for the knn imputer
            # missing_values = params['data_preprocessing']['SimpleImputer']['missing_values']
            # strategy = params['data_preprocessing']['SimpleImputer']['strategy']


            # # parameters for the ordianl encoding 
            # handle_unknown = params['data_preprocessing']['OrdinalEncoder']['handle_unknown']
            # unknown_value = params['data_preprocessing']['OrdinalEncoder']['unknown_value']

            # # creating pipeline
            # pipeline = Pipeline([
            #     ('imputor',SimpleImputer(strategy=strategy, missing_values=missing_values)),
            #     ('encoder',OrdinalEncoder(handle_unknown=handle_unknown, unknown_value=unknown_value))
            # ])

            # loading the traning pipeline
            pipeline_path = params['data_preprocessing']['pipeline_path']

            if not os.path.exists(pipeline_path):
                os.makedirs(pipeline_path)

            with open(os.path.join(pipeline_path,"preprocessingTrainingPipelineForIndependentFeatures.pkl"), "rb") as z:
                pipeline = pickle.load(z)

            # transforming data using created pipeline
            df_transformed = pipeline.fit_transform(df)

            # creating a dataframe out of the array 
            df_transformed = pd.DataFrame(df_transformed, columns=df.columns)

            transformed_data_path = params['data_preprocessing']['preprocessed_data_dir_pred']
            transformed_data_filename = params['data_preprocessing']['preprocessed_data_filename_pred']

            # checking for the existence of the folder to save the csv file
            if not os.path.exists(transformed_data_path):
                os.makedirs(transformed_data_path)

            # writing a transformed data to a csv file
            df_transformed.to_csv(os.path.join(transformed_data_path,transformed_data_filename), header=True, index=False)

            Logger().log(f, "Processing of the data completed successfully!")
        
        except Exception as e:
            Logger().log(f, f"Exception occurred while preprocessing a prediction data. Exception: {str(e)}")
            raise e
        
        finally:
            f.close()
        


if __name__=='__main__':


    try:
        # creating an object of the class
        obj = PredictionPreprocessing()

        # replacing question mark with 'b' in the feature column stalk-root
        obj.replaceQuestionMark()

        # imputing missing values and encoding categorical columns
        obj.transformPipeline()

    except Exception as e:
        raise e




            


