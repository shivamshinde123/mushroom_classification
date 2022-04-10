import os

import pandas as pd

from performLogging import Logger
from model_methods import *
import yaml
import pathlib
import shutil
import json

class predictionsUsingTheTrainedModels:

    """
    Descriptions: This class contains the methods which will predict the flight fare for the given records in the csv file

    Written By: Shivam Shinde

    Version: 1.0

    Revision: None
    """

    def __init__(self):
        pass

    def predictUsingModel(self):

        """
        Description : This method is used to predict what category the mushrrom belong when the attributes of the mushroom are given

        Written By: Shivam Shinde

        Version: 1.0

        Revision: None
        :return: path of the csv file containing the predicted values
        """
        with open(os.path.join("config", "params.yaml")) as p:
            params = yaml.safe_load(p)

        main_log_dir = params['PredictionLogs']['main_log_dir']
        filename = params['PredictionLogs']['Prediction']

        if not os.path.exists(main_log_dir):
            os.makedirs(main_log_dir)

        whole_path = os.path.join(main_log_dir, filename)
        whole_path = pathlib.Path(whole_path)

        f = open(whole_path, 'a+')

        try:
            # deleting the prediction files from the previous code run
            prediction_output_dir = params['pred_data_preparation']['Prediction_Output_File_dir']
            prediction_output_filename = params['pred_data_preparation']['Prediction_Output_Filename']

            if os.path.exists(prediction_output_dir):
                shutil.rmtree(prediction_output_dir)

            # reading the preprocessed prediction input data
            preprocessed_data_dir = params['data_preprocessing']['preprocessed_data_dir_pred']
            preprocessed_data_filename = params['data_preprocessing']['preprocessed_data_filename_pred']
            data = pd.read_csv(os.path.join(preprocessed_data_dir,preprocessed_data_filename))

            # clustering the data into 4 clusters
            # loading the trained kmeans model
            mm = modelMethods()
            kmeans_model1 = mm.loadingSavedModel("KMeansCluster", 5)
            clusterNumbers = kmeans_model1.predict(data)

            # getting the cluster column name
            clusterColumnName = params['pred_data_preparation']['clusteringColumnName']

            # creating a column in the data which will contains the cluster number for the particular observation
            data[clusterColumnName] = clusterNumbers

            clusters = data[clusterColumnName].unique()

            # creating an empty list which contains the prediction results
            prediction_results = []

            # creating an empty list which will contain the indices of the predictions
            prediction_indices = []

            for i in clusters:

                clusterData = data[data[clusterColumnName] == i]

                # adding the indices of the observations for the cluster number i
                for j in clusterData.index:
                    prediction_indices.append(j)

                clusterFeatures = clusterData.drop(columns=[clusterColumnName], axis=1)

                # finding the saved model for cluster number i
                model_name = modelMethods().findCorrectModel(i)
                model = modelMethods().loadingSavedModel(model_name, i)

                # predicting the flight fare for each of the observations in the data
                predictions = model.predict(clusterFeatures)

                # appending the prediction results found in an empty list
                for prediction in predictions:
                    prediction_results.append(prediction)

            if not os.path.exists(prediction_output_dir):
                os.makedirs(prediction_output_dir)

            prediction_df = pd.DataFrame(prediction_results,columns=["class"])
            prediction_df['Index-column'] = prediction_indices

            # sorting the predictions dataframe using the values from the column named Index_column
            # prediction_df.sort_values('Index-column',ascending=True,inplace=True)

            # # getting the original file from predictin batch file folder
            path_for_input_file = params['data_source']['batch_files_pred']
            files = os.listdir(path_for_input_file)
            for file in files:
                if file.split('.')[-1] == 'csv':
                    input_file = pd.read_csv(os.path.join(path_for_input_file,file))
                    input_file['Index-column'] = input_file.index

            # merging the prediction_df and original prediction input dataframe based on the Index-column feature
            output_file = pd.merge(input_file, prediction_df, how="outer", on="Index-column")
            output_file.drop(columns=['Index-column'], axis=1, inplace=True)

            # converting back the encoded dependent feature values to its original form using the json file saved in the encoding_details directory
            encoding_details_path = params['data_preprocessing']['encoding_details']
            with open(os.path.join(encoding_details_path,"encoding_details_dependent_feature.json"),"r+") as g:
                json_file = json.load(g)

            e = json_file['encoded_values']['e']
            p = json_file['encoded_values']['p']

            map_dict = dict()
            map_dict[e] = 'e'
            map_dict[p] = 'p'

            output_file['class'] = output_file['class'].map(map_dict)
            
            if not os.path.exists(prediction_output_dir):
                os.makedirs(prediction_output_dir)

            prediction_output_file_path = os.path.join(prediction_output_dir,prediction_output_filename)

            output_file.to_csv(prediction_output_file_path,header=True)

            list = []
            for i in range(output_file.shape[0]):
                row = output_file.iloc[i,:]
                list.append(row.to_dict())

            Logger().log(f, f"Prediction results placed at the path: {prediction_output_file_path}")

            return list
        except Exception as e:
            Logger().log(f, f"Exception occurred while predicting the mushroom category using the saved "
                                           f"models. Exception: {str(e)}")
            raise e


if __name__ == '__main__':

    try:
        # creating a object of a class
        obj = predictionsUsingTheTrainedModels()

        from model_methods import *

        # calling the method of class
        obj.predictUsingModel()
    
    except Exception as e:
        raise e