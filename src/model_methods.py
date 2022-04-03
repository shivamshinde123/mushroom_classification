import os
import pathlib
import pickle
import shutil
import warnings
import yaml
from performLogging import Logger

warnings.simplefilter(action='ignore', category=FutureWarning)

class modelMethods:
    """

    Description: This class will contain the methods used for saving, loading and finding the correct model for
    correct cluster

    Written By: Shivam Shinde

    Version: 1.0

    Revision: None

    """

    def __init__(self):
        pass

    def modelSaving(self, model, filename,clusterno):

        """
        Description: This method is used to save the created model as a python pickle file

        :param model: Reference of the created model

        :param filename: Name of the model after saving

        :return: None
        """
        with open(os.path.join("config", "params.yaml")) as p:
            params = yaml.safe_load(p)

        main_log_dir = params['TrainingLogs']['main_log_dir']
        filename = params['TrainingLogs']['model_methods']

        if not os.path.exists(main_log_dir):
            os.makedirs(main_log_dir)

        whole_path = os.path.join(main_log_dir, filename)
        whole_path = pathlib.Path(whole_path)

        f = open(whole_path, 'a+')

        Logger().log(f, "Saving the created model into the python pickle file")

        model_dir = params['saved_models']['model_dir']
        try:
            if filename == "KMeansCluster":
                path = os.path.join(model_dir, "ClusteringModel")
            else:
                path = os.path.join(model_dir, "ModelForClusterNo"+str(clusterno))

            # removing the previously created models
            if os.path.exists(path):
                shutil.rmtree(model_dir)
                os.makedirs(path)
            else:
                os.makedirs(path)

            # saving the model as a python pickle file
            pickle.dump(model, open(os.path.join(path, f"{filename}.pkl"),"wb"))

            Logger().log(f, f"Model {model} saved successfully in {path}")

        except Exception as e:
            Logger().log(f, f"Exception occurred while saving the model {model}. Exception: {str(e)}")
            raise e

        finally:
            f.close()

    def loadingSavedModel(self, filename, clusterno):

        """
        Description: This method is used to load the saved method for the respective cluster

        Written By: Shivam Shinde

        Version: 1.0

        Revision: None

        :param clusterno: Cluster number for which the model is to be loaded
        :param filename: Name of the model that needs to be saved

        :return: Model
        """

        with open(os.path.join("config", "params.yaml")) as p:
            params = yaml.safe_load(p)

        main_log_dir = params['TrainingLogs']['main_log_dir']
        filename = params['TrainingLogs']['model_methods']

        if not os.path.exists(main_log_dir):
            os.makedirs(main_log_dir)

        whole_path = os.path.join(main_log_dir, filename)
        whole_path = pathlib.Path(whole_path)

        f = open(whole_path, 'a+')

        model_dir = params['saved_models']['model_dir']

        try:
            Logger().log(f, f"Loading the model {filename}.pkl")
            if filename == "KMeansCluster":
                path = os.path.join(model_dir, "ClusteringModel")
            else:
                path = os.path.join(model_dir, "ModelForClusterNo" + str(clusterno))

            # loading the saved model
            path1 = os.path.join(path, f"{filename}.pkl")
            model = pickle.load(open(path1,"rb"))
            Logger().log(f, f"Model {filename} loaded successfully")

            # returning the model
            return model

        except  Exception as e:
            Logger().log(f, f"Exception occurred while loading the model {filename}. "
                                              f"Exception: {str(e)}")
            raise e

        finally:
            f.close()

    def findCorrectModel(self, clusterNumber):

        """
        Description: This method is used to find the correct model given the  cluster number

        Written By: Shivam Shinde

        Version: 1.0

        Revision: None

        :param clusterNumber: Cluster number
        
        :return: Model name
        """

        with open(os.path.join("config", "params.yaml")) as p:
            params = yaml.safe_load(p)

        main_log_dir = params['TrainingLogs']['main_log_dir']
        filename = params['TrainingLogs']['model_methods']

        if not os.path.exists(main_log_dir):
            os.makedirs(main_log_dir)

        whole_path = os.path.join(main_log_dir, filename)
        whole_path = pathlib.Path(whole_path)

        f = open(whole_path, 'a+')

        model_dir = params['saved_models']['model_dir']

        Logger().log(f, f"Finding the appropriate model for cluster number {clusterNumber}")
        try:
            # finding the appropriate model for the given cluster number
            for file in os.listdir(model_dir):

                path = os.path.join(model_dir,file)
                path = pathlib.Path(path)
                if (path.stem[-1]) == str(clusterNumber):
                    for file1 in os.listdir(path):
                        model_name = file1.split('.')[0]
                        Logger().log(f,
                                        f"Successfully found the name of the model for the cluster number "
                                        f"{clusterNumber}")
                        # returning the model
                        return model_name
                else:
                    continue

        except Exception as e:
            Logger().log(f, f"Exception occurred while finding the name of the model for the "
                                              f"cluster number {clusterNumber}. Exception: {str(e)}")
            raise e

        finally:
            f.close()


