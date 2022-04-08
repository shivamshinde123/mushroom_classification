import warnings

from kneed import KneeLocator
from sklearn.cluster import KMeans
import os
import pathlib
import yaml
from performLogging import Logger
from model_methods import modelMethods
import pandas as pd

warnings.simplefilter(action='ignore', category=FutureWarning)

class Cluster:

    """
    Description: This method is used to assign a cluster to every observation in the data

    Written By: Shivam Shinde

    Version: 1.0

    Revision: None
    """

    def __init__(self):
        pass
    
    def createElbowPlot(self,data):

        """
        Description: This method is used to create a elbow plot using KMeans clustering algorithm. This method also
        returns the ideal number of cluster for the provided data

        Written By: Shivam Shinde

        Version: 1.0

        Revision: None

        :param data: The data which need to be clustered

        :return: Ideal number of clusters
        """

        with open(os.path.join("config", "params.yaml")) as p:
            params = yaml.safe_load(p)

        main_log_dir = params['TrainingLogs']['main_log_dir']
        filename = params['TrainingLogs']['clustering']

        if not os.path.exists(main_log_dir):
            os.makedirs(main_log_dir)

        whole_path = os.path.join(main_log_dir, filename)
        whole_path = pathlib.Path(whole_path)

        f = open(whole_path, 'a+')

        Logger().log(f, "Finding the optimal number of clusters into which the data can be split")

        wcss = []
        try:
            # finding the value of wcss for the number of clusters from 2 to 11
            for i in range(2,11):
                kmeans = KMeans(n_clusters=i, init="k-means++",random_state=345)
                kmeans.fit(data)
                wcss.append(kmeans.inertia_)

            # finding the optimal number of clusters for the data
            kn = KneeLocator(range(2,11),wcss, curve="convex",direction="decreasing")
            Logger().log(f, f"Optimal number of clusters for the provided data is {kn.knee}")
            return kn.knee

        except Exception as e:
            Logger().log(f, f"Exception occurred while finding the optimal number of clusters for "
                                              f"the data. Exception: {str(e)}")
            raise e


    def createCluster(self,data,numOfClusters):

        """
        Description: This method is used to create and assign a cluster number to every observation in the data

        Written By: Shivam Shinde

        Version: 1.0

        Revision: None

        :param data: The data on which the clustering needs to be performed

        :param numOfClusters: Ideal number of clusters into which the data needs to be clustered

        :return: Data having an additional column containing the cluster number for each of the observation in the data
        """

        with open(os.path.join("config", "params.yaml")) as p:
            params = yaml.safe_load(p)

        main_log_dir = params['TrainingLogs']['main_log_dir']
        filename = params['TrainingLogs']['clustering']

        if not os.path.exists(main_log_dir):
            os.makedirs(main_log_dir)

        whole_path = os.path.join(main_log_dir, filename)
        whole_path = pathlib.Path(whole_path)

        f = open(whole_path, 'a+')

        Logger().log(f, "Performing the clustering on the data")
        try:
            # creating a clustering model for the data
            kmeans = KMeans(n_clusters=numOfClusters, init="k-means++", random_state=38497)

            # predicting the cluster number to which every data observation belong to.
            y_means = kmeans.fit_predict(data)

            # saving the clustering model created
            modelmethods = modelMethods()
            modelmethods.modelSaving(kmeans,"KMeansCluster",numOfClusters)

            # getting the name of the clustering column name using the parameter yaml file
            clusterColumnName = params['training']['clusteringColumnName']

            # adding a column containing cluster number for each of the data observations
            data[clusterColumnName] = y_means

            Logger().log(f, f"Successfully created {str(numOfClusters)} clusters for the data")

            return data

        except Exception as e:
            Logger().log(f, f"Exception occurred while clustering the data. Exception: {str(e)}")
            raise e

        finally:
            f.close()

