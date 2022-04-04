import warnings

from sklearn.model_selection import train_test_split

from performLogging import Logger
from clustering import Cluster
from model_methods import modelMethods
from modeltuner import modelTuner
from sklearn.base import BaseEstimator, ClassifierMixin

import os
import yaml
import pathlib
import pandas as pd

warnings.simplefilter(action='ignore', category=FutureWarning)


class ModelWithSameOutputEverytime(BaseEstimator,ClassifierMixin):


    """
    Description: This method is used to create a machine learning model which returns the same value no matter what the prediction data is.

    Written By: Shivam Shinde

    Version: 1.0

    Revision: None

    :returns: None
    """

    def __init__(self):
        pass

    def fit(self, X, y = None):
        self.unique_label = y[1]
        return self

    def predict(self, X, y = None):
        return self.unique_label * len(X)


class modelTraining:

    """
    Description: This class a method which is used to train a machine learning model for each data cluster

    Written By: Shivam Shinde

    Version: 1.0

    Revision: None

    :returns: None
    """

    def __init__(self):
        pass

    def trainingModels(self):
        """
        Description: This method is used to train the machine learning model for the every cluster of the data

        Written By: Shivam Shinde

        Version: 1.0

        Revision: None
        
        :return: None
        """

        with open(os.path.join("config", "params.yaml")) as p:
            params = yaml.safe_load(p)

        main_log_dir = params['TrainingLogs']['main_log_dir']
        filename = params['TrainingLogs']['values_from_schema']

        if not os.path.exists(main_log_dir):
            os.makedirs(main_log_dir)

        whole_path = os.path.join(main_log_dir, filename)
        whole_path = pathlib.Path(whole_path)

        f = open(whole_path, 'a+')
        try:
            Logger().log(
                f, "*************MACHINE LEARNING MODEL TRAINING FOR ALL THE CLUSTERS STARTED**************")


            # getting the path where the training preprocessed data is placed
            preprocessed_data_dir = params['data_preprocessing']['preprocessed_data_dir']
            preprocessed_data_filename = params['data_preprocessing']['preprocessed_data_filename']

            preprocessed_data_whole_file_path = os.path.join(preprocessed_data_dir,preprocessed_data_filename)

            # preprocessing the obtained data
            Logger().log(f, "Getting training preprocessed data.")

            df = pd.read_csv(preprocessed_data_whole_file_path)
            X = df.drop(columns=['class'])
            y = df['class']

            Logger().log(f, "Got training preprocessed data")

            # clustering the training and testing data into the same number of clusters
            Logger().log(f, "Training_Clustering of the data started!!")
            c = Cluster()
            noOfClusters = c.createElbowPlot(X)
            X = c.createCluster(X, noOfClusters)
            Logger().log(f, "Training_Clustering of the data completed!!")

            # Adding one more column to X i.e. dependent feature
            X['class'] = y

            # getting the name of the clustering column name using the parameter yaml file
            clusterColumnName = params['training']['clusteringColumnName']

            # finding the unique numbers in the ClusterNumber column of the X
            clusters = X[clusterColumnName].unique()


            for i in clusters:
                Logger().log(f, f"*************for the cluster number {i}**************")
                clusterData = X[X[clusterColumnName] == i]

                clusterFeatures = clusterData.drop(
                    columns=[clusterColumnName, 'class'], axis=1)
                clusterLabel = clusterData['class']

                # getting the parameters for the train-test split of the data
                test_size = params['training']['test_size']
                random_state = params['base']['random_state']

                # splitting the cluster data into train and test data
                X_train, X_test, y_train, y_test = train_test_split(
                    clusterFeatures, clusterLabel, test_size=test_size, random_state=random_state,shuffle=True,stratify=clusterLabel)

                Logger().log(f,f"Finding the best model for the cluster {i}")

                # Here we will first check whether the label column of the cluster data contains more than 1 unique values since sklearn models require atleast 2 unique values.
                # If the label column of the cluster data contains only one unique value then we will create a model which will return that unique value as prediction no matter
                # what the input is. And otherwise the sklearn models will be used for the model training
                if pd.DataFrame(y_train)['class'].nunique() == 1:
                    model_which_gives_same_value_everytime = ModelWithSameOutputEverytime().fit(X_train,y_train)
                    mm = modelMethods()
                    mm.modelSaving(model_which_gives_same_value_everytime, "ModelWithConstantOutput", i)


                else:
                    # finding the best model for this cluster
                    mt = modelTuner()
                    bestModelName, bestModel = mt.bestModelFinder(
                        X_train, X_test, y_train, y_test)

                    # saving the best model obtained
                    mm = modelMethods()
                    mm.modelSaving(bestModel, bestModelName, i)

                Logger().log(
                    f,
                    f"Training of the machine learning model for the data cluster {i} successfully completed")

            Logger().log(f, "***************MACHINE LEARNING MODEL TRAINING FOR ALL CLUSTERS COMPLETED "
                                           "SUCCESSFULLY*************")

        except Exception as e:
            Logger().log(
                f, f"Exception occurred while training the machine learning model. Exception: {str(e)}")
            raise e

        finally:
            f.close()



if __name__ == "__main__":

    try:
        # creating a object of a class
        obj = modelTraining()

        # training the model 
        obj.trainingModels()

    except Exception as e:
        raise e

