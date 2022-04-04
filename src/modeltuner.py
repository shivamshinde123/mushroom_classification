import warnings

import numpy as np
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, GradientBoostingClassifier, AdaBoostClassifier, \
    VotingClassifier
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
import mlflow
import mlflow.sklearn

warnings.simplefilter(action='ignore', category=FutureWarning)

from performLogging import Logger
import os
import yaml
import pathlib


class modelTuner:
    """
    Description: This class contains  the methods which will be used to find the model with the highest f1 score

    Written By: Shivam Shinde

    Version: 1.0

    Revision: None
    """

    def __init__(self):
        self.svc = SVC()
        self.knc = KNeighborsClassifier()
        self.rfc = RandomForestClassifier()
        self.gbc = GradientBoostingClassifier()
        self.abc = AdaBoostClassifier()
        self.xgb = XGBClassifier()

    def tuneSVC(self, xtrain, ytrain):

        """
        Description: This method is used to tune the hyperparameters of the SVC machine learning model.

        Written By: Shivam Shinde

        Version: 1.0

        Revision: None

        :param xtrain: training independent feature data

        :param ytrain: training dependent feature data

        :return: tuned SVR model
        """

        with open(os.path.join("config", "params.yaml")) as p:
            params = yaml.safe_load(p)

        main_log_dir = params['TrainingLogs']['main_log_dir']
        filename = params['TrainingLogs']['bestModelFinding']

        if not os.path.exists(main_log_dir):
            os.makedirs(main_log_dir)

        whole_path = os.path.join(main_log_dir, filename)
        whole_path = pathlib.Path(whole_path)

        f = open(whole_path, 'a+')
        try:
            Logger().log(f, "Finding the best hyperparameters for the SVC machine learning model")

            # getting the paramter dictionary as a parameter grid from the parameters yaml file
            param_grid = params['training']['svc']['param_grid']
            cv = params['training']['svc']['cv']
            verbose = params['training']['svc']['verbose']

            # performing the grid search cv to find the best hyperparameters for the support vector classification
            svc_grid = GridSearchCV(self.svc, param_grid=param_grid, cv=cv, n_jobs=-1,verbose=verbose)
            svc_grid.fit(xtrain, ytrain)

            # creating a support vector classification model using the best hyperparameters found in last step earlier
            supportVectorClassif = svc_grid.best_estimator_
            supportVectorClassif.fit(xtrain, ytrain)

            # returning the tuned support vector classification machine learning algorithm
            return supportVectorClassif

        except Exception as e:
            Logger().log(f, f"Exception occurred while tuning the support vector classification model "
                                           f"learning algorithm. Exception: {str(e)}")
            raise e

        finally:
            f.close()

    
    def tuneKNClassifier(self,xtrain,ytrain):

        """
        Description: This method is used to tune the hyperparameters of the KNeighbours regressor

        Written By: Shivam Shinde

        Version: 1.0

        Revision: None

        :param xtrain: training independent feature data

        :param ytrain: training dependent feature data

        :return: tuned KNeighbour regressor model
        """
        with open(os.path.join("config", "params.yaml")) as p:
            params = yaml.safe_load(p)

        main_log_dir = params['TrainingLogs']['main_log_dir']
        filename = params['TrainingLogs']['bestModelFinding']

        if not os.path.exists(main_log_dir):
            os.makedirs(main_log_dir)

        whole_path = os.path.join(main_log_dir, filename)
        whole_path = pathlib.Path(whole_path)

        f = open(whole_path, 'a+')
        
        try:
            Logger().log(f, "Finding the best hyperparameters for the KNeighbours clssification machine learning model")

            # getting the dictionary as a paramter grid from parameter yaml file
            cv = params['training']['knn']['cv']
            param_grid = params['training']['knn']['param_grid']

            # performing randomized search cv to find the best hyperparameter
            knc_grid = GridSearchCV(self.knc, param_grid=param_grid ,cv=cv, n_jobs=-1)
            knc_grid.fit(xtrain,ytrain)

            # training the k-neighbors regressor using the hyperparameters found using the randomized search cv
            knc = knc_grid.best_estimator_
            knc.fit(xtrain,ytrain)

            # returing the tuned model
            return knc

        except Exception as e:
            Logger().log(f, f"Exception occurred while tuning the KNeighbours classification model. Exception: {str(e)}")
            raise e

        finally:
            f.close()

    def tuneRandomForestClassifier(self, xtrain, ytrain):

        """
        Description: This method is used to tune the hyperparameters of the random forest classifier

        Written By: Shivam Shinde

        Version: 1.0

        Revision: None

        :param xtrain: training independent feature data

        :param ytrain: training dependent feature data

        :return: tuned random forest regressor model
        """

        with open(os.path.join("config", "params.yaml")) as p:
            params = yaml.safe_load(p)

        main_log_dir = params['TrainingLogs']['main_log_dir']
        filename = params['TrainingLogs']['bestModelFinding']

        if not os.path.exists(main_log_dir):
            os.makedirs(main_log_dir)

        whole_path = os.path.join(main_log_dir, filename)
        whole_path = pathlib.Path(whole_path)

        f = open(whole_path, 'a+')

        try:
            Logger().log(f,
                            "Finding the best hyperparameters for the random forest regressor machine learning model")

            # getting the dictionary as a paramter grid from the parameter yaml file
            cv = params['training']['random_forest']['cv']
            verbose = params['training']['random_forest']['verbose']
            param_grid = params['training']['random_forest']['param_grid']

            # performing the grid search cv using the hyperparameter dictionary created earlier
            rfc_grid = GridSearchCV(self.rfc, param_grid=param_grid, cv=cv, verbose=verbose, n_jobs=-1)
            rfc_grid.fit(xtrain, ytrain)

            # creating a random forest classifier using the best hyperparameters obtained earlier
            randomforestreg = rfc_grid.best_estimator_
            randomforestreg.fit(xtrain, ytrain)

            # returning the tuned random forest classifier
            return randomforestreg

        except Exception as e:
            Logger().log(f,
                            f"Exception occurred while tuning the random forest classifier. Exception: {str(e)}")
            raise e

        finally:
            f.close()

    def tuneGradientBoostingClassifier(self,xtrain,ytrain):

        """
        Description: This method is used to tune the hyperparameters of the gradient boost classifier

        Written By: Shivam Shinde

        Version: 1.0

        Revision: None

        :param xtrain: training independent feature data

        :param ytrain: training dependent feature data

        :return: tuned gradient boost regressor model
        """

        with open(os.path.join("config", "params.yaml")) as p:
            params = yaml.safe_load(p)

        main_log_dir = params['TrainingLogs']['main_log_dir']
        filename = params['TrainingLogs']['bestModelFinding']

        if not os.path.exists(main_log_dir):
            os.makedirs(main_log_dir)

        whole_path = os.path.join(main_log_dir, filename)
        whole_path = pathlib.Path(whole_path)

        f = open(whole_path, 'a+')

        try:
            Logger().log(f, "Finding the best hyperparameters for the gradient boosting regressor model")

            # getting the dictionary as a parameter grid from the paramters yaml file
            cv = params['training']['gradient_boost']['cv']
            verbose = params['training']['gradient_boost']['verbose']
            param_grid = params['training']['gradient_boost']['param_grid']

            # finding the best hyperparameters for the model using the GridSearchCV
            gbc_grid = GridSearchCV(self.gbc, param_grid=param_grid, cv=cv, verbose=verbose, n_jobs=-1)
            gbc_grid.fit(xtrain, ytrain)

            # using the found hyperparameters to train the model
            gbc = gbc_grid.best_estimator_
            gbc.fit(xtrain, ytrain)

            return gbc

        except Exception as e:
            Logger().log(f, f"Exception occurred while tuning the gradient boosting classifier machine"
                                           f"learning model. Exception: {str(e)}")
            raise e

        finally:
            f.close()

    def tunexgboost(self, xtrain, ytrain):

        """
        Description: This method is used to tune the hyperparameters of the xgboost classifier

        Written By: Shivam Shinde

        Version: 1.0

        Revision: None

        :param xtrain: training independent feature data

        :param ytrain: training dependent feature data

        :return: tuned xgboost regressor model
        """

        with open(os.path.join("config", "params.yaml")) as p:
            params = yaml.safe_load(p)

        main_log_dir = params['TrainingLogs']['main_log_dir']
        filename = params['TrainingLogs']['bestModelFinding']

        if not os.path.exists(main_log_dir):
            os.makedirs(main_log_dir)

        whole_path = os.path.join(main_log_dir, filename)
        whole_path = pathlib.Path(whole_path)

        f = open(whole_path, 'a+')

        try:
            # Logger().log(f,
            #                 "Finding the best hyperparameters for the xgboost classifier machine learning algorithm")

            # getting the dictionary as a parameter grid from the paramters yaml file
            cv = params['training']['xg_boost']['cv']
            verbose = params['training']['xg_boost']['verbose']
            param_grid = params['training']['xg_boost']['param_grid']

            #  performing the grid search cv to find the best hyperparameters for the xgboost classifier
            xgboost_grid = GridSearchCV(self.xgb, param_grid=param_grid, cv=cv, verbose=verbose, n_jobs=-1)
            xgboost_grid.fit(xtrain, ytrain)

            # finding the best estimator
            xgboostReg = xgboost_grid.best_estimator_
            xgboostReg.fit(xtrain, ytrain)

            # returning the best estimator
            return xgboostReg

        except Exception as e:
            Logger().log(f,
                            f"Exception occurred while tuning the xgboost classification machine learning algorithm. Exception: {str(e)}")
            raise e

        finally:
            f.close()


    def tuneAdaBoostingClassifier(self,xtrain,ytrain):

        """
        Description: This method is used to tune the hyperparameters of the ada boost classifier

        Written By: Shivam Shinde

        Version: 1.0

        Revision: None

        :param xtrain: training independent feature data

        :param ytrain: training dependent feature data

        :return: tuned ada boosting regressor model
        """
        with open(os.path.join("config", "params.yaml")) as p:
            params = yaml.safe_load(p)

        main_log_dir = params['TrainingLogs']['main_log_dir']
        filename = params['TrainingLogs']['bestModelFinding']

        if not os.path.exists(main_log_dir):
            os.makedirs(main_log_dir)

        whole_path = os.path.join(main_log_dir, filename)
        whole_path = pathlib.Path(whole_path)

        f = open(whole_path, 'a+')

        try:
            Logger().log(f, "Finding the best hyperparameters for the ada boosting regressor model")

            # getting a dictionary which will contain all the possible values of the hyperparameters
            cv = params['training']['ada_boost']['cv']
            param_grid = params['training']['ada_boost']['param_grid']

            # finding the best hyperparameters for the model using the GridSearchCV
            abc_grid = GridSearchCV(self.xgb, param_grid=param_grid, cv=cv, n_jobs=-1)
            abc_grid.fit(xtrain, ytrain)

            # using the found hyperparameters to train the model
            abc = abc_grid.best_estimator_
            abc.fit(xtrain, ytrain)

            return abc

        except Exception as e:
            Logger().log(f, f"Exception occurred while tuning the ada boosting classifier machine "
                                           f"learning model. Exception: {str(e)}")
            raise e

        finally:
            f.close()

    def tuneVotingClassifier(self,xtrain,ytrain):

        """
        Description: This method is used to return the voting classifier made up of ada boosting classifier,
        gradient boosting classifier and xgboost classifier machine learning models

        Written By: Shivam Shinde

        Version: 1.0

        Revision: None

        :param xtrain: training independent feature data

        :param ytrain: training dependent feature data

        :return: voting regressor machine learning model
        """

        with open(os.path.join("config", "params.yaml")) as p:
            params = yaml.safe_load(p)

        main_log_dir = params['TrainingLogs']['main_log_dir']
        filename = params['TrainingLogs']['bestModelFinding']

        if not os.path.exists(main_log_dir):
            os.makedirs(main_log_dir)

        whole_path = os.path.join(main_log_dir, filename)
        whole_path = pathlib.Path(whole_path)

        f = open(whole_path, 'a+')

        try:
            Logger().log(f, "Creating a voting classifier using Ada boosting classifier, Gradient "
                                           "boosting classifier and xgboosting classifier")

            abc = self.tuneAdaBoostingClassifier(xtrain,ytrain)
            gbc = self.tuneGradientBoostingClassifier(xtrain,ytrain)
            xgb = self.tunexgboost(xtrain,ytrain)

            vc = VotingClassifier(estimators=[
                ('abr', abc),
                ('gbr', gbc),
                ('xgb', xgb)])

            vc.fit(xtrain,ytrain)
            return vc

        except Exception as e:
            Logger().log(f, f"Exception occurred while creating voting classifier machine "
                                           f"learning model. Exception: {str(e)}")
            raise e

        finally:
            f.close()

    def createStackingClassifier(self, xtrain, ytrain):

        """
        Description: This method is used to create a stacking classifier using SVC , random forest classifier and 
        K neighbors classifier as a base estimator and xgboost as a final (meta) estimator
        
        Written By: Shivam Shinde
        
        Version: 1.0
        
        Revision: None
        
        :return: stacking regressor model
        """

        with open(os.path.join("config", "params.yaml")) as p:
            params = yaml.safe_load(p)

        main_log_dir = params['TrainingLogs']['main_log_dir']
        filename = params['TrainingLogs']['bestModelFinding']

        if not os.path.exists(main_log_dir):
            os.makedirs(main_log_dir)

        whole_path = os.path.join(main_log_dir, filename)
        whole_path = pathlib.Path(whole_path)

        f = open(whole_path, 'a+')

        try:
            # fetching the base and metas models
            model1 = self.tuneVotingClassifier(xtrain, ytrain)
            model2 = self.tuneAdaBoostingClassifier(xtrain, ytrain)
            model3 = self.tuneRandomForestClassifier(xtrain, ytrain)
            metamodel = self.tunexgboost(xtrain, ytrain)

            # creating a list of base estimators
            estimators = [
                ('vc', model1),
                ('abc', model2),
                ('rfc', model3)
            ]

            # creating and fitting a stacking classifier
            stackingReg = StackingClassifier(estimators=estimators, final_estimator=metamodel)
            stackingReg.fit(xtrain, ytrain)

            # returning a created stacking classifier
            return stackingReg

        except Exception as e:
            Logger().log(f,
                            f"Exception occurred while creating a stacking classifier. Exception: {str(e)}")
            raise e
        
        finally:
            f.close()


    def bestModelFinder(self, xtrain, xtest, ytrain, ytest):

        """
        Description: This method is used to find the best model using the f1 score

        Written By: Shivam Shinde

        Version: 1.0

        Revision: None

        :param xtrain: training independent feature data

        :param ytrain: training dependent feature data

        :param xtest: testing independent feature data

        :param ytest: testing dependent feature data
        
        :return: best model that fits the concerned cluster data
        """
        with open(os.path.join("config", "params.yaml")) as p:
            params = yaml.safe_load(p)

        main_log_dir = params['TrainingLogs']['main_log_dir']
        filename = params['TrainingLogs']['bestModelFinding']

        if not os.path.exists(main_log_dir):
            os.makedirs(main_log_dir)

        whole_path = os.path.join(main_log_dir, filename)
        whole_path = pathlib.Path(whole_path)

        f = open(whole_path, 'a+')

        try:
            Logger().log(f , "*********finding best model for cluster************")

            # finding the evaluation metrices score for SVC model
            with mlflow.start_run(nested=True):
                svc = self.tuneSVC(xtrain, ytrain)
                svc_predictions = svc.predict(xtest)

                precision_score_svc = precision_score(ytest, svc_predictions)
                recall_score_svc = recall_score(ytest, svc_predictions)
                roc_auc_score_svc = roc_auc_score(ytest, svc_predictions)
                f1_score_svc = f1_score(ytest, svc_predictions)

                Logger().log(f , f"The precision score for the SVC model is {np.round(precision_score_svc,3)}")
                Logger().log(f , f"The recall score for the SVC model is {np.round(recall_score_svc,3)}")
                Logger().log(f , f"The roc auc score for the SVRCmodel is {np.round(roc_auc_score_svc,3)}")
                Logger().log(f , f"The f1 score for the SVC model is {np.round(f1_score_svc,3)}")

                mlflow.log_metric('precision_score_svc', precision_score_svc)
                mlflow.log_metric('recall_score_svc', recall_score_svc)
                mlflow.log_metric('roc_auc_score_svc', roc_auc_score_svc)
                mlflow.log_metric('f1_score_svc', f1_score_svc)

            # finding the evaluation metrices for the KNeighbors classification model
            with mlflow.start_run(nested=True):
                knc = self.tuneKNClassifier(xtrain,ytrain)
                knc_prediction = knc.predict(xtest)

                precision_score_knc = precision_score(ytest, knc_prediction)
                recall_score_knc = recall_score(ytest, knc_prediction)
                roc_auc_score_knc = roc_auc_score(ytest, knc_prediction)
                f1_score_knc = f1_score(ytest, knc_prediction)

                Logger().log(f , f"The precision score for the K nearest neighbours classification model is {np.round(precision_score_knc,3)}")
                Logger().log(f , f"The recall score for the K nearest neighbours classification model is {np.round(recall_score_knc,3)}")
                Logger().log(f , f"The roc auc score for the K nearest neighbours classification model is {np.round(roc_auc_score_knc,3)}")
                Logger().log(f , f"The f1 score for the K nearest neighbours classification model is {np.round(f1_score_knc,3)}")

                mlflow.log_metric('precision_score_knc', precision_score_knc)
                mlflow.log_metric('recall_score_knc', recall_score_knc)
                mlflow.log_metric('roc_auc_score_knc', roc_auc_score_knc)
                mlflow.log_metric('f1_score_knc', f1_score_knc)


            # finding the evaluation metrices for random forest classification model
            with mlflow.start_run(nested=True):
                rfc = self.tuneRandomForestClassifier(xtrain, ytrain)
                rfc_prediction = rfc.predict(xtest)

                precision_score_rfc = precision_score(ytest, rfc_prediction)
                recall_score_rfc = recall_score(ytest, rfc_prediction)
                roc_auc_score_rfc = roc_auc_score(ytest, rfc_prediction)
                f1_score_rfc = f1_score(ytest, rfc_prediction)

                Logger().log(f , f"The precision score for the random forest classification model is {np.round(precision_score_knc,3)}")
                Logger().log(f , f"The recall score for the random forest classification model is {np.round(recall_score_knc,3)}")
                Logger().log(f , f"The roc auc score for the random forest classification model is {np.round(roc_auc_score_knc,3)}")
                Logger().log(f , f"The f1 score for the random forest classification model is {np.round(f1_score_knc,3)}")

                mlflow.log_metric('precision_score_rfc', precision_score_rfc)
                mlflow.log_metric('recall_score_rfc', recall_score_rfc)
                mlflow.log_metric('roc_auc_score_rfc', roc_auc_score_rfc)
                mlflow.log_metric('f1_score_rfc', f1_score_rfc)



            # finding the evaluation metrics for gradient boosting classification model
            with mlflow.start_run(nested=True):
                gbc = self.tuneGradientBoostingClassifier(xtrain,ytrain)
                gbc_prediction = gbc.predict(xtest)

                precision_score_gbc = precision_score(ytest, gbc_prediction)
                recall_score_gbc = recall_score(ytest, gbc_prediction)
                roc_auc_score_gbc = roc_auc_score(ytest, gbc_prediction)
                f1_score_gbc = f1_score(ytest, gbc_prediction)

                Logger().log(f , f"The precision score for the gradient boosting classification model is {np.round(precision_score_gbc,3)}")
                Logger().log(f , f"The recall score for the gradient boosting classification model is {np.round(recall_score_gbc,3)}")
                Logger().log(f , f"The roc auc score for the gradient boosting classification model is {np.round(roc_auc_score_gbc,3)}")
                Logger().log(f , f"The f1 score for the gradient boosting classification model is {np.round(f1_score_gbc,3)}")

                mlflow.log_metric('precision_score_gbc', precision_score_gbc)
                mlflow.log_metric('recall_score_gbc', recall_score_gbc)
                mlflow.log_metric('roc_auc_score_gbc', roc_auc_score_gbc)
                mlflow.log_metric('f1_score_gbc', f1_score_gbc)

            # finding the evaluation metrics for xgboost classification model
            with mlflow.start_run(nested=True):
                xgb = self.tunexgboost(xtrain, ytrain)
                xgb_prediction = xgb.predict(xtest)

                precision_score_xgb = precision_score(ytest, xgb_prediction)
                recall_score_xgb = recall_score(ytest, xgb_prediction)
                roc_auc_score_xgb = roc_auc_score(ytest, xgb_prediction)
                f1_score_xgb = f1_score(ytest, xgb_prediction)

                Logger().log(f , f"The precision score for the xgboost classification model is {np.round(precision_score_xgb,3)}")
                Logger().log(f , f"The recall score for the xgboost classification model is {np.round(recall_score_xgb,3)}")
                Logger().log(f , f"The roc auc score for the xgboost classification model is {np.round(roc_auc_score_xgb,3)}")
                Logger().log(f , f"The f1 score for the xgboost classification model is {np.round(f1_score_xgb,3)}")

                mlflow.log_metric('precision_score_xgb', precision_score_xgb)
                mlflow.log_metric('recall_score_xgb', recall_score_xgb)
                mlflow.log_metric('roc_auc_score_xgb', roc_auc_score_xgb)
                mlflow.log_metric('f1_score_xgb', f1_score_xgb)

        
            # finding the evaluation metrics for ada boosting classification model
            with mlflow.start_run(nested=True):
                abc = self.tuneAdaBoostingClassifier(xtrain, ytrain)
                abc_prediction = abc.predict(xtest)

                precision_score_abc = precision_score(ytest, abc_prediction)
                recall_score_abc = recall_score(ytest, abc_prediction)
                roc_auc_score_abc = roc_auc_score(ytest, abc_prediction)
                f1_score_abc = f1_score(ytest, abc_prediction)

                Logger().log(f , f"The precision score for the ada boost classification model is {np.round(precision_score_abc,3)}")
                Logger().log(f , f"The recall score for the ada boost classification model is {np.round(recall_score_abc,3)}")
                Logger().log(f , f"The roc auc score for the ada boost classification model is {np.round(roc_auc_score_abc,3)}")
                Logger().log(f , f"The f1 score for the ada boost classification model is {np.round(f1_score_abc,3)}")

                mlflow.log_metric('precision_score_abc', precision_score_abc)
                mlflow.log_metric('recall_score_abc', recall_score_abc)
                mlflow.log_metric('roc_auc_score_abc', roc_auc_score_abc)
                mlflow.log_metric('f1_score_abc', f1_score_abc)



            # finding the evaluation metrics for the voting classifier model created
            with mlflow.start_run(nested=True):
                vc = self.tuneVotingClassifier(xtrain,ytrain)
                vc_prediction = vc.predict(xtest)
                
                precision_score_vc = precision_score(ytest, vc_prediction)
                recall_score_vc = recall_score(ytest, vc_prediction)
                roc_auc_score_vc = roc_auc_score(ytest, vc_prediction)
                f1_score_vc = f1_score(ytest, vc_prediction)

                Logger().log(f , f"The precision score for the voting classification model is {np.round(precision_score_vc,3)}")
                Logger().log(f , f"The recall score for the voting classification model is {np.round(recall_score_vc,3)}")
                Logger().log(f , f"The roc auc score for the voting classification model is {np.round(roc_auc_score_vc,3)}")
                Logger().log(f , f"The f1 score for the voting classification model is {np.round(f1_score_vc,3)}")

                mlflow.log_metric('precision_score_vc', precision_score_vc)
                mlflow.log_metric('recall_score_vc', recall_score_vc)
                mlflow.log_metric('roc_auc_score_vc', roc_auc_score_vc)
                mlflow.log_metric('f1_score_vc', f1_score_vc)



            # finding the evaluation metrics for the stacking classification model
            with mlflow.start_run(nested=True):
                sc = self.createStackingClassifier(xtrain, ytrain)
                sc_prediction = sc.predict(xtest)

                precision_score_sc = precision_score(ytest, sc_prediction)
                recall_score_sc = recall_score(ytest, sc_prediction)
                roc_auc_score_sc = roc_auc_score(ytest, sc_prediction)
                f1_score_sc = f1_score(ytest, sc_prediction)

                Logger().log(f , f"The precision score for the stacking classification model is {np.round(precision_score_sc,3)}")
                Logger().log(f , f"The recall score for the stacking classification model is {np.round(recall_score_sc,3)}")
                Logger().log(f , f"The roc auc score for the stacking classification model is {np.round(roc_auc_score_sc,3)}")
                Logger().log(f , f"The f1 score for the stacking classification model is {np.round(f1_score_sc,3)}")

                mlflow.log_metric('precision_score_vc', precision_score_sc)
                mlflow.log_metric('recall_score_vc', recall_score_sc)
                mlflow.log_metric('roc_auc_score_vc', roc_auc_score_sc)
                mlflow.log_metric('f1_score_vc', f1_score_sc)

        
            Logger().log(f , "*********found the best model for cluster************")

            ## here we will use the f1_score as a evaluation metrics for choosing the best model for mushroom classification

            max_f1 = max([f1_score_sc,f1_score_vc,f1_score_abc,f1_score_xgb,f1_score_gbc,f1_score_rfc,f1_score_knc,f1_score_svc])

            if max_f1 == f1_score_sc:
                return "StackingClassifier", sc
            elif max_f1 == f1_score_vc:
                return "VotingClassifier", vc
            elif max_f1 == f1_score_xgb:
                return "XGBClassifier", xgb
            elif max_f1 == f1_score_rfc:
                return "RandomForestClassifier", rfc
            elif max_f1 == f1_score_gbc:
                return "GradientBoostingClassifier", gbc
            elif max_f1 == f1_score_abc:
                return "AdaBoostingClassifier", abc
            elif max_f1 == f1_score_knc:
                return "KNeighborsClassifier", knc
            else:
                return "SVC", svc



        except Exception as e:
            Logger().log(f , f"Exception occurred while finding the best machine learning model. "
                                           f"Exception: {str(e)}")
            raise e

        finally:
            f.close()
