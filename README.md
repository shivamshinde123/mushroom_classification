Mushroom Classification
==============================

The main goal is to predict which mushroom is poisonous & which is edible given mushroom features

Project Organization
``` bash
|   .dvcignore
|   .env
|   .gitignore
|   dvc.lock
|   dvc.yaml
|   LICENSE
|   Makefile
|   mlruns.dvc
|   README.md
|   requirements.txt
|   setup.py
|   test_environment.py
|   tox.ini
|   tree.txt
+---docs
|
|
+---data
|
|
+---mlruns
|
|
+---models
|   +---ClusteringModel
|   |       KMeansCluster.pkl
|   |       
|   +---ModelForClusterNo0
|   |       StackingClassifier.pkl
|   |       
|   +---ModelForClusterNo1
|   |       StackingClassifier.pkl
|   |       
|   +---ModelForClusterNo2
|   |       StackingClassifier.pkl
|   |       
|   +---ModelForClusterNo3
|   |       ModelWithConstantOutput.pkl
|   |       
|   \---ModelForClusterNo4
|           StackingClassifier.pkl
|           
+---notebooks
|       .gitkeep
|       EDA and preprocessing.ipynb
|       
+---PredictionLogs
|       columnWithAllMissingValuesValidation.txt
|       DatabaseLogs.txt
|       GoodAndBadDataFileCreationLogs.txt
|       numberOfColumnsValidation.txt
|       Prediction.txt
|       preprocessingLogs.txt
|       RawDataFileNameValidation.txt
|       RawPredictionDataTransformation.txt
|       
+---Prediction_Batch_Files
|       .gitignore
|       Mushroom_Data_26112022_103005.csv
|       Mushroom_Data_26112022_103005.csv.dvc
|       
+---Prediction_Database
|       Prediction.db
|       
+---Prediction_Output_File
|       Predictions.csv
|       
+---references
|       .gitkeep
|       
+---reports
|   |   .gitkeep
|   |   
|   \---figures
|           .gitkeep
|           
+---src
|   |   app.py
|   |   clustering.py
|   |   modeltraining.py
|   |   modeltuner.py
|   |   model_methods.py
|   |   performLogging.py
|   |   predictionDatabaseOperations.py
|   |   predictionPreprocessing.py
|   |   predictionRawDataTransformation.py
|   |   predictionRawDataValidation.py
|   |   Predictions_using_trained_model.py
|   |   trainingDatabaseOperations.py
|   |   trainingPreprocessing.py
|   |   trainingRawDataTransformation.py
|   |   trainingRawDataValidation.py
|   |   __init__.py
|   |   
|   +---templates
|   |       404.html
|   |       500.html
|   |       base.html
|   |       index.html
|   |       results.html
|   |       
|   \---__pycache__
|             
+---TrainingLogs
|       bestModelFindingLogs.txt
|       clusteringLogs.txt
|       columnWithAllMissingValuesValidation.txt
|       DatabaseLogs.txt
|       GoodAndBadDataFileCreationLogs.txt
|       modelMethodsLogs.txt
|       numberOfColumnsValidation.txt
|       preprocessingLogs.txt
|       RawDataFileNameValidation.txt
|       RawTrainingDataTransformation.txt
|       valuesFromSchemaLog.txt
|       
+---Training_Batch_Files
|       .gitignore
|       Mushroom_Data_23042021_023412.csv
|       Mushroom_Data_23042021_023412.csv.dvc
|       
\---Training_Database
|        Training.db
|
|
```


--------
