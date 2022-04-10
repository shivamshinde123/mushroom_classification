import pathlib
from flask import Flask, redirect, render_template, request, Response, url_for, session
import secrets
import json

from Predictions_using_trained_model import predictionsUsingTheTrainedModels
from predictionDatabaseOperations import PredictionDBOperations
from predictionPreprocessing import PredictionPreprocessing
from predictionRawDataTransformation import RawPredictionDataTransformation
from predictionRawDataValidation import PredictionRawDataValidation
from modeltraining import *
from model_methods import *
app = Flask(__name__)

secret = secrets.token_urlsafe(32)

app.secret_key = secret


@app.errorhandler(404)
def page_not_found(error):
    return render_template('404.html'), 404


@app.errorhandler(500)
def internal_error(error):
    return render_template('500.html'), 500


@app.route('/', methods=['GET'])
def index():
    return render_template("index.html")


# @app.route('/results')
# def results():
#     return render_template('results.html')


@app.route('/predictions', methods=['POST'])
def prediction():

    if request.form is not None:

        path = request.form['folderpath']
        path = pathlib.Path(path)

        # _______________________PREDICTIN DATA VALIDATION STEP_____________________________
        # creating a object of a class
        obj = PredictionRawDataValidation()

        # getting the pattern of the raw data filename
        raw_data_filename_pattern = obj.manualRegexCreation()

        # validating the raw data filename
        obj.validatePredictionDataFileName(
            raw_data_filename_pattern, path)

        # validating the number of columns in the raw data file
        obj.validateNumberOfColumns(22)

        # checking the any of the raw data file contain any column with all missing values
        obj.validateMissingValuesInWholeColumn()

        # ______________________PREDICTION DATA TRANSFORMATION STEP_________________________________
        # creating object of the class
        obj1 = RawPredictionDataTransformation()

        # adding quotes to the string values from the dataframe
        obj1.addingQuotesToStringColumns()

        # removing hyphens from the column names of the dataframe
        obj1.removeHyphenFromColumnName()

        # ______________________PREDICTION DATA DATABASE INSERTION STEP____________________________________
        # creating a object of a class
        obj2 = PredictionDBOperations()

        # deleting the previously created table from the database
        obj2.deleteDB()

        # getting the dictonary containing the name and datatypes of the columns
        with open(os.path.join("config", "params.yaml")) as p:
            params = yaml.safe_load(p)

        schema_prediction_path = params['pred_data_preparation']['schema_prediction']

        with open(schema_prediction_path, "r") as k:
            json_file = json.load(k)
            name_dtype_dict = json_file['ColumnNames']

        # creating a database and the creating a table into it
        obj2.createTableIntoDb(name_dtype_dict)

        # adding the data from the file from good data foler into the database
        obj2.insertGoodDataIntoTable()

        # exporting the data from the database as a csv file
        obj2.getDataFromDbTableIntoCSV()

        # _____________________PREDICTIN DATA PREPROCESSING STEP_____________________________________
        # creating an object of the class
        obj3 = PredictionPreprocessing()

        # replacing question mark with 'b' in the feature column stalk-root
        obj3.replaceQuestionMark()

        # imputing missing values and encoding categorical columns
        obj3.transformPipeline()

        # ______________________PREDICTION USING TRAINED MODELS_____________________________________
        # creating a object of a class
        obj4 = predictionsUsingTheTrainedModels()

        # calling the method of class
        list = obj4.predictUsingModel()

        return render_template('predictions.html',list=list)


if __name__ == '__main__':
    app.run()
