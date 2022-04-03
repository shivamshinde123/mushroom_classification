import json
import os
import re
import shutil
from datetime import datetime
import yaml
import pandas as pd
import pathlib
from performLogging import Logger


class rawDataValidation:

    """
    Description: This class contains the methods used for the validation of the raw training data

    Written By: Shivam Shinde

    Version: 1.0

    Revision: None

    """

    def __init__(self):
        pass

    def valuesFromSchema(self):
        """
        Description: This method is used for fetching the information from the schema_training json file.

        Raises: Exception on failure

        Written By: Shivam Shinde

        Version: 1.0

        Revision: None

        :return: Info such as number of columns or column names in the provided data.
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
            training_schema_file = params["data_preparation"]["schema_training"]
            with open(training_schema_file, 'r') as f:
                dict1 = json.load(f)

            LengthOfDateStampInFile = dict1['LengthOfDateStampInFile']
            LengthOfTimeStampInFile = dict1['LengthOfTimeStampInFile']
            NumberOfColumns = dict1['NumberOfColumns']
            ColumnNames = dict1['ColumnNames']

            message = f"Length of datestamp and timestamp in the file: {LengthOfDateStampInFile} and {LengthOfTimeStampInFile}, number of columns in the data: {NumberOfColumns}"
            Logger().log(f, message)
            return LengthOfDateStampInFile, LengthOfTimeStampInFile, NumberOfColumns, ColumnNames

        except Exception as e:
            message1 = f"Exception occurred while fetching the info about the data from the schema_training json file. Exception: {str(e)}"
            Logger().log(f, message1)
            message2 = f"Fetching info about the data from the schema training json file unsuccessful"
            Logger().log(f, message2)
            raise e

        finally:
            f.close()

    def manualRegexCreation(self):
        """

        Description: This method is used for the creation of the regular expression which should be matched with the
        data file name provided by the client.

        Raises: Exception on failure

        Writen by: Shivam Shinde

        Version: 1.0
        Revision: None

        :return: Python regular expression

        """

        try:
            regex = r"Mushroom\_Data\_\d{8}\_\d{6}\.csv"
            return regex

        except Exception as e:
            raise e

    def createDirectoryForGoodAndBadRawData(self):
        """

        Description: This method is used to create directory for the good (data which passed the validation) and bad
        date (data which did not pass the validation)

        Raises: Exception on failure

        Writen by: Shivam Shinde

        Version: 1.0

        Revision: None

        """
        with open(os.path.join("config", "params.yaml")) as p:
            params = yaml.safe_load(p)

        main_log_dir = params['TrainingLogs']['main_log_dir']
        filename = params['TrainingLogs']['good_bad_data_dir_creation']

        if not os.path.exists(main_log_dir):
            os.makedirs(main_log_dir)

        whole_path = os.path.join(main_log_dir, filename)
        whole_path = pathlib.Path(whole_path)

        f = open(whole_path, 'a+')

        try:
            path = params['data_preparation']['good_validated_raw_dir']
            path = pathlib.Path(path)
            if not os.path.exists(path):
                os.makedirs(path)

            path = params['data_preparation']['bad_validated_raw_dir']
            path = pathlib.Path(path)
            if not os.path.exists(path):
                os.makedirs(path)

        except OSError as oe:
            message = f"Exception occurred while creating directories for the bad and good training validated data. Exception: {str(oe)}"
            Logger().log(f, message)
            raise oe

        finally:
            f.close()

    def deleteExistingGoodRawTrainingDataFolder(self):
        """

        Description: This method is used to delete the directory of the raw good (data which passed the validation)
        training data

        Raises: Exception on failure

        Writen By: Shivam Shinde

        Version: 1.0

        Revision: None

        """
        with open(os.path.join("config", "params.yaml")) as p:
            params = yaml.safe_load(p)

        main_log_dir = params['TrainingLogs']['main_log_dir']
        filename = params['TrainingLogs']['good_bad_data_dir_creation']

        if not os.path.exists(main_log_dir):
            os.makedirs(main_log_dir)

        whole_path = os.path.join(main_log_dir, filename)
        whole_path = pathlib.Path(whole_path)

        f = open(whole_path, "a+")

        try:
            path = params['data_preparation']['good_validated_raw_dir']
            path = pathlib.Path(path)
            if os.path.isdir(path):
                shutil.rmtree(path)

            message = "Deleted the folder for good raw training data successfully."
            Logger().log(f, message)

        except OSError as oe:
            message = f"Exception occurred while deleting the folder for good raw training data. Exception: {str(oe)}"
            Logger().log(f, message)
            raise oe

        finally:
            f.close()


    def deleteExistingBadRawTrainingDataFolder(self):
        """

        Description: This method is used to delete the directory of the raw bad (data which did not pass the
        validation) training data

        Raises: Exception on failure

        Writen By: Shivam Shinde

        Version: 1.0

        Revision: None

        """
        with open(os.path.join("config", "params.yaml")) as p:
            params = yaml.safe_load(p)

        main_log_dir = params['TrainingLogs']['main_log_dir']
        filename = params['TrainingLogs']['good_bad_data_dir_creation']

        if not os.path.exists(main_log_dir):
            os.makedirs(main_log_dir)

        whole_path = os.path.join(main_log_dir, filename)
        whole_path = pathlib.Path(whole_path)

        f = open(whole_path, 'a+')

        try:
            path = params['data_preparation']['bad_validated_raw_dir']
            path = pathlib.Path(path)
            if os.path.isdir(path):
                shutil.rmtree(path)

            message = "Deleted the folder for bad raw training data successfully."
            Logger().log(f, message)

        except OSError as oe:
            message = f"Exception occurred while deleting the folder for bad raw training data. Exception: {str(oe)}"
            Logger().log(f, message)
            raise oe

        finally:
            f.close()

    def moveBadDataFilesToArchievedBad(self):
        """

        Description: This method is used to move the bad training data files to the archived bad data folder.

        Raises: Exception on failure

        Writen By: Shivam Shinde

        Version: 1.0

        Revision: None

        """
        now = datetime.now()
        date = now.date()
        time = now.strftime("%H%M%S")

        with open(os.path.join("config", "params.yaml")) as p:
            params = yaml.safe_load(p)

        main_log_dir = params['TrainingLogs']['main_log_dir']
        filename = params['TrainingLogs']['good_bad_data_dir_creation']

        if not os.path.exists(main_log_dir):
            os.makedirs(main_log_dir)

        whole_path = os.path.join(main_log_dir, filename)
        whole_path = pathlib.Path(whole_path)

        l = open(whole_path, 'a+')
        try:
            source = params['data_preparation']['bad_validated_raw_dir']
            source = pathlib.Path(source)
            if os.path.isdir(source):

                path = params['data_preparation']['TrainingArchiveBadData']
                path = pathlib.Path(path)
                if not os.path.isdir(path):
                    os.makedirs(path)

                destination = path + str(date) + "_" + str(time)
                destination = pathlib.Path(destination)
                if not os.path.isdir(destination):
                    os.makedirs(destination)

                files = os.listdir(source)
                for f in files:
                    if f not in os.listdir(destination):
                        shutil.move(source + f, destination)

                Logger().log(l, "Bad data file were successfully moved from training raw data folder to the archived bad foler")

                if os.path.isdir(source):
                    shutil.rmtree(source)
                Logger().log(l, "Bad data folder from training raw data folder deleted successfully.")

        except OSError as oe:
            Logger().log(l, "Exception occurred while moving the bad data files from training raw data folder to the archived bad data folder")
            raise oe

        finally:
            f.close()

    def validateTrainingDataFileName(self, regex):
        """
        Description: This method is used to validate the data file name provided by the client.

        Written By: Shivam Shinde

        Version: 1.0

        Revision: None

        :param regex: This parameter is the regular expression that would be matched against the data file name
        provided by the client.

        :return: None

        """

        # Deleting the good and bad data folders in case previous execution failed
        self.deleteExistingGoodRawTrainingDataFolder()
        self.deleteExistingBadRawTrainingDataFolder()

        with open(os.path.join("config", "params.yaml")) as p:
            params = yaml.safe_load(p)        

        data_files_path = params['data_source']['batch_files']
        raw_data_files = [f for f in os.listdir(data_files_path)]

        main_log_dir = params['TrainingLogs']['main_log_dir']
        filename = params['TrainingLogs']['RawDataFileNameValidation']

        if not os.path.exists(main_log_dir):
            os.makedirs(main_log_dir)

        whole_path = os.path.join(main_log_dir, filename)
        whole_path = pathlib.Path(whole_path)

        f = open(whole_path, 'a+')

        try:
            good_data_path = params['data_preparation']['good_validated_raw_dir']
            good_data_path = pathlib.Path(good_data_path)

            bad_data_path = params['data_preparation']['bad_validated_raw_dir']
            bad_data_path = pathlib.Path(bad_data_path)

            # creating new directories for good and bad data
            self.createDirectoryForGoodAndBadRawData()

            for file in raw_data_files:

                if file.split(".")[-1] == "csv":
                    if re.match(regex, file):
                        shutil.copy(f"{os.path.join(data_files_path,file)}",
                                good_data_path)
                        Logger().log(f, "Valid file name! File moved to the good data folder")
                        Logger().log(f, "The validation for the data file name passed!")
                    else:
                        shutil.copy(f"{os.path.join(data_files_path,file)}",
                                    bad_data_path)
                        Logger().log(f, "Invalid file name! File moved to the bad data folder")

        except Exception as e:
            Logger().log(
                f, f"Exception occurred while validating the file name. Exception: {str(e)}")
            raise e

        finally:
            f.close()


    def validateNumberOfColumns(self, numColumns):
        """

        Description: This method is used to validate the number of columns in the data provided by the client.

        Written By: Shivam Shinde

        Version: 1.0

        Revision: None

        :param numColumns: This parameter is used to match the number of columns in the date.

        :return: None

        """

        with open(os.path.join("config", "params.yaml")) as p:
            params = yaml.safe_load(p)

        main_log_dir = params['TrainingLogs']['main_log_dir']
        filename = params['TrainingLogs']['numberOfColumnsValidation']

        if not os.path.exists(main_log_dir):
            os.makedirs(main_log_dir)

        whole_path = os.path.join(main_log_dir, filename)
        whole_path = pathlib.Path(whole_path)

        f = open(whole_path, 'a+')
        try:
            good_data_path = params['data_preparation']['good_validated_raw_dir']
            good_data_path = pathlib.Path(good_data_path)

            bad_data_path = params['data_preparation']['bad_validated_raw_dir']
            bad_data_path = pathlib.Path(bad_data_path)

            for file in os.listdir(good_data_path):
                csv_file = pd.read_csv(os.path.join(good_data_path, file))
                if csv_file.shape[1] == numColumns:
                    Logger().log(f, "Validation for number of columns in the data passed!")
                else:
                    Logger().log(f, "Validation for number of columns failed!")
                    Logger().log(f, "Moving the file to the bad data folder!")
                    shutil.move(os.path.join(
                        good_data_path, file), bad_data_path)
                    Logger().log(f, "File moved to the bad data folder!")

        except Exception as e:
            Logger().log(f, "Exception occurred while validating the number of columns in the data. Exception:" + str(e))
            raise e

        finally:
            f.close()


    def validateMissingValuesInWholeColumn(self):
        """

        Description: This method is used to check if any file in good data folder has any column with all the missing
        value.

        Written By: Shivam Shinde

        Version: 1.0

        Revision: None

        :return: None

        """
        with open(os.path.join("config", "params.yaml")) as p:
            params = yaml.safe_load(p)

        main_log_dir = params['TrainingLogs']['main_log_dir']
        filename = params['TrainingLogs']['columnWithAllMissingValuesValidation']

        if not os.path.exists(main_log_dir):
            os.makedirs(main_log_dir)

        whole_path = os.path.join(main_log_dir, filename)
        whole_path = pathlib.Path(whole_path)

        f = open(whole_path, 'a+')

        try:
            good_data_path = params['data_preparation']['good_validated_raw_dir']
            good_data_path = pathlib.Path(good_data_path)
            
            bad_data_path = params['data_preparation']['bad_validated_raw_dir']
            bad_data_path = pathlib.Path(bad_data_path)

            for file in os.listdir(good_data_path):
                csv_file = pd.read_csv(os.path.join(good_data_path, file))
                columns = csv_file.columns
                for column in columns:
                    noOfMissingValues = csv_file[column].isnull().sum()
                    if noOfMissingValues == csv_file.shape[0]:
                        Logger().log(f, "Columns with all missing values validation failed for the  file:" + str(file))
                        Logger().log(f, "Moving the file " + file +
                                     " from good data folder to bad data folder")
                        shutil.move(os.path.join(
                            good_data_path, file), bad_data_path)
                        Logger().log(f, 'Moved the file ' + str(file) +
                                     'from good data folder to bad data folder')
                        break
                    else:
                        pass
                Logger().log(
                    f, f"There are no columns with all missing values in the file {file} provided by the client")

        except Exception as e:
            Logger().log(f, "Exception occurred in the validation of files with columns with all the missing values. Exception: " + str(e))
            raise e

        finally:
            f.close()


if __name__ == '__main__':

    try:

        # creating a object of a class
        obj = rawDataValidation()

        # getting the pattern of the raw data filename
        raw_data_filename_pattern = obj.manualRegexCreation()

        # validating the raw data filename
        obj.validateTrainingDataFileName(raw_data_filename_pattern)

        # validating the number of columns in the raw data file
        obj.validateNumberOfColumns(23)

        # checking the any of the raw data file contain any column with all missing values
        obj.validateMissingValuesInWholeColumn()

    except Exception as e:
        raise e
