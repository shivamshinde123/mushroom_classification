from email import header
import os

import pandas as pd
import yaml
from performLogging import Logger
import pathlib


class RawPredictionDataTransformation:

    """

    Description: This class includes the methods which transforms the data so that there won't be any exception while
    dumping the data into the database.

    Written By: Shivam Shinde

    Version: 1.0

    Revision: None

    """

    def __init__(self):
        pass

    def addingQuotesToStringColumns(self):
        """

        Description: This method is used to add the quotes to the columns which contains the string values. This is
        done so that there won't be any exception while adding this data to the database

        Written By: Shivam Shinde
        Version: 1.0

        Revision: None

        :return: None

        """

        with open(os.path.join("config", "params.yaml")) as p:
            params = yaml.safe_load(p)

        main_log_dir = params['PredictionLogs']['main_log_dir']
        filename = params['PredictionLogs']['RawPredictionDataTransformation']

        if not os.path.exists(main_log_dir):
            os.makedirs(main_log_dir)

        whole_path = os.path.join(main_log_dir, filename)
        whole_path = pathlib.Path(whole_path)

        f = open(whole_path, 'a+')
        try:
            good_data_path = params['pred_data_preparation']['good_validated_raw_dir']
            good_data_path = pathlib.Path(good_data_path)

            for file in os.listdir(good_data_path):
                csv_file = pd.read_csv(os.path.join(good_data_path, file))

                column_lst = [
                    column for column in csv_file.columns if csv_file[column].dtypes == 'O']

                for column in column_lst:
                    csv_file[column] = csv_file[column].apply(
                        lambda a: "'" + str(a) + "'")

                csv_file.to_csv(os.path.join(
                    good_data_path, file), index=None, header=True)
                Logger().log(
                    f, 'Quotes added successfully to the values of columns having string values')

        except Exception as e:
            Logger().log(
                f, "Exception occurred while adding the quotes to the values in the columns having string values. Exception: " + str(e))
            raise e

        finally:
            f.close()

    def removeHyphenFromColumnName(self):
        """

        Description: This method is used to remove the hyphen from the column names of data.

        Written By:  Shivam Shinde

        Version: 1.0

        Revision: None

        :return: None

        """
        with open(os.path.join("config", "params.yaml")) as p:
            params = yaml.safe_load(p)

        main_log_dir = params['PredictionLogs']['main_log_dir']
        filename = params['PredictionLogs']['RawPredictionDataTransformation']

        if not os.path.exists(main_log_dir):
            os.makedirs(main_log_dir)

        whole_path = os.path.join(main_log_dir, filename)
        whole_path = pathlib.Path(whole_path)

        f = open(whole_path, 'a+')
        try:
            good_data_path = params['pred_data_preparation']['good_validated_raw_dir']
            good_data_path = pathlib.Path(good_data_path)

            for file in os.listdir(good_data_path): 
                if file.split('.')[-1] == "csv":
                    csv_file = pd.read_csv(os.path.join(good_data_path, file))

                    columns = csv_file.columns

                    for column in columns:
                        if '-' in str(column):
                            new_column = column.replace('-','_')
                            csv_file.rename(columns={column:new_column},inplace=True)
                    csv_file.to_csv(os.path.join(good_data_path,file),header=True,index=None)
                    Logger().log(
                        f, "Removed the hyphens from the column names successfully")
        except Exception as e:
            Logger().log(
                f, "Exception occurred while removing the hyphens from the column names. Exception: "+str(e))
            raise e

        finally:
            f.close()


if __name__ == '__main__':

    try:
        # creating object of the class
        obj = RawPredictionDataTransformation()

        # adding quotes to the string values from the dataframe
        obj.addingQuotesToStringColumns()

        # removing hyphens from the column names of the dataframe
        obj.removeHyphenFromColumnName()

    except Exception as e:
        raise e
