import csv
import os
import shutil
import sqlite3
from telnetlib import EC
import yaml
from performLogging import Logger
import pathlib
import json

class PredictionDBOperations:

    """
    Description: This class contains the methods that deal with the database operations.
    Written By: Shivam Shinde 
    Version: 1.0
    Revision: None

    """

    def __init__(self):
        pass

    def dbConnection(self):
        """
        Description: This method is used to create a connection with the sqlite3 database

        On Failure: Raises exception

        Written By: Shivam Shinde

        Version: 1.0

        Revision: None

        :return: Databases connector

        """

        with open(os.path.join("config", "params.yaml")) as p:
            params = yaml.safe_load(p)

        main_log_dir = params['PredictionLogs']['main_log_dir']
        filename = params['PredictionLogs']['databaseLogs']

        if not os.path.exists(main_log_dir):
            os.makedirs(main_log_dir)

        whole_path = os.path.join(main_log_dir, filename)
        whole_path = pathlib.Path(whole_path)

        f = open(whole_path, 'a+')
        try:
            database_path = params['pred_data_preparation']['prediction_db_dir']
            database_name = params['pred_data_preparation']['prediction_db']

            if not os.path.exists(database_path):
                os.makedirs(database_path)

            conn = sqlite3.connect(os.path.join(database_path, database_name))
            Logger().log(
                f, f"Connection with the database {database_name} made")
            return conn

        except ConnectionError as ce:
            Logger().log(
                f, f"Connection error occurred while creating a connection to the database. Error: {str(ce)}")
            raise ce
        
        finally:
            f.close()


    def createTableIntoDb(self, columnNamesDict):
        """

        Description: This method is used create a table in the existing database

        Written By: Shivam Shinde

        On Failure: Raises exception

        Version: 1.0

        Revision: None

        :param databaseName: Name of the database into which table is to be added

        :param columnNamesDict: Dictionary having column names as keys and their datatype as values

        :return: None

        """

        with open(os.path.join("config", "params.yaml")) as p:
            params = yaml.safe_load(p)

        main_log_dir = params['PredictionLogs']['main_log_dir']
        filename = params['PredictionLogs']['databaseLogs']

        if not os.path.exists(main_log_dir):
            os.makedirs(main_log_dir)

        whole_path = os.path.join(main_log_dir, filename)
        whole_path = pathlib.Path(whole_path)

        f = open(whole_path, 'a+')

        conn = self.dbConnection()
        cursor = conn.cursor()

        table_name = params['pred_data_preparation']['table_name']

        cursor.execute(f"DROP TABLE IF EXISTS {table_name}")

        try:            
            for key in columnNamesDict.keys():
                datatype = columnNamesDict[key]

                # Here in try block we check if the table is existed or not and if it is then add the columns to it
                # In catch block, we will create a table
                try:
                    command = f"""ALTER TABLE {table_name} ADD {key} {datatype}"""
                    cursor.execute(command)
                except:
                    command = f"""CREATE TABLE IF NOT EXISTS {table_name} ({key} {datatype})"""
                    cursor.execute(command)

            conn.commit()
            Logger().log(f, f"Table named {table_name} created successfully in the database")
            conn.close()

        except Exception as e:
            Logger().log(
                f, f"Exception occurred while creating the table inside the database. Exception: {str(e)}")
            conn.close()
            raise e

        finally:
            f.close()

    
    def deleteDB(self):

        """
        Description: This method is used to delete the database
        On Failure: Raises exception
        Written By: Shivam Shinde
        Version: 1.0
        Revision: None
        :return: None
        """

        with open(os.path.join("config", "params.yaml")) as p:
            params = yaml.safe_load(p)

        main_log_dir = params['PredictionLogs']['main_log_dir']
        filename = params['PredictionLogs']['databaseLogs']

        if not os.path.exists(main_log_dir):
            os.makedirs(main_log_dir)

        whole_path = os.path.join(main_log_dir, filename)
        whole_path = pathlib.Path(whole_path)

        f = open(whole_path, 'a+')

        conn = self.dbConnection()
        cursor = conn.cursor()

        try:
            db_dir = params['pred_data_preparation']['prediction_db_dir']
            db_name = params['pred_data_preparation']['prediction_db']

            db_path = os.path.join(db_dir,db_name)
            db_path = pathlib.Path(db_path)

            query = f"DROP DATABASE {db_path}"

            cursor.execute(query)

        except Exception as e:
            conn.rollback()
            Logger().log(f, f"Exception occured while deleting the database. Exception: {str(e)}")

        finally:
            f.close()

            
    def insertGoodDataIntoTable(self):
        """

        Description: This method is used to add the data into the already created table.
        On Failure: Raises exception
        Written By: Shivam Shinde
        Version: 1.0
        Revision: None
        :param database: Name of the database into which the table is present.
        :return: None

        """

        with open(os.path.join("config", "params.yaml")) as p:
            params = yaml.safe_load(p)

        main_log_dir = params['PredictionLogs']['main_log_dir']
        filename = params['PredictionLogs']['databaseLogs']

        if not os.path.exists(main_log_dir):
            os.makedirs(main_log_dir)

        whole_path = os.path.join(main_log_dir, filename)
        whole_path = pathlib.Path(whole_path)

        f = open(whole_path, 'a+')

        conn = self.dbConnection()
        cursor = conn.cursor()

        good_data_path = params['pred_data_preparation']['good_validated_raw_dir']
        bad_data_path = params['pred_data_preparation']['bad_validated_raw_dir']

        table_name = params['pred_data_preparation']['table_name']

        for file in os.listdir(good_data_path):
            try:
                filepath = os.path.join(good_data_path,file)

                with open(filepath, 'r') as p:
                    next(p)
                    reader = csv.reader(p, delimiter='\n')
                    Logger().log(f, f"{file} File loaded successfully!!")
                    for line in enumerate(reader):
                        for list_ in line[1]:
                            try:
                                cursor.execute(
                                    f"INSERT INTO {table_name} values ({list_})")
                                conn.commit()
                            except Exception as e:
                                raise e

            except Exception as e:
                conn.rollback()
                filepath = os.path.join(good_data_path,file)
                shutil.move(filepath, bad_data_path)
                Logger().log(
                    f, f"Error occurred while inserting the data into the table. Exception: {str(e)}")
                raise e

            
            finally:
                f.close()
                conn.close()


    def getDataFromDbTableIntoCSV(self):
        """

        Description: This method is used to fetch the data from the table inside the database and store it as csv file into some directory.
        Written By: Shivam Shinde
        Version: 1.0
        Revision: None
        :param database: The name of the database into which the table is present.
        :return: None

        """
        with open(os.path.join("config", "params.yaml")) as p:
            params = yaml.safe_load(p)

        main_log_dir = params['PredictionLogs']['main_log_dir']
        filename = params['PredictionLogs']['databaseLogs']

        if not os.path.exists(main_log_dir):
            os.makedirs(main_log_dir)

        whole_path = os.path.join(main_log_dir, filename)
        whole_path = pathlib.Path(whole_path)

        f = open(whole_path, 'a+')

        path_of_fileFromDb = params['pred_data_preparation']['Prediction_FileFromDB']
        filename = params['pred_data_preparation']['master_csv']

        table_name = params['pred_data_preparation']['table_name']

        try:
            conn = self.dbConnection()
            cursor = conn.cursor()

            query = f"SELECT * FROM {table_name}"
            cursor.execute(query)

            results = cursor.fetchall()

            headers = [i[0] for i in cursor.description]

            # Make the CSV output directory
            if not os.path.exists(path_of_fileFromDb):
                os.makedirs(path_of_fileFromDb)

            # Open CSV file for writing.
            csvfile = csv.writer(open(os.path.join(path_of_fileFromDb,filename), 'w', newline=''), delimiter=',',
                                 lineterminator='\r\n', quoting=csv.QUOTE_ALL, escapechar='\\')

            # Add the headers and data to the CSV file.
            csvfile.writerow(headers)
            csvfile.writerows(results)

            Logger().log(f, "File exported successfully!!")
            conn.close()

        except Exception as e:
            Logger().log(
                f, f"Exception occurred while exporting the data file from the database. Exception: {str(e)}")
            raise e
        
        finally:
            f.close()



if __name__ == '__main__':

    try:
        # creating a object of a class 
        obj = PredictionDBOperations()

        # deleting the previously created table from the database
        obj.deleteDB()

        # getting the dictonary containing the name and datatypes of the columns
        with open(os.path.join("config", "params.yaml")) as p:
            params = yaml.safe_load(p)
        
        schema_prediction_path = params['pred_data_preparation']['schema_prediction']

        with open(schema_prediction_path, "r") as k:
            json_file = json.load(k)
            name_dtype_dict = json_file['ColumnNames']

        # creating a database and the creating a table into it
        obj.createTableIntoDb(name_dtype_dict)

        # adding the data from the file from good data foler into the database
        obj.insertGoodDataIntoTable()

        # exporting the data from the database as a csv file 
        obj.getDataFromDbTableIntoCSV()
    
    except Exception as e:
        raise e