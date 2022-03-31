import csv
import os
import shutil
import sqlite3
from telnetlib import EC
import yaml
from logging import Logger


class DBOperations:

    """
    Description: This class contains the methods that deal with the database operations.
    Written By: Shivam Shinde 
    Version: 1.0
    Revision: None

    """

    def __init__(self):
        self.params = yaml.safe_load(
            open(os.path.join("config", "params.yaml")))

    def dbConnection(self):
        """
        Description: This method is used to create a connection with the sqlite3 database

        On Failure: Raises exception

        Written By: Shivam Shinde

        Version: 1.0

        Revision: None

        :return: Databases connector

        """
        path_for_log = self.params['TrainingLogs']['databaseLogs']

        if not os.path.exists(path_for_log):
            os.makedirs(path_for_log)

        f = open(path_for_log, 'a+')
        try:
            database_path = self.params['data_preparation']['training_db_dir']
            database_name = self.params['data_preparation']['training_db']

            if not os.path.exists(database_path):
                os.makedirs(self.path)

            conn = sqlite3.connect(os.path.join(database_path, database_name))
            Logger().log(
                f, f"Connection with the database {database_name} made")
            f.close()
            return conn

        except ConnectionError as ce:
            Logger().log(
                f, f"Connection error occurred while creating a connection to the database. Error: {str(ce)}")
            f.close()
            raise ce

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

        path_for_log = self.params['TrainingLogs']['databaseLogs']

        if not os.path.exists(path_for_log):
            os.makedirs(path_for_log)

        f = open(path_for_log, 'a+')

        conn = self.dbConnection()
        cursor = conn.cursor()

        cursor.execute(
            ''' SELECT count(name) FROM sqlite_master WHERE type='table' AND name='goodRawData' ''')

        try:
            if cursor.fetchone()[0] == 1:
                Logger().log(f, "Table named goodRawData created in the database goodRawDataDb")
                f.close()
                conn.close()

            else:
                for key in columnNamesDict.keys():
                    datatype = columnNamesDict[key]

                    # Here in try block we check if the table is existed or not and if it is then add the columns to it
                    # In catch block, we will create a table
                    try:
                        command = f"""ALTER TABLE goodRawData ADD {key} {datatype}"""
                        cursor.execute(command)
                    except:
                        command = f"""CREATE TABLE goodRawData ({key} {datatype})"""
                        cursor.execute(command)

                conn.commit()
                Logger().log(f, "Table named goodRawData created successfully in the database")
                f.close()
                conn.close()

        except Exception as e:
            Logger().log(
                f, f"Exception occurred while creating the table inside the database. Exception: {str(e)}")
            f.close()
            conn.close()
            raise e

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

        path_for_log = self.params['TrainingLogs']['databaseLogs']

        if not os.path.exists(path_for_log):
            os.makedirs(path_for_log)

        f = open(path_for_log, 'a+')

        conn = self.dbConnection()
        cursor = conn.cursor()

        good_data_path = self.params['data_preparation']['good_validated_raw_dir']
        bad_data_path = self.params['data_preparation']['bad_validated_raw_dir']

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
                                    f"INSERT INTO goodRawData values ({list_})")
                                conn.commit()
                            except Exception as e:
                                raise e

            except Exception as e:
                conn.rollback()
                filepath = os.path.join(good_data_path,file)
                shutil.move(filepath, bad_data_path)
                Logger().log(
                    f, f"Error occurred while inserting the data into the table. Exception: {str(e)}")
                f.close()
                conn.close()
                raise e

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
        path_for_log = self.params['TrainingLogs']['databaseLogs']

        if not os.path.exists(path_for_log):
            os.makedirs(path_for_log)

        f = open(path_for_log, 'a+')

        path_of_fileFromDb = self.params['data_preparation']['Training_FileFromDB']
        filename = self.params['data_preparation']['master_csv']

        try:
            conn = self.dbConnection()
            cursor = conn.cursor()

            query = "SELECT * FROM goodRawData"
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
            f.close()
            conn.close()

        except Exception as e:
            Logger().log(
                f, f"Exception occurred while exporting the data file from the database. Exception: {str(e)}")
            f.close()
            raise e



if __name__ == '__main__':

    try:
        # creating a object of a class 
        obj = DBOperations()

        # creating a database and the creating a table into it
        obj.createTableIntoDb()

        # adding the data from the file from good data foler into the database
        obj.insertGoodDataIntoTable()

        # exporting the data from the database as a csv file 
        obj.getDataFromDbTableIntoCSV()
    
    except Exception as e:
        raise e