import os, sys
import json
import certifi
import pymongo
import pandas as pd
import numpy as np
from flightprice.logging.logger import logging
from flightprice.exception.exception import FlightException

ca = certifi.where()

class FlightDataExtract():
    def __init__(self, database, mongo_client, collection):
        try:
            self.database = database
            self.mongo_client = pymongo.MongoClient(mongo_client, tlsCAFile=ca)
            self.collection = collection
        except Exception as e:
            raise FlightException(e, sys)

    def csv_to_json_converter(self, file_path):
        try:
            data = pd.read_csv(file_path)
            data.reset_index(drop=True, inplace=True)
            records = list(json.loads(data.T.to_json()).values())
            return records
        except Exception as e:
            raise FlightException(e, sys)

    def insert_data_mongodb(self,records):
        try:
            self.records = records
            self.database = self.mongo_client[self.database]
            self.collection = self.database[self.collection]
            self.collection.insert_many(self.records)
            return len(self.records)
        except Exception as e:
            raise FlightException(e, sys)

if __name__ == "__main__":
    FILE_PATH = "Flight_Data/flight_data.csv"
    MONGO_CLIENT = "mongodb+srv://AtharvaAI:#Atharva07@cluster0.tlufxez.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
    DATABASE = "AtharvaAI"
    COLLECTION = "FlightData"

    flight_obj = FlightDataExtract(database=DATABASE,mongo_client=MONGO_CLIENT,collection=COLLECTION)
    records = flight_obj.csv_to_json_converter(file_path=FILE_PATH)
    no_of_records = flight_obj.insert_data_mongodb(records=records)
    print(no_of_records)

