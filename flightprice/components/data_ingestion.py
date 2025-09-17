from flightprice.entity.config_entity import DataIngestionConfig
from flightprice.entity.artifact_entity import DataIngestionArtifact
from flightprice.exception.exception import FlightException
from flightprice.logging.logger import logging

import os, sys
import pandas as pd
import numpy as np
import pymongo
import certifi
from sklearn.model_selection import train_test_split

MONGO_CLIENT = "mongodb+srv://AtharvaAI:#Atharva07@cluster0.tlufxez.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
ca = certifi.where()

class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config
            self.client = pymongo.MongoClient(MONGO_CLIENT, tlsCAFile=ca)
        except Exception as e:
            raise FlightException(e, sys)

    def export_collection_as_dataframe(self) -> pd.DataFrame:
        try:
            logging.info("Exporting collection data as dataframe")
            database_name = self.data_ingestion_config.database_name
            collection_name = self.data_ingestion_config.collection_name
            logging.info(f"Database name: {database_name} and Collection name: {collection_name}")
            df = pd.DataFrame(list(self.client[database_name][collection_name].find()))
            logging.info(f"Exported {df.shape[0]} rows and {df.shape[1]} columns data from collection: {collection_name} of database: {database_name}")
            if "_id" in df.columns:
                logging.info("Dropping _id column")
                df = df.drop(columns=["_id"], axis=1)
            return df
        except Exception as e:
            raise FlightException(e, sys)

    def export_data_into_feature_store(self, df: pd.DataFrame):
        try:
            feature_store_file_path = self.data_ingestion_config.feature_store_file_path
            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path, exist_ok=True)
            df.to_csv(feature_store_file_path, index=False, header=True)
            logging.info(f"Exported data into feature store at path: {feature_store_file_path}")
        except Exception as e:
            raise FlightException(e, sys)

    def split_data_as_train_test(self, df: pd.DataFrame):
        try:
            train_set, test_set = train_test_split(
                df,
                test_size=self.data_ingestion_config.train_test_split_ratio,
            )
            logging.info(f"Split data into train and test set")
            logging.info(f"Exited split_data_as_train_test method of DataIngestion class")

            train_dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)
            os.makedirs(train_dir_path, exist_ok=True)
            train_set.to_csv(self.data_ingestion_config.training_file_path, index=False, header=True)

            test_dir_path = os.path.dirname(self.data_ingestion_config.testing_file_path)
            os.makedirs(test_dir_path, exist_ok=True)
            test_set.to_csv(self.data_ingestion_config.testing_file_path, index=False, header=True)
            logging.info(f"Exported train and test file at path: {self.data_ingestion_config.training_file_path} and {self.data_ingestion_config.testing_file_path}")
        except Exception as e:
            raise FlightException(e, sys)

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        try:
            dataframe = self.export_collection_as_dataframe()
            logging.info("Exported collection data as dataframe")
            self.export_data_into_feature_store(dataframe)
            logging.info("Exported data into feature store")
            self.split_data_as_train_test(dataframe)
            logging.info("Split data into train and test set")
            data_ingestion_artifact = DataIngestionArtifact(
                train_file_path=self.data_ingestion_config.training_file_path,
                test_file_path=self.data_ingestion_config.testing_file_path,
            )
            logging.info(f"Data ingestion artifact created: {data_ingestion_artifact}")
            return data_ingestion_artifact
        except Exception as e:
            raise FlightException(e, sys)
