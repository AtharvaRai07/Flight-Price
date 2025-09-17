from flightprice.components.data_ingestion import DataIngestion
from flightprice.entity.config_entity import DataIngestionConfig, TrainingPipelineConfig

import os, sys
from flightprice.exception.exception import FlightException
from flightprice.logging.logger import logging


if __name__ == "__main__":
    try:
        logging.info("Starting data ingestion")
        training_pipeline_config = TrainingPipelineConfig()
        data_ingestion_config = DataIngestionConfig(training_pipeline_config=training_pipeline_config)
        data_ingestion_obj = DataIngestion(data_ingestion_config=data_ingestion_config)

        logging.info("Initiating data ingestion")

        data_ingestion_artifact = data_ingestion_obj.initiate_data_ingestion()
        print(data_ingestion_artifact)

    except Exception as e:
        raise FlightException(e, sys)

