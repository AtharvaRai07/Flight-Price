from flightprice.components.data_ingestion import DataIngestion
from flightprice.components.data_validation import DataValidation
from flightprice.components.data_transformation import DataTransformation

from flightprice.entity.config_entity import TrainingPipelineConfig
from flightprice.entity.config_entity import DataIngestionConfig
from flightprice.entity.config_entity import DataValidationConfig
from flightprice.entity.config_entity import DataTransformationConfig

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

        logging.info("Completed data ingestion")

        logging.info("Starting data validation")
        data_validation_config = DataValidationConfig(training_pipeline_config=training_pipeline_config)
        data_validation_obj = DataValidation(data_validation_config=data_validation_config,
                                             data_ingestion_artifact=data_ingestion_artifact)

        logging.info("Initiating data validation")
        data_validation_artifact = data_validation_obj.initiate_data_validation()
        print(data_validation_artifact)
        logging.info("Completed data validation")

        logging.info("Stating data transformation")
        data_transformation_config = DataTransformationConfig(training_pipeline_config=training_pipeline_config)
        data_tranformation_obj = DataTransformation(data_transformation_config=data_transformation_config,
                                                    data_validation_artifact=data_validation_artifact)

        logging.info("Initiating data transformation")
        data_transformation_artifact = data_tranformation_obj.initiate_data_transformation()
        print(data_transformation_artifact)
        logging.info("Completed data transformation")

    except Exception as e:
        raise FlightException(e, sys)

