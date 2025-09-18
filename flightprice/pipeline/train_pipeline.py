import os, sys
from flightprice.exception.exception import FlightException
from flightprice.logging.logger import logging

from flightprice.components.data_ingestion import DataIngestion
from flightprice.components.data_validation import DataValidation
from flightprice.components.data_transformation import DataTransformation
from flightprice.components.model_trainer import ModelTrainer

from flightprice.entity.config_entity import(
    TrainingPipelineConfig,
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig
)

from flightprice.entity.artifact_entity import(
    DataIngestionArtifact,
    DataValidationArtifact,
    DataTransformationArtifact,
    ModelTrainerArtifact
)

class TrainPipeline:
    def __init__(self):
        try:
            self.training_pipeline_config = TrainingPipelineConfig()
        except Exception as e:
            raise FlightException(e, sys)

    def start_data_ingestion(self):
        try:
            self.data_ingestion_config = DataIngestionConfig(training_pipeline_config=self.training_pipeline_config)
            logging.info("Starting Data Ingestion")
            data_ingestion = DataIngestion(data_ingestion_config=self.data_ingestion_config)
            logging.info("Initiating Data Ingestion")
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            logging.info("Completed Data Ingestion")
            return data_ingestion_artifact
        except Exception as e:
            raise FlightException(e, sys)

    def start_data_validation(self, data_ingestion_artifact: DataIngestionArtifact):
        try:
            self.data_validation_config = DataValidationConfig(training_pipeline_config=self.training_pipeline_config)
            logging.info("Starting Data Validation")
            data_validation = DataValidation(data_validation_config=self.data_validation_config,
                                             data_ingestion_artifact=data_ingestion_artifact)
            logging.info("Initiating Data Validation")
            data_validation_artifact = data_validation.initiate_data_validation()
            logging.info("Completed Data Validation")
            return data_validation_artifact
        except Exception as e:
            raise FlightException(e, sys)

    def start_data_transformation(self, data_validation_artifact: DataValidationArtifact):
        try:
            data_transformation_config = DataTransformationConfig(training_pipeline_config=self.training_pipeline_config)
            logging.info("Starting Data Transformation")
            data_transformation = DataTransformation(data_transformation_config=data_transformation_config,
                                                     data_validation_artifact=data_validation_artifact)
            logging.info("Initiating Data Transformation")
            data_transformation_artifact = data_transformation.initiate_data_transformation()
            logging.info("Completed Data Transformation")
            return data_transformation_artifact
        except Exception as e:
            raise FlightException(e, sys)

    def start_model_trainer(self, data_transformation_artifact: DataTransformationArtifact):
        try:
            model_trainer_config = ModelTrainerConfig(training_pipeline_config=self.training_pipeline_config)
            logging.info("Starting Model Training")
            model_trainer = ModelTrainer(model_trainer_config=model_trainer_config,
                                         data_transformation_artifact=data_transformation_artifact)
            logging.info("Initiating Model Training")
            model_trainer_artifact = model_trainer.initiate_model_training()
            logging.info("Completed Model Training")
            return model_trainer_artifact
        except Exception as e:
            raise FlightException(e, sys)

    def run_pipeline(self):
        try:
            data_ingestion_artifact = self.start_data_ingestion()
            data_validation_artifact = self.start_data_validation(data_ingestion_artifact=data_ingestion_artifact)
            data_transformation_artifact = self.start_data_transformation(data_validation_artifact=data_validation_artifact)
            model_trainer_artifact = self.start_model_trainer(data_transformation_artifact=data_transformation_artifact)
            return model_trainer_artifact
        except Exception as e:
            raise FlightException(e, sys)

if __name__ == "__main__":
    try:
        train_pipeline = TrainPipeline()
        train_pipeline.run_pipeline()
    except Exception as e:
        raise FlightException(e, sys)
