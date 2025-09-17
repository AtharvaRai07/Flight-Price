import os, sys
import numpy as np
import pandas as pd

"""
Defining common constants for training pipeline
"""
TARGET_COLUMN = ""
PIPELINE_NAME: str = "FlightPrice"
ARTIFACT_DIR: str = "Artifacts"
FILE_NAME: str = "flight_data.csv"

MONGO_CLIENT = "mongodb+srv://AtharvaAI:#Atharva07@cluster0.tlufxez.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"

SCHEMA_FILE_PATH = os.path.join("data_schema", "schema.yaml")

SAVED_MODEL_DIR = os.path.join("saved_models")

MODEL_FILE_NAME = "model.pkl"

"""
Data Ingestion related constants start with DATA_INGESTION VAR NAME
"""

DATA_INGESTION_COLLECTION_NAME: str = "FlightData"
DATA_INGESTION_DATABASE_NAME: str = "AtharvaAI"
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.2


""""
Data Validataion related constant start with DATA_VALIDATION VAR NAME
"""

DATA_VALIDATION_DIR_NAME: str = "data_validation"
DATA_VALIDATION_VALID_DIR: str = "validated"
DATA_VALIDATION_INVALID_DIR: str = "invalid"
DATA_VALIDATION_DRIFT_REPORT_DIR: str = "drift_report"
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME: str = "report.yaml"
PREPROCESSING_OBJECT_FILE_NAME = "preprocessing.pkl"
