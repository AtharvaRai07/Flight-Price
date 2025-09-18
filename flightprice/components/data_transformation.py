import os, sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from flightprice.entity.config_entity import DataTransformationConfig
from flightprice.entity.artifact_entity import DataTransformationArtifact, DataValidationArtifact
from flightprice.exception.exception import FlightException
from flightprice.logging.logger import logging

from flightprice.constants.training_pipeline import TARGET_COLUMN
from flightprice.constants.training_pipeline import DATA_TRANSFORMATION_OHE_PARAMS
from flightprice.constants.training_pipeline import DATA_TRANSFORMATION_PREPROCESSOR_OBJECT_FILE_PATH
from flightprice.utils.main_utils.utils import save_numpy_array_data,save_object


class DataTransformation:
    def __init__(self, data_transformation_config: DataTransformationConfig,
                 data_validation_artifact: DataValidationArtifact):
        try:
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact
        except Exception as e:
            raise FlightException(e, sys)

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise FlightException(e, sys)

    def create_preprocessor_object(self):
        logging.info("Doing one hot encoding for categorical columns and standard scaling for numerical columns")
        try:

            # One-Hot Encoder for categorical variables
            ohe: OneHotEncoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
            logging.info(f"OneHotEncoder object: {ohe}")

            # Standard Scaler for feature scaling
            scaler: StandardScaler = StandardScaler()
            logging.info(f"StandardScaler object: {scaler}")

            # Create preprocessing pipeline
            preprocessor: Pipeline = Pipeline(steps=[
                ('ohe', ohe),
                ('scaler', scaler)
            ])

            logging.info("Data transformation pipeline created successfully")
            return preprocessor
        except Exception as e:
            raise FlightException(e, sys)

    def initiate_data_transformation(self,) -> DataTransformationArtifact:
        try:
            train_df = DataTransformation.read_data(self.data_validation_artifact.valid_train_file_path)
            test_df = DataTransformation.read_data(self.data_validation_artifact.valid_test_file_path)

            logging.info("Splitting input and target feature from training and testing dataframe.")
            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN],axis=1)
            target_feature_train_df = train_df[TARGET_COLUMN]

            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN],axis=1)
            target_feature_test_df = test_df[TARGET_COLUMN]

            preprocessor = self.create_preprocessor_object()
            preprocessor_obj = preprocessor.fit(input_feature_train_df)

            transformed_input_train_feature = preprocessor_obj.transform(input_feature_train_df)
            transformed_input_test_feature = preprocessor_obj.transform(input_feature_test_df)

            train_arr = np.c_[transformed_input_train_feature, np.array(target_feature_train_df)]
            test_arr = np.c_[transformed_input_test_feature, np.array(target_feature_test_df)]

            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path,train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path,test_arr)
            save_object(self.data_transformation_config.transformed_object_file_path,preprocessor_obj)

            save_object(DATA_TRANSFORMATION_PREPROCESSOR_OBJECT_FILE_PATH,preprocessor_obj)

            data_transformation_artifact = DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )

            return data_transformation_artifact

        except Exception as e:
            raise FlightException(e, sys)

