from flightprice.entity.config_entity import DataValidationConfig
from flightprice.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from flightprice.exception.exception import FlightException
from flightprice.logging.logger import logging
from flightprice.constants.training_pipeline import SCHEMA_FILE_PATH
from flightprice.utils.main_utils.utils import read_yaml_file, write_yaml_file

import os, sys
import pandas as pd
from scipy.stats import ks_2samp

class DataValidation:
    def __init__(self,data_ingestion_artifact: DataIngestionArtifact,
                 data_validation_config: DataValidationConfig,):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)

        except Exception as e:
            raise FlightException(e, sys)

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise FlightException(e, sys)

    def validate_number_of_columns(self, df: pd.DataFrame) -> bool:
        try:
            number_of_columns = len(self._schema_config['columns'])
            logging.info(f"Required number of columns: {number_of_columns}")
            logging.info(f"Dataframe has columns: {len(df.columns)}")
            if len(df.columns) == number_of_columns:
                return True
            return False

        except Exception as e:
            raise FlightException(e, sys)

    def detect_data_drift(self, base_df: pd.DataFrame, current_df: pd.DataFrame, threshold= 0.05) -> bool:
        try:
            status = True
            report = {}
            for column in base_df.columns:
                if base_df[column].dtype.kind not in 'biufc':
                    logging.info(f"Skipping non-numeric column: {column}")
                    continue
                logging.info(f"Detecting data drift for column: {column}")
                d1 = base_df[column]
                d2 = current_df[column]
                same_distribution = ks_2samp(d1, d2)
                if same_distribution.pvalue <= threshold:
                    logging.info(f"Data drift detected for column {column} (p-value: {same_distribution.pvalue})")
                    status = False
                    report.update({
                        column: {
                            "p_value": float(same_distribution.pvalue),
                            "drift_status": status
                        }
                    })
                else:
                    logging.info(f"No data drift detected for column {column} (p-value: {same_distribution.pvalue})")
                    report.update({
                        column: {
                            "p_value": float(same_distribution.pvalue),
                            "drift_status": False
                        }
                    })
            logging.info(f"Data drift detection report: {report}")
            drift_report_file_path = self.data_validation_config.drift_report_file_path
            dir_path = os.path.dirname(drift_report_file_path)
            os.makedirs(dir_path, exist_ok=True)

            logging.info(f"Saving drift report at: {drift_report_file_path}")
            write_yaml_file(drift_report_file_path, report)

        except Exception as e:
            raise FlightException(e, sys)



    def initiate_data_validation(self) -> DataValidationArtifact:
        try:

            train_file_path = self.data_ingestion_artifact.train_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path

            logging.info("Reading training and testing data")
            train_df = DataValidation.read_data(train_file_path)
            test_df = DataValidation.read_data(test_file_path)


            valid_train_dir_path = os.path.dirname(self.data_validation_config.valid_train_file_path)
            os.makedirs(valid_train_dir_path, exist_ok=True)
            valid_test_dir_path = os.path.dirname(self.data_validation_config.valid_test_file_path)
            os.makedirs(valid_test_dir_path, exist_ok=True)
            invalid_train_dir_path = os.path.dirname(self.data_validation_config.invalid_train_file_path)
            os.makedirs(invalid_train_dir_path, exist_ok=True)
            invalid_test_dir_path = os.path.dirname(self.data_validation_config.invalid_test_file_path)
            os.makedirs(invalid_test_dir_path, exist_ok=True)

            valid_train_file_path = None
            invalid_train_file_path = None
            if self.validate_number_of_columns(df=train_df):
                valid_train_file_path = self.data_validation_config.valid_train_file_path
                train_df.to_csv(valid_train_file_path, index=False)
            else:
                logging.error("Training data validation failed")
                invalid_train_file_path = self.data_validation_config.invalid_train_file_path
                train_df.to_csv(invalid_train_file_path, index=False)

            valid_test_file_path = None
            invalid_test_file_path = None
            if self.validate_number_of_columns(df=test_df):
                valid_test_file_path = self.data_validation_config.valid_test_file_path
                test_df.to_csv(valid_test_file_path, index=False)
            else:
                logging.error("Testing data validation failed")
                invalid_test_file_path = self.data_validation_config.invalid_test_file_path
                test_df.to_csv(invalid_test_file_path, index=False)

            validation_status = (valid_train_file_path is not None and valid_test_file_path is not None)
            if validation_status:
                logging.info("Data drift detection using KS test")
                self.detect_data_drift(base_df=train_df, current_df=test_df)

            data_validation_artifact = DataValidationArtifact(
                validation_status = validation_status,
                valid_train_file_path = valid_train_file_path,
                valid_test_file_path = valid_test_file_path,
                invalid_train_file_path = invalid_train_file_path,
                invalid_test_file_path = invalid_test_file_path,
                drift_report_file_path = self.data_validation_config.drift_report_file_path
            )

            return data_validation_artifact

        except Exception as e:
            raise FlightException(e, sys)

