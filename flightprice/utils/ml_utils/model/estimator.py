import os, sys
from flightprice.constants.training_pipeline import SAVED_MODEL_DIR, MODEL_FILE_NAME
from flightprice.exception.exception import FlightException
from flightprice.logging.logger import logging

class FlightModel:
    def __init__(self, preprocessor, model):
        try:
            self.preprocessor = preprocessor
            self.model = model
        except Exception as e:
            raise FlightException(e, sys)

    def predict(self, x):
        try:
            x_transform = self.preprocessor.transform(x)
            logging.info("Used preprocessor to transform input data")
            y_hat = self.model.predict(x_transform)
            logging.info("Used model to make predictions")
            return y_hat
        except Exception as e:
            raise FlightException(e, sys)
