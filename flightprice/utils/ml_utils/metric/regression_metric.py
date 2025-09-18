import numpy as np
import sys
from flightprice.exception.exception import FlightException
from flightprice.logging.logger import logging
from flightprice.entity.artifact_entity import RegressionMetricArtifact
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def get_regression_score(y_true, y_pred) -> RegressionMetricArtifact:
    try:
        model_r2_score = r2_score(y_true, y_pred)
        model_mae = mean_absolute_error(y_true, y_pred)
        model_mse = mean_squared_error(y_true, y_pred)
        model_rmse = np.sqrt(model_mse)

        regression_metric = RegressionMetricArtifact(
            r2_score=model_r2_score,
            mae_score=model_mae,
            mse_score=model_mse,
            rmse_score=model_rmse
        )
        return regression_metric

    except Exception as e:
        raise FlightException(e, sys)
