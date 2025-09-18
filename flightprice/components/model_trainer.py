import os, sys
import mlflow
from flightprice.exception.exception import FlightException
from flightprice.logging.logger import logging

from flightprice.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact
from flightprice.entity.config_entity import ModelTrainerConfig

from flightprice.utils.main_utils.utils import save_object, load_object, evaluate_models
from flightprice.utils.main_utils.utils import load_numpy_array_data
from flightprice.utils.ml_utils.metric.regression_metric import get_regression_score
from flightprice.utils.ml_utils.model.estimator import FlightModel

from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge,Lasso
from catboost import CatBoostRegressor
# from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import StratifiedKFold

class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig,
                 data_transformation_artifact: DataTransformationArtifact):

        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise FlightException(e, sys)

    def track_mlflow_experiment(self,best_model, classification_train_metric, classification_test_metric):
        try:
            with mlflow.start_run():
                mlflow.set_tag("Best_Model_Name", best_model)

                #Training metrics
                mlflow.log_metric("Train_R2_Score", classification_train_metric.r2_score)
                mlflow.log_metric("Train_MAE_Score", classification_train_metric.mae_score)
                mlflow.log_metric("Train_MSE_Score", classification_train_metric.mse_score)
                mlflow.log_metric("Train_RMSE_Score", classification_train_metric.rmse_score)

                #Testing metrics
                mlflow.log_metric("Test_R2_Score", classification_test_metric.r2_score)
                mlflow.log_metric("Test_MAE_Score", classification_test_metric.mae_score)
                mlflow.log_metric("Test_MSE_Score", classification_test_metric.mse_score)
                mlflow.log_metric("Test_RMSE_Score", classification_test_metric.rmse_score)

                mlflow.log_artifact(self.model_trainer_config.trained_model_file_path, artifact_path="pipeline")

        except Exception as e:
            raise FlightException(e, sys)

    def train_model(self, X_train, y_train, X_test, y_test) -> ModelTrainerArtifact:
        try:
            models = {
            "LinearRegression": LinearRegression(),
            "Ridge": Ridge(),
            "Lasso": Lasso(),
            "SVR": SVR(),
            "KNeighborsRegressor": KNeighborsRegressor(),
            "DecisionTreeRegressor": DecisionTreeRegressor(),
            "RandomForestRegressor": RandomForestRegressor(),
            "AdaBoostRegressor": AdaBoostRegressor(),
            # "XGBRegressor": XGBRegressor(),
            "CatBoostRegressor": CatBoostRegressor(verbose=0),
            "LGBMRegressor": LGBMRegressor()
            }

            params = {
                "LinearRegression": {},
                "Ridge": {},
                "Lasso": {},
                "SVR" : {},
                "KNeighborsRegressor": {},
                "DecisionTreeRegressor": {},
                "RandomForestRegressor": {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                },
                "AdaBoostRegressor": {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2, 0.3]
                },
                # # "XGBRegressor": {
                # #     # 'n_estimators': [100, 200, 300, 500, 1000],
                # #     # 'max_depth': [3, 4, 5, 6, 7, 8],
                # #     # 'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
                # #     # 'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
                # #     # 'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
                # #     # 'colsample_bylevel': [0.6, 0.7, 0.8, 0.9, 1.0],
                # #     # 'reg_alpha': [0, 0.1, 0.5, 1.0],
                # #     # 'reg_lambda': [0, 0.1, 0.5, 1.0],
                # #     # 'gamma': [0, 0.1, 0.2, 0.5],
                # #     # 'min_child_weight': [1, 3, 5, 7]
                # },
                "CatBoostRegressor": {
                    # 'iterations': [100, 200, 300, 500, 1000],
                    # 'depth': [3, 4, 5, 6, 7, 8, 10],
                    # 'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
                    # 'l2_leaf_reg': [1, 3, 5, 7, 9],
                    # 'border_count': [32, 64, 128, 255],
                    # 'bagging_temperature': [0, 0.5, 1.0],
                    # 'random_strength': [0, 1, 2, 3],
                    # 'od_type': ['IncToDec', 'Iter'],
                    # 'od_wait': [10, 20, 30, 50]
                },
                "LGBMRegressor": {
                    'n_estimators': [100, 200, 300, 500, 1000],
                    'max_depth': [3, 5, 7, 10, 15, -1],
                    'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2]
                }
            }

            model_report: dict = evaluate_models(X_train,y_train,X_test,y_test,models,params)
            logging.info(f"Model Report : {model_report}")

            best_model_name = max(model_report, key=model_report.get)
            best_model_score = model_report[best_model_name]

            logging.info(f"Best Model Found on both training and testing dataset is, Model Name : {best_model_name} , R2 Score : {best_model_score}")
            best_model = models[best_model_name]
            logging.info(f"Best Model Details : {best_model}")

            y_train_pred = best_model.predict(X_train)
            classification_train_metric = get_regression_score(y_train,y_train_pred)

            y_test_pred = best_model.predict(X_test)
            classification_test_metric = get_regression_score(y_test,y_test_pred)

            prepprocessor = load_object(self.data_transformation_artifact.transformed_object_file_path)

            model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
            os.makedirs(model_dir_path, exist_ok=True)

            Flight_Model = FlightModel(preprocessor=prepprocessor, model=best_model)
            save_object(self.model_trainer_config.trained_model_file_path, Flight_Model)
            logging.info(f"Trained model saved at : {self.model_trainer_config.trained_model_file_path}")

            save_object("final_model/model.pkl",obj=best_model)

            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                train_metric_artifact=classification_train_metric,
                test_metric_artifact=classification_test_metric
            )
            logging.info(f"Model Trainer Artifact : {model_trainer_artifact}")
            return model_trainer_artifact

        except Exception as e:
            raise FlightException(e, sys)


    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            logging.info("Loading transformed training and testing array")
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path

            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)

            X_train, y_train, X_test, y_test = (
                train_arr[:,:-1], train_arr[:,-1],
                test_arr[:,:-1], test_arr[:,-1]
            )

            model_trainer_artifact = self.train_model(X_train, y_train, X_test, y_test)
            return model_trainer_artifact
        except Exception as e:
            raise FlightException(e, sys)



