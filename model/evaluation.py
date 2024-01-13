import logging
from abc import ABC, abstractmethod

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

class Evaluation(ABC):
    """Abstract Class definng the starategy of the evaluation method"""

    @abstractmethod
    def calculate_score(self,y_true:np.ndarray,y_pred:np.ndarray) -> float:
        pass

class MSE(Evaluation):

    """MSE evaluation"""

    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        try:
            logging.info("Enterd the calculate score method")
            mse = mean_squared_error(y_true,y_pred)
            logging.info("The mean squared erro value is "+str(mse))
            return mse
        except Exception as e:
            logging.error(
                "Exception occured in calculate_score method of the MSE class"+str(e)
            )

            raise e
        
class R2Score(Evaluation):
    """R2Score evaluation"""

    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        try:
            logging.info("Enterd the calculate score method")
            r2 = r2_score(y_true, y_pred)
            logging.info("The r2 score value is: " + str(r2))
            return r2
        except Exception as e:
            logging.error(
                "Exception occured in calculate_score method of the R2 class"+str(e)
            )

            raise e

class RMSE(Evaluation):
    """RMSE evaluation"""

    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        try:
            logging.info("Entered the calculate_score method of the RMSE class")
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            logging.info("The root mean squared error value is: " + str(rmse))
            return rmse
        except Exception as e:
            logging.error(
                "Exception occured in calculate_score method of the RMSE class"+str(e)
            )

            raise e