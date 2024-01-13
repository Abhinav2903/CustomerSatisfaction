import logging
from abc import ABC, abstractmethod
from sklearn.linear_model import LinearRegression
import optuna
class Model(ABC):
    """Abstrac class for all models
    """
    @abstractmethod
    def train(self, X_train,y_train):
        """ Trains the model
        Args:
            X_train:Trainng data
            y_train: Training labels
        Returns:
            None    
        """
        pass

    @abstractmethod
    def optimize(self, trial, x_train, y_train, x_test, y_test):
        """
        Optimizes the hyperparameters of the model.

        Args:
            trial: Optuna trial object
            x_train: Training data
            y_train: Target data
            x_test: Testing data
            y_test: Testing target
        """
        pass
class LinearRegressionModel(Model):
    """Linear Regressin Model"""

    def train(self,X_train,y_train,**kwargs):
        """Trains the model
            Args:
                X_train: Trainng data
                y_train: Trainning labels
        """    
        try:
            reg = LinearRegression(**kwargs)
            reg.fit(X_train,y_train)
            logging.info("Model Training completed")
            return reg
        except Exception as e:
            logging.info("Excpetion in training emodel:{}".format(e))
            raise e
        
    def optimize(self, trial, x_train, y_train, x_test, y_test):
        n_estimators = trial.suggest_int("n_estimators", 1, 200)
        max_depth = trial.suggest_int("max_depth", 1, 20)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
        reg = self.train(x_train, y_train, n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split)
        return reg.score(x_test, y_test)

class HyperparameterTuner:
    """
    Class for performing hyperparameter tuning. It uses Model strategy to perform tuning.
    """

    def __init__(self, model, x_train, y_train, x_test, y_test):
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def optimize(self, n_trials=100):
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: self.model.optimize(trial, self.x_train, self.y_train, self.x_test, self.y_test), n_trials=n_trials)
        return study.best_trial.params

    
        