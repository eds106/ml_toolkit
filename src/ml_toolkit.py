from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

# OOP approach needs to be reconsidered. Could be a module with useful helper functions, or a full OOP approach. Could incorporate both?
class MLToolkit:
    def __init__(self):
        self.model = None
        self.r_squared = None
        self.mse = None
        self.rmse = None
        self.mae = None


    def set_model(self, model):
        self.model = model
        return model


    def calculate_metrics(self, X_test, y_test, y_pred, model=self.model):
        r_squared = model.score(X_test, y_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)

        self.r_squared = r_squared
        self.mse = mse
        self.rmse = rmse
        self.mae = mae

        return r_squared, mse, rmse, mae
