import numpy as np

# scaler
class MeanScaler:
    def fit_transform(self, y):
        self.mean = np.mean(y)
        return y / self.mean
    
    def inverse_transform(self, y):
        return y * self.mean

    def transform(self, y):
        return y / self.mean


# metric
def MAPEval(y_pred, y_true):
    y_true = np.array(y_true).ravel() + 1e-4
    y_pred = np.array(y_pred).ravel()    
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100