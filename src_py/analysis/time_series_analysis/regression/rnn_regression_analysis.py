import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, Dropout, Activation

from src_py.analysis.combined_analysis import CombinedAnalysis


class RnnRegressionAnalysis(CombinedAnalysis):
    def __init__(self):
        super().__init__()
        self._model = None

    def train_model(self):
        x_train = self._data_for_train.copy().drop(columns=['overnight_jump'])
        y_train = self._data_for_train.copy()['overnight_jump']
        x_train = x_train.fillna(0.0)
        y_train = y_train.fillna(0.0)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
        regressor = self.model_construction(x_train.shape[1])
        regressor.fit(x_train, y_train, epochs=100, batch_size=100)
        print(regressor.summary())
        self._model = regressor

    def model_construction(self, input_shape):
        regressor = Sequential()
        regressor.add(SimpleRNN(units=50, return_sequences=False, input_shape=(input_shape, 1)))
        regressor.add(Dropout(0.2))
        regressor.add(Dense(units=1))
        regressor.add(Activation("linear"))
        regressor.compile(optimizer='adam', loss='mean_squared_error')
        return regressor

    def back_test_model(self):
        x_validate = self._data_for_back_test.copy().drop(columns=['overnight_jump'])
        y_validate = self._data_for_back_test.copy()['overnight_jump']
        x_validate = x_validate.fillna(0.0)
        y_validate = y_validate.fillna(0.0)
        x_validate = np.reshape(x_validate, (x_validate.shape[0], x_validate.shape[1], 1))

        y_pred = self._model.predict(x_validate)
        mae = mean_absolute_error(y_true=y_validate, y_pred=y_pred)
        rmse = mean_squared_error(y_true=y_validate, y_pred=y_pred, squared=False)
        print('MAE: ', mae, 'RMSE: ', rmse)


if __name__ == "__main__":
    analysis = RnnRegressionAnalysis()
    analysis.run_analysis()
