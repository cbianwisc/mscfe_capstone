from keras import Sequential
from keras.layers import Dropout, Dense, Activation, LSTM

from src_py.analysis.time_series_analysis.rnn_regression_analysis import RnnRegressionAnalysis


class LstmRegressionAnalysis(RnnRegressionAnalysis):
    def model_construction(self, input_shape):
        regressor = Sequential()
        regressor.add(LSTM(units=50, return_sequences=False, input_shape=(input_shape, 1)))
        regressor.add(Dropout(0.2))
        regressor.add(Dense(units=1))
        regressor.add(Activation("linear"))
        regressor.compile(optimizer='adam', loss='mean_squared_error')
        return regressor


if __name__ == "__main__":
    analysis = LstmRegressionAnalysis()
    analysis.run_analysis()
