from keras import Sequential
from keras.layers import Dropout, Dense, Activation, LSTM

from src_py.analysis.time_series_analysis.classification.rnn_classification_analysis import RnnClassificationAnalysis


class LstmClassificationAnalysis(RnnClassificationAnalysis):
    def model_construction(self, input_shape):
        regressor = Sequential()
        regressor.add(LSTM(units=16, return_sequences=False, input_shape=(input_shape, 1)))
        regressor.add(Dropout(0.2))
        regressor.add(Dense(units=8, activation="relu"))
        regressor.add(Dense(units=3, activation="softmax"))
        regressor.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
        return regressor


if __name__ == "__main__":
    analysis = LstmClassificationAnalysis()
    analysis.run_analysis()
