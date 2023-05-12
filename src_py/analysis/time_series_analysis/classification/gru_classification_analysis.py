from keras import Sequential
from keras.layers import Dropout, Dense, Activation, GRU

from src_py.analysis.time_series_analysis.classification.rnn_classification_analysis import RnnClassificationAnalysis


class GruClassificationAnalysis(RnnClassificationAnalysis):
    def model_construction(self, input_shape):
        regressor = Sequential()
        regressor.add(GRU(units=50, return_sequences=False, input_shape=(input_shape, 1)))
        regressor.add(Dropout(0.2))
        regressor.add(Dense(units=3))
        regressor.add(Activation("softmax"))
        regressor.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return regressor


if __name__ == "__main__":
    analysis = GruClassificationAnalysis()
    analysis.run_analysis()
