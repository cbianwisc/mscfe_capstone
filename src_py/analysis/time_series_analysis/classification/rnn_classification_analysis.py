import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, precision_score
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, Dropout, Activation

from src_py.analysis.time_series_analysis.time_series_analysis import TimeSeriesAnalysis
from src_py.data_retriever_and_processor.classifier import classify_output_data


class RnnClassificationAnalysis(TimeSeriesAnalysis):
    def __init__(self):
        super().__init__()
        self._model = None

    def train_model(self):
        x_train = self._data_for_train.copy().drop(columns=['overnight_jump'])
        x_train = x_train.fillna(0.0)
        y_train = classify_output_data(self._data_for_train.copy())['overnight_jump']

        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
        regressor = self.model_construction(x_train.shape[1])
        regressor.fit(x_train, y_train, epochs=100, batch_size=100)
        print(regressor.summary())
        self._model = regressor

    def model_construction(self, input_shape):
        regressor = Sequential()
        regressor.add(SimpleRNN(units=50, return_sequences=False, input_shape=(input_shape, 1)))
        regressor.add(Dropout(0.2))
        regressor.add(Dense(units=3))
        regressor.add(Activation("softmax"))
        regressor.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return regressor

    def validate_model(self):
        x_validate = self._data_for_validate.copy().drop(columns=['overnight_jump'])
        y_validate = classify_output_data(self._data_for_validate.copy())['overnight_jump']
        x_validate = x_validate.fillna(0.0)
        x_validate = np.reshape(x_validate, (x_validate.shape[0], x_validate.shape[1], 1))

        y_pred_raw = self._model.predict(x_validate)
        df_pred_raw = pd.DataFrame(y_pred_raw, columns=[0, 1, 2])
        y_pred = df_pred_raw.idxmax(axis=1)
        accuracy_score_ = balanced_accuracy_score(y_validate, y_pred)
        precision_score_ = precision_score(y_validate, y_pred, average='weighted')
        print('accuracy_score: ', accuracy_score_, 'precision_score: ', precision_score_)


if __name__ == "__main__":
    analysis = RnnClassificationAnalysis()
    analysis.run_analysis()
