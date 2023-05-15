import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, precision_score
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, Dropout, Activation

from src_py.analysis.combined_analysis import CombinedAnalysis
from src_py.data_retriever_and_processor.classifier import classify_output_data


class RnnClassificationAnalysis(CombinedAnalysis):
    def __init__(self):
        super().__init__()
        self._model = None

    def train_model(self):
        x_train = self._data_for_train.copy().drop(columns=['overnight_jump'])
        x_train = x_train.fillna(0.0)
        y_train = classify_output_data(self._data_for_train.copy())['overnight_jump']

        class_weight = self.calculate_class_weight(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
        regressor = self.model_construction(x_train.shape[1])
        print(regressor.summary())
        regressor.fit(x_train, y_train, epochs=100, batch_size=300, class_weight=class_weight)

        self._model = regressor

    def calculate_class_weight(self, y_train):
        weight_0 = 1.0 / sum(y_train == 0)
        weight_1 = 1.0 / sum(y_train == 1)
        weight_2 = 1.0 / sum(y_train == 2)
        return {0: weight_0, 1: weight_1, 2: weight_2}

    def model_construction(self, input_shape):
        regressor = Sequential()
        regressor.add(SimpleRNN(units=16, return_sequences=False, input_shape=(input_shape, 1)))
        regressor.add(Dropout(0.2))
        regressor.add(Dense(units=8, activation="relu"))
        regressor.add(Dense(units=3, activation="softmax"))
        regressor.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
        return regressor

    def back_test_model(self):
        x_validate = self._data_for_back_test.copy().drop(columns=['overnight_jump'])
        y_validate = classify_output_data(self._data_for_back_test.copy())['overnight_jump']
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
