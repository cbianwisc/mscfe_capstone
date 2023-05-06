import pandas as pd
from sklearn.model_selection import KFold

from src_py.data_retriever_and_processor.data_preprocessor_factor_analysis import generate_output_data
from src_py.data_retriever_and_processor.input_retriever import retrieve_input_data
from src_py.analysis.main_analysis import MainAnalysis


class FactorAnalysis(MainAnalysis):
    def __init__(self):
        super().__init__()
        self._input_data = pd.DataFrame()
        self._output_data = pd.DataFrame()
        self._combined_data = pd.DataFrame()

    def get_input_data(self):
        self._input_data = retrieve_input_data()

    def preprocess_data(self):
        self.get_input_data()
        self._output_data = generate_output_data(self._retrieved_data)
        self._combined_data = self._input_data.merge(self._output_data, left_index=True, right_index=True)

    def divide_data(self):
        kf = KFold(n_splits=10, random_state=233, shuffle=True)
        for X_train, X_validate in kf.split(self._combined_data):
            self._data_for_train = self._combined_data.iloc[X_train, :]
            self._data_for_validate = self._combined_data.iloc[X_validate, :]

    def train_model(self):
        pass

    def validate_model(self):
        pass

    def back_test_model(self):
        pass
