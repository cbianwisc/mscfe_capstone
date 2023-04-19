import pandas as pd
from sklearn.model_selection import KFold

from src_py.data_preprocessor_factor_analysis import split_on_daily_basis
from src_py.main_analysis import MainAnalysis


class FactorAnalysis(MainAnalysis):
    def __init__(self):
        super().__init__()
        self._input_data = pd.DataFrame()
        self._output_data = pd.DataFrame()
        self._combined_data = pd.DataFrame()

    def get_input_data(self):
        pass

    def preprocess_data(self):
        self._output_data = split_on_daily_basis(self._retrieved_data)
        self._combined_data = self._input_data.merge(self._output_data, on=['date'])

    def divide_data(self):
        kf = KFold(n_splits=10, random_state=233, shuffle=True)
        for X_train, X_test in kf.split(self._combined_data):
           df_train = X_train
           df_test = X_test

    def train_model(self):
        pass

    def validate_model(self):
        pass

    def test_model(self):
        pass