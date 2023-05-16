from abc import ABC, abstractmethod

import pandas as pd

from src_py.data_retriever_and_processor import data_reader

YEARS_OF_DATA = 5
FORWARD_TEST_DAYS = 50
ALMOST_ZERO_THRESHOLD = 0.01


class MainAnalysis(ABC):
    """
    The class serves as the main analysis structure, intended by corresponding analysis, including factor analysis and
    time series analysis
    """
    def __init__(self):
        self._retrieved_data = pd.DataFrame()
        self._data_for_train = pd.DataFrame()
        self._data_for_back_test = pd.DataFrame()
        self._data_for_forward_test = pd.DataFrame()
        self._model = None
        self._raw_coefficient = [0]

    def retrieve_data(self):
        """
        retrieve data, currently set to point to an existing csv file in repo
        """
        self._retrieved_data = data_reader.retrieve_data(YEARS_OF_DATA)

    @abstractmethod
    def preprocess_data(self):
        pass

    @abstractmethod
    def divide_data(self):
        pass

    @abstractmethod
    def train_model(self):
        pass

    @abstractmethod
    def back_test_model(self):
        pass

    @abstractmethod
    def forward_test_model(self):
        pass

    def run_analysis(self):
        self.retrieve_data()
        self.preprocess_data()
        self.divide_data()
        self.train_model()
        self.back_test_model()
        self.forward_test_model()
