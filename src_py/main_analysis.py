from abc import ABC, abstractmethod

import pandas as pd

from src_py import data_reader


class MainAnalysis(ABC):
    """
    The class serves as the main analysis structure, intended by corresponding analysis, including factor analysis and
    time series analysis
    """
    def __init__(self):
        self._retrieved_data = pd.DataFrame()

    def retrieve_data(self):
        """
        retrieve data, currently set to point to an existing csv file in repo
        """
        self._retrieved_data = data_reader.retrieve_data()

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
    def validate_model(self):
        pass

    @abstractmethod
    def test_model(self):
        pass


