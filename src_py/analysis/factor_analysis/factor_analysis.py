import math

import numpy as np
import pandas as pd
from pandas._libs.tslibs.offsets import BDay
from sklearn.model_selection import KFold

from src_py.data_retriever_and_processor.data_preprocessor_factor_analysis import generate_output_data
from src_py.data_retriever_and_processor.input_retriever import retrieve_input_data
from src_py.analysis.main_analysis import MainAnalysis, FORWARD_TEST_DAYS, ALMOST_ZERO_THRESHOLD


class FactorAnalysis(MainAnalysis):
    def __init__(self):
        super().__init__()
        self._actual_for_forward_test = None
        self._input_data = pd.DataFrame()
        self._output_data = pd.DataFrame()
        self._combined_data = pd.DataFrame()

    def get_input_data(self):
        start_date = self._retrieved_data['datetime'].iloc[0].date() - BDay(1)
        end_date = self._retrieved_data['datetime'].iloc[-1].date()
        self._input_data = retrieve_input_data(start_date, end_date)

    def preprocess_data(self):
        self.get_input_data()
        self._output_data = generate_output_data(self._retrieved_data)
        self._combined_data = self._input_data.merge(self._output_data, left_index=True, right_index=True)

    def divide_data(self):
        self.separate_forward_test_data_out()
        kf = KFold(n_splits=10, random_state=233, shuffle=True)
        for X_train, X_validate in kf.split(self._combined_data):
            self._data_for_train = self._combined_data.iloc[X_train, :]
            self._data_for_back_test = self._combined_data.iloc[X_validate, :]

    def separate_forward_test_data_out(self):
        end_date = self._retrieved_data['datetime'].iloc[-1].date()
        forward_test_start_date = (end_date - BDay(FORWARD_TEST_DAYS)).date()
        input_forward_test = self._input_data.loc[self._input_data.index >= forward_test_start_date].copy()
        self._data_for_forward_test = input_forward_test
        self._actual_for_forward_test = self.get_actual_prices_forward_test(forward_test_start_date)
        self._combined_data = self._combined_data.loc[self._combined_data.index < forward_test_start_date]

    def get_actual_prices_forward_test(self, start_date):
        df_before_jump = self.get_price_before_jump_forward_test(start_date)[:-1]
        df_after_jump = self.get_price_after_jump_forward_test(start_date)[:-1]
        df_combined = df_before_jump.merge(df_after_jump, left_index=True, right_index=True)
        return df_combined

    def get_price_before_jump_forward_test(self, start_date):
        df_during_forward_test_date = self._retrieved_data.loc[self._retrieved_data['date'] >= start_date]
        series_price_before_jump = df_during_forward_test_date.groupby('date').last()['Open']
        df_price_before_jump = pd.DataFrame(series_price_before_jump)
        df_price_before_jump.columns = ['before_jump_actual']
        return df_price_before_jump

    def get_price_after_jump_forward_test(self, start_date):
        df_during_forward_test_date = self._retrieved_data.loc[self._retrieved_data['date'] >= start_date]
        series_price_after_jump = df_during_forward_test_date.groupby('date').first()['Close']
        series_price_after_jump = series_price_after_jump.shift(-1)
        df_price_after_jump = pd.DataFrame(series_price_after_jump)
        df_price_after_jump.columns = ['after_jump_actual']
        return df_price_after_jump

    def train_model(self):
        pass

    def back_test_model(self):
        pass

    def forward_test_model(self):
        df_input_for_forward_test = self.filter_parameters(self._data_for_forward_test, self._raw_coefficient[0])
        y_forward_test_predict = self._model.predict(df_input_for_forward_test)
        if len(y_forward_test_predict.shape) == 1:
            if np.issubdtype(y_forward_test_predict[0], np.integer):
                # classification -- sparse
                self.forward_test_classification(y_forward_test_predict)
                pass
            else:
                # regression
                pass
        else:
            # classification -- one-hot
            df_pred_raw = pd.DataFrame(y_forward_test_predict, columns=[0, 1, 2])
            y_pred = df_pred_raw.idxmax(axis=1)
            self.forward_test_classification(y_pred)
            pass

    def forward_test_classification(self, classification_pred):
        before_jump_actual = self._actual_for_forward_test['before_jump_actual']
        after_jump_actual = self._actual_for_forward_test['after_jump_actual']
        p_and_l_when_predict_up = after_jump_actual - before_jump_actual
        p_and_l_when_predict_down = before_jump_actual - after_jump_actual
        net_profit = pd.Series(np.where(classification_pred == 0, p_and_l_when_predict_down, 0.0))
        net_profit = pd.Series(np.where(classification_pred == 2, p_and_l_when_predict_up, net_profit))

        print(sum(net_profit))
        pass

    def filter_parameters(self, df, coef_list):
        # overwritten by inheritance
        return df
