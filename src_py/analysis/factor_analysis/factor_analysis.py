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
        # classification_pred = classification_pred[classification_pred.index.isin(self._actual_for_forward_test.index)]
        classification_pred = classification_pred[:-1]
        before_jump_actual = self._actual_for_forward_test['before_jump_actual']
        after_jump_actual = self._actual_for_forward_test['after_jump_actual']
        p_and_l_when_predict_up = after_jump_actual - before_jump_actual
        p_and_l_when_predict_down = before_jump_actual - after_jump_actual
        daily_net_profit = pd.Series(np.where(classification_pred == 0, p_and_l_when_predict_down, 0.0))
        daily_net_profit = pd.Series(np.where(classification_pred == 2, p_and_l_when_predict_up, daily_net_profit))
        
        net_profit = self.net_profit(daily_net_profit)
        profit_factor = self.profit_factor(daily_net_profit)
        win_ratio = self.win_ratio(daily_net_profit)
        average_winner = self.average_winner(daily_net_profit)
        average_loser = self.average_loser(daily_net_profit)
        expected_value = self.expected_value(daily_net_profit)
        expectation = self.expectation(daily_net_profit)
        biggest_winner = self.biggest_winner(daily_net_profit)
        biggest_loser = self.biggest_loser(daily_net_profit)
        winning_streak = self.winning_streak(daily_net_profit)
        pass

    def net_profit(self, daily_net_profit):
        return sum(daily_net_profit)

    def profit_factor(self, daily_net_profit):
        gross_profits = sum(pd.Series(np.where(daily_net_profit >= 0.0, daily_net_profit, 0.0)))
        gross_loss = 0 - sum(pd.Series(np.where(daily_net_profit < 0.0, daily_net_profit, 0.0)))
        return gross_profits / gross_loss

    def win_ratio(self, daily_net_profit):
        num_win = sum(pd.Series(np.where(daily_net_profit >= 0.0, 1.0, 0.0)))
        num_lose = sum(pd.Series(np.where(daily_net_profit < 0.0, 1.0, 0.0)))
        return num_win / (num_lose + num_win)

    def average_winner(self, daily_net_profit):
        gross_profits = sum(pd.Series(np.where(daily_net_profit > 0.0, daily_net_profit, 0.0)))
        num_win = sum(pd.Series(np.where(daily_net_profit >= 0.0, 1.0, 0.0)))
        return gross_profits / num_win

    def average_loser(self, daily_net_profit):
        gross_loss = 0 - sum(pd.Series(np.where(daily_net_profit < 0.0, daily_net_profit, 0.0)))
        num_lose = sum(pd.Series(np.where(daily_net_profit < 0.0, 1.0, 0.0)))
        return gross_loss / num_lose

    def expected_value(self, daily_net_profit):
        win_ratio = self.win_ratio(daily_net_profit)
        lose_ratio = 1.0 - win_ratio
        average_winner = self.average_winner(daily_net_profit)
        average_loser = self.average_loser(daily_net_profit)
        return (win_ratio * average_winner) - (lose_ratio * average_loser)

    def expectation(self, daily_net_profit):
        expected_value = self.expected_value(daily_net_profit)
        average_loser = self.average_loser(daily_net_profit)
        return expected_value / average_loser

    def biggest_winner(self, daily_net_profit):
        return max(daily_net_profit)

    def biggest_loser(self, daily_net_profit):
        return min(daily_net_profit)

    def winning_streak(self, daily_net_profit):
        wins = pd.Series(np.where(daily_net_profit >= 0.0, 1.0, 0.0))
        wins_shift = wins.shift(-1)
        wins_start = pd.Series(np.where(wins_shift == 0.0, wins, 0.0))
        wins_streak_id = wins_start.cumsum()
        wins_streak = wins_streak_id.groupby(wins_streak_id).cumcount() + 1
        return max(wins_streak)

    def losing_streak(self, daily_net_profit):
        loses_start = daily_net_profit.ne(daily_net_profit.shift())
        wins_streak_id = wins_start.cumsum()
        wins_streak = wins_streak_id.cumcount() + 1
        return max(wins_streak)
    def filter_parameters(self, df, coef_list):
        # overwritten by inheritance
        return df
