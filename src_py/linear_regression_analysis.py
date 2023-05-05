import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src_py.factor_analysis import FactorAnalysis

COEFFICIENT_SIGNIFICANT_CUTOFF = 0.05


class LinearRegressionAnalysis(FactorAnalysis):
    def __init__(self):
        super().__init__()
        self._raw_coefficient = None
        self._model = None

    def train_model(self):
        x_train = self._data_for_train.copy().drop(columns=['overnight_jump'])
        y_train = self._data_for_train.copy()['overnight_jump']
        raw_reg = LinearRegression().fit(x_train, y_train)
        self._raw_coefficient = raw_reg.coef_
        x_train_significant = self.filter_parameters(x_train, self._raw_coefficient)
        self._model = LinearRegression().fit(x_train_significant, y_train)
        # print(main_reg.score(x_train_significant, y_train))

    def filter_parameters(self, df: pd.DataFrame, coefficient_list: []) -> pd.DataFrame:
        column_list = df.columns
        coefficient_significant = column_list[abs(coefficient_list) > COEFFICIENT_SIGNIFICANT_CUTOFF]
        return df.copy()[coefficient_significant]

    def validate_model(self):
        x_validate = self._data_for_validate.copy().drop(columns=['overnight_jump'])
        y_validate = self._data_for_validate.copy()['overnight_jump']
        x_validate_filtered = self.filter_parameters(x_validate, self._raw_coefficient)
        r_score = self._model.score(x_validate_filtered, y_validate)

        y_pred = self._model.predict(x_validate_filtered)
        mae = mean_absolute_error(y_true=y_validate, y_pred=y_pred)
        rmse = mean_squared_error(y_true=y_validate, y_pred=y_pred, squared=False)
        print('R-score: ', r_score, 'MAE: ', mae, 'RMSE: ', rmse)


if __name__ == "__main__":
    analysis = LinearRegressionAnalysis()
    analysis.run_analysis()

