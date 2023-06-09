import pandas as pd
from sklearn import svm
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src_py.analysis.combined_analysis import CombinedAnalysis

COEFFICIENT_SIGNIFICANT_CUTOFF = 0.01


class SvmRegressionAnalysis(CombinedAnalysis):
    def train_model(self):
        x_train = self._data_for_train.copy().drop(columns=['overnight_jump'])
        y_train = self._data_for_train.copy()['overnight_jump']
        raw_reg = svm.SVR(kernel='linear').fit(x_train, y_train)
        self._raw_coefficient = raw_reg.coef_
        if all((abs(self._raw_coefficient) <= COEFFICIENT_SIGNIFICANT_CUTOFF).tolist()[0]):
            print("None of the factors is significant: ", self._raw_coefficient)
            exit()

        x_train_significant = self.filter_parameters(x_train, self._raw_coefficient[0])
        self._model = svm.SVR(kernel='linear').fit(x_train_significant, y_train)
        df_coef = pd.DataFrame({
            'Factor': x_train_significant.columns,
            'Coefficient': pd.Series(self._model.coef_[0])
        })
        print(df_coef)

    def filter_parameters(self, df: pd.DataFrame, coefficient_list: []) -> pd.DataFrame:
        column_list = df.columns
        coefficient_significant = column_list[abs(coefficient_list) > COEFFICIENT_SIGNIFICANT_CUTOFF]
        return df.copy()[coefficient_significant]

    def back_test_model(self):
        x_validate = self._data_for_back_test.copy().drop(columns=['overnight_jump'])
        y_validate = self._data_for_back_test.copy()['overnight_jump']
        x_validate_filtered = self.filter_parameters(x_validate, self._raw_coefficient[0])
        r_score = self._model.score(x_validate_filtered, y_validate)

        y_pred = self._model.predict(x_validate_filtered)
        mae = mean_absolute_error(y_true=y_validate, y_pred=y_pred)
        rmse = mean_squared_error(y_true=y_validate, y_pred=y_pred, squared=False)
        print('R-score: ', r_score, 'MAE: ', mae, 'RMSE: ', rmse)


if __name__ == "__main__":
    analysis = SvmRegressionAnalysis()
    analysis.run_analysis()

