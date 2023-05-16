import pandas as pd
from sklearn import svm
from sklearn.metrics import balanced_accuracy_score, precision_score

from src_py.analysis.combined_analysis import CombinedAnalysis
from src_py.analysis.factor_analysis.factor_analysis import FactorAnalysis
from src_py.data_retriever_and_processor.classifier import classify_output_data

COEFFICIENT_SIGNIFICANT_CUTOFF = 0.5


class SvmClassificationAnalysis(CombinedAnalysis):
    def train_model(self):
        x_train = self._data_for_train.copy().drop(columns=['overnight_jump'])
        y_train = classify_output_data(self._data_for_train.copy())['overnight_jump']
        raw_reg = svm.SVC(kernel='linear', class_weight='balanced').fit(x_train, y_train)
        self._raw_coefficient = raw_reg.coef_
        if all((abs(self._raw_coefficient) <= COEFFICIENT_SIGNIFICANT_CUTOFF).tolist()[0]):
            print("None of the factors is significant: ", self._raw_coefficient)
            exit()

        x_train_significant = self.filter_parameters(x_train, self._raw_coefficient[0])
        self._model = svm.SVC(kernel='linear', class_weight='balanced').fit(x_train_significant, y_train)
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
        y_validate = classify_output_data(self._data_for_back_test.copy())['overnight_jump']
        x_validate_filtered = self.filter_parameters(x_validate, self._raw_coefficient[0])
        score = self._model.score(x_validate_filtered, y_validate)

        y_pred = self._model.predict(x_validate_filtered)
        accuracy_score_ = balanced_accuracy_score(y_validate, y_pred)
        precision_score_ = precision_score(y_validate, y_pred, average='weighted')
        print('Score: ', score, 'accuracy_score: ', accuracy_score_, 'precision_score: ', precision_score_)


if __name__ == "__main__":
    analysis = SvmClassificationAnalysis()
    analysis.run_analysis()

