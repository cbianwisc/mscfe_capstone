from sklearn.metrics import balanced_accuracy_score, precision_score
from sklearn.neural_network import MLPClassifier

from src_py.analysis.combined_analysis import CombinedAnalysis
from src_py.analysis.factor_analysis.factor_analysis import FactorAnalysis
from src_py.data_retriever_and_processor.classifier import classify_output_data

COEFFICIENT_SIGNIFICANT_CUTOFF = 0.5


class NeuralNetworkClassificationAnalysis(CombinedAnalysis):
    def __init__(self):
        super().__init__()
        self._raw_coefficient = None
        self._model = None

    def train_model(self):
        x_train = self._data_for_train.copy().drop(columns=['overnight_jump'])
        y_train = classify_output_data(self._data_for_train.copy())['overnight_jump']
        clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15,), random_state=1)
        clf.fit(x_train, y_train)
        self._model = clf

    def back_test_model(self):
        x_validate = self._data_for_back_test.copy().drop(columns=['overnight_jump'])
        y_validate = classify_output_data(self._data_for_back_test.copy())['overnight_jump']
        score = self._model.score(x_validate, y_validate)

        y_pred = self._model.predict(x_validate)
        accuracy_score_ = balanced_accuracy_score(y_validate, y_pred)
        precision_score_ = precision_score(y_validate, y_pred, average='weighted')
        print('Score: ', score, 'accuracy_score: ', accuracy_score_, 'precision_score: ', precision_score_)


if __name__ == "__main__":
    analysis = NeuralNetworkClassificationAnalysis()
    analysis.run_analysis()

