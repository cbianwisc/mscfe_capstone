from src_py.data_retriever_and_processor.input_retriever import retrieve_input_data
from src_py.analysis.factor_analysis.regression.linear_regression_analysis import LinearRegressionAnalysis
from src_py.data_retriever_and_processor.pca_processor import pca_analysis


class LinearRegressionAnalysisAfterPca(LinearRegressionAnalysis):
    def get_input_data(self):
        super().get_input_data()
        self._input_data = pca_analysis(self._input_data)


if __name__ == "__main__":
    analysis = LinearRegressionAnalysisAfterPca()
    analysis.run_analysis()
