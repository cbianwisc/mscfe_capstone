from src_py.analysis.factor_analysis.factor_analysis import FactorAnalysis
from src_py.data_retriever_and_processor.data_preprocessor_time_series import prepare_daily_inputs


class TimeSeriesAnalysis(FactorAnalysis):
    def get_input_data(self):
        df_daily_inputs = prepare_daily_inputs(self._retrieved_data)
        self._input_data = df_daily_inputs
