from src_py.analysis.factor_analysis.factor_analysis import FactorAnalysis
from src_py.analysis.main_analysis import YEARS_OF_DATA
from src_py.data_retriever_and_processor.data_preprocessor_time_series import prepare_daily_inputs
from src_py.data_retriever_and_processor.data_reader_vix import prepare_daily_inputs_vix


class CombinedAnalysis(FactorAnalysis):
    def get_input_data(self):
        super().get_input_data()
        df_daily_inputs = prepare_daily_inputs(self._retrieved_data)
        self._input_data = self._input_data.merge(df_daily_inputs, left_index=True, right_index=True)
        df_vix = prepare_daily_inputs_vix(YEARS_OF_DATA)
        self._input_data = self._input_data.merge(df_vix, left_index=True, right_index=True)
        self._input_data = self._input_data.fillna(0.0)
