import numpy as np
import pandas as pd

from src_py.data_retriever_and_processor.data_preprocessor_factor_analysis import split_on_daily_basis
from src_py.data_retriever_and_processor.data_reader import retrieve_data, calculate_log_return


def prepare_daily_inputs(df):
    df_wap = df.copy()[['Wap', 'datetime']]
    df_wap_log_return = calculate_log_return(df_wap, column_names=['Wap'])
    df_daily_marked = split_on_daily_basis(df_wap_log_return)
    df_time_marked = mark_time_from_datetime(df_daily_marked)
    df_time_marked = df_time_marked.replace([np.inf, -np.inf], 0.0)
    df_pivoted = df_time_marked.pivot(index='date', columns='time', values='Wap_return')
    return df_pivoted


def mark_time_from_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """
    converting the time series data to discrete based on date
    :param df:
    :return:
    """
    df['time'] = df['datetime'].map(lambda x: x.time())
    return df


if __name__ == "__main__":
    df_data_retrieved = retrieve_data()
    df_daily_inputs = prepare_daily_inputs(df_data_retrieved)
    print(df_daily_inputs)
