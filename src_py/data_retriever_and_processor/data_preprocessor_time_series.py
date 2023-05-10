import pandas as pd

from src_py.data_retriever_and_processor.data_preprocessor_factor_analysis import split_on_daily_basis
from src_py.data_retriever_and_processor.data_reader import retrieve_data


def prepare_daily_inputs(df):
    df_daily_marked = split_on_daily_basis(df)
    df_time_marked = mark_time_from_datetime(df_daily_marked)
    df_wap = df_time_marked.copy()[['Wap', 'time', 'date']]
    df_pivoted = df_wap.pivot(index='date', columns='time', values='Wap')
    df_pivoted['date'] = df_pivoted.index
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
