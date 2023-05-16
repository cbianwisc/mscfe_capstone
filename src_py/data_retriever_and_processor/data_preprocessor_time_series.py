import numpy as np
import pandas as pd

from src_py.data_retriever_and_processor.data_preprocessor_factor_analysis import split_on_daily_basis
from src_py.data_retriever_and_processor.data_reader import retrieve_data, calculate_log_return
from src_py.data_retriever_and_processor.input_retriever import standardize_data


def prepare_daily_inputs(df):
    df_wap = df.copy()[['Wap', 'datetime']]
    df_wap_log_return = calculate_log_return(df_wap, column_names=['Wap'])
    df_daily_marked = split_on_daily_basis(df_wap_log_return)
    df_time_marked = mark_time_from_datetime(df_daily_marked)
    df_time_marked = df_time_marked.replace([np.inf, -np.inf], 0.0)
    df_pivoted = df_time_marked.pivot(index='date', columns='time', values='Wap_return')
    df_pivoted = df_pivoted.fillna(0.0)
    # pick only full hour columns, ie. drop columns like 9:20 am
    # except within the last 2 hours, ie. keep columns like 6:20 pm
    columns_list_in_datetime = df_pivoted.columns.tolist()
    columns_to_use = [x for x in columns_list_in_datetime if (x.minute == 0 or x.hour >= 18)]
    df_pivoted_shortlisted = df_pivoted[columns_to_use]
    df_pivoted_shortlisted.columns = ['QQQ_' + str(x) for x in df_pivoted_shortlisted.columns.tolist()]
    df_standardized = standardize_data(df_pivoted_shortlisted)
    return df_standardized


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
