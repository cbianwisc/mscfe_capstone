import math
import pandas as pd
import os
import datetime


def read_data(raw_csv_dir: str) -> pd.DataFrame:
    """
    read raw data from csv file
    :param raw_csv_dir: directory of the csv file
    :return: the csv file content as dataframe
    """
    df = pd.read_csv(raw_csv_dir)
    return df


def read_time_stamp(df: pd.DataFrame) -> pd.DataFrame:
    """
    conver the time column from time stamp strings to datetime
    :param df: df with time stamp strings
    :return: df with datetime
    """
    series_orig_time_stamp = df['Time']
    series_time_stamp = series_orig_time_stamp.map(lambda x: x.rstrip(' US/Eastern'))
    series_datetime = series_time_stamp.map(lambda x: datetime.datetime.strptime(x, '%Y%m%d %H:%M:%S'))
    df['datetime'] = series_datetime
    return df


def sort_by_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    sort the dataframe by datetime column
    :param df: unsorted df with datetime
    :return: sorted df
    """
    return df.sort_values(by=['datetime']).copy(deep=True).reset_index()


def pick_column(df: pd.DataFrame, column_name=None) -> pd.DataFrame:
    """
    pick only columns needed
    :param df: df with multiple columns
    :param column_name: default to the column Close if not provided
    :return: df with only datetime and columns in the column name list
    """
    if column_name is None:
        column_name = ['Close']

    column_name = ['datetime'] + column_name
    return df[column_name]


def calculate_log_return(df: pd.DataFrame, column_name=None) -> pd.DataFrame:
    """
    calculate logarithmic return, aka continuously compounded return
    :param df:
    :param column_name: use all df columns if not provided
    :return: df with log return of each column
    """
    if column_name is None:
        column_name = df.columns.tolist()

    for col in column_name:
        if col == 'datetime':
            continue
        series_curr = df[column_name]
        series_last = series_curr.shift(1)
        series_divided = series_curr.divide(series_last, fill_value=0.0)
        df[column_name + '_return'] = math.log(series_divided)
    return df


def time_splitting(df: pd.DataFrame, time:datetime.time) -> pd.DataFrame:
    """
    converting the
    :param df:
    :param time:
    :return:
    """
    return df



if __name__ == "__main__":
    curr_dir = os.getcwd()
    home_dir = os.path.dirname(curr_dir)
    datasets_dir = home_dir + '\\Datasets'
    one_yr_data_dir = datasets_dir + '\\1YearOfData'
    one_yr_5m_data_dir = one_yr_data_dir + '\\QQQ_1Y_m05_TRADES.csv'
    df_raw_data = read_data(one_yr_5m_data_dir)
    df_time_fixed = read_time_stamp(df_raw_data)
    df_time_sorted = sort_by_time(df_time_fixed)
    df_close_price = pick_column(df_time_sorted)
    df_close_return = calculate_log_return(df_close_price)
    print(df_close_return)


