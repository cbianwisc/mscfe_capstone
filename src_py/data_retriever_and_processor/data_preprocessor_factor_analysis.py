import pandas as pd
from src_py.data_retriever_and_processor.data_reader import retrieve_data


def split_on_daily_basis(df: pd.DataFrame) -> pd.DataFrame:
    """
    converting the time series data to discrete based on date
    :param df:
    :return:
    """
    df['date'] = df['datetime'].map(lambda x: x.date())
    return df


def calculate_overnight_jump(df: pd.DataFrame) -> pd.DataFrame:
    """
    calculate the overnight jump
    :param df:
    :return:
    """
    series_daily_first_open = df.groupby('date').first()['Open']
    series_daily_last_close = df.groupby('date').last()['Close']
    series_open = series_daily_first_open.iloc[1:]
    series_last_day_close = series_daily_last_close.shift(-1).iloc[:-1]
    series_overnight_jump = series_open / series_last_day_close
    series_index = series_overnight_jump.index.to_series()
    series_index_shifted = series_index.shift(1)
    df_overnight_jump = series_overnight_jump.to_frame(name='overnight_jump')
    df_overnight_jump_use_last_date = df_overnight_jump.set_index(series_index_shifted)[1:-1]
    return df_overnight_jump_use_last_date


def generate_output_data(df_data: pd.DataFrame) -> pd.DataFrame:
    """
    genereate output data for analysis
    :return:
    """
    df_split = split_on_daily_basis(df_data)
    df_jump = calculate_overnight_jump(df_split)
    return df_jump


if __name__ == "__main__":
    df_data_retrieved = retrieve_data()
    df_output = generate_output_data(df_data_retrieved)
    print(df_output)
