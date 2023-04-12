import datetime
import pandas as pd
from src_py.data_reader import test_setup


def split_on_daily_time(df: pd.DataFrame, daily_time: datetime.time) -> pd.DataFrame:
    """
    converting the time series data to discrete based on time daily_time
    :param df:
    :param daily_time:
    :return:
    """
    start_time = df['datetime'][0]
    end_time = df['datetime'][df.shape[0] - 1]

    iter_start_time = start_time
    while iter_start_time < end_time:
        print(i)
        i += 1

    mask = df['Sales'] >= s
    df1 = df[mask]
    return df

if __name__ == "__main__":
    df_close_return = test_setup()
    daily_time = datetime.time(4, 0, 0)
    df_split = split_on_daily_time(df_close_return, daily_time)

    print(df_close_return)