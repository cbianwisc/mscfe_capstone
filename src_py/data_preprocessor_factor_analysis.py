import datetime
import pandas as pd
from src_py.data_reader import retrieve_data


def split_on_daily_basis(df: pd.DataFrame) -> pd.DataFrame:
    """
    converting the time series data to discrete based on date
    :param df:
    :return:
    """
    df['date'] = df['datetime'].map(lambda x: x.date())
    return df


if __name__ == "__main__":
    df_close_return = retrieve_data()
    daily_time = datetime.time(4, 0, 0)
    df_split = split_on_daily_basis(df_close_return)

    print(df_close_return)