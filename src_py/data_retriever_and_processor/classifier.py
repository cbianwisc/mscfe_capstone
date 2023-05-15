import math

import pandas as pd

from src_py.data_retriever_and_processor.data_preprocessor_factor_analysis import generate_output_data
from src_py.data_retriever_and_processor.data_reader import retrieve_data

ALMOST_ZERO_THRESHOLD = 0.01


def classify_output_data(df):
    list_jump = df['overnight_jump'].to_list()
    y = [0] * len(list_jump)
    for i in range(len(list_jump)):
        if math.isnan(list_jump[i]):
            y[i] = 1
        elif list_jump[i] < 0.0 - ALMOST_ZERO_THRESHOLD:
            y[i] = 0
        elif list_jump[i] > 0.0 + ALMOST_ZERO_THRESHOLD:
            y[i] = 2
        else:
            y[i] = 1
    df_ret = df.copy().drop(columns=['overnight_jump'])
    df_ret['overnight_jump'] = pd.Series(y, index=df_ret.index)
    return df_ret


if __name__ == "__main__":
    df_data_retrieved = retrieve_data()
    df_output = generate_output_data(df_data_retrieved)
    df_classified = classify_output_data(df_output)
    print(df_classified)
