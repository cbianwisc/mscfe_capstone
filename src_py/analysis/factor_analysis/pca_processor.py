import pandas as pd
from sklearn.decomposition import PCA

from src_py.data_retriever_and_processor.input_retriever import retrieve_input_data

PCA_EXPLAINED_VARIANCE_RATIO_CUTOFF = 0.95


def pca_analysis(df):
    for i in range(len(df.columns)):
        pca = PCA(n_components=i)
        pca.fit(df)
        # print(pca.singular_values_)
        if sum(pca.explained_variance_ratio_) > PCA_EXPLAINED_VARIANCE_RATIO_CUTOFF:
            break
    return pd.DataFrame(pca.fit_transform(df), index=df.index)


if __name__ == "__main__":
    df_retrieved = retrieve_input_data()
    df_after_pca = pca_analysis(df_retrieved)
    print(df_after_pca)
