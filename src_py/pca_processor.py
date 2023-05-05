from sklearn.decomposition import PCA

from src_py.input_retriever import retrieve_input_data

PCA_EXPLAINED_VARIANCE_RATIO_CUTOFF = 0.95


def pca_analysis(df):
    for i in range(len(df.columns)):
        pca = PCA(n_components=i)
        # transpose so dimension reduction for each day
        pca.fit(df)
        print(pca.singular_values_)
        if sum(pca.explained_variance_ratio_) > PCA_EXPLAINED_VARIANCE_RATIO_CUTOFF:
            break
    return pca.fit_transform(df)


if __name__ == "__main__":
    df_retrieved = retrieve_input_data()
    df_after_pca = pca_analysis(df_retrieved)
    print(df_after_pca)
