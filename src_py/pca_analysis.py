from sklearn.decomposition import PCA


def pca_analysis(df):
    for i in range(len(df.columns)):
        pca = PCA(n_components=i)
        # transpose so dimension reduction for each day
        pca.fit(df.T)
        print(pca.explained_variance_ratio_)
        # print(pca.singular_values_)
        if sum(pca.explained_variance_ratio_) > 0.95:
            break
