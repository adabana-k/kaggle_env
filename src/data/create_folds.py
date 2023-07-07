from sklearn.model_selection import KFold


def create_folded_data_file(train, k):

    # 乱数シード
    seed = 42

    # クロスバリデーションして決定境界を可視化
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)  # KFoldでクロスバリデーション分割指定
    for i, (train_index, test_index) in enumerate(kf.split(train)):
        train.loc[test_index, "kfold"] = i

    train = train.astype({'kfold': "int8"})
    return train
