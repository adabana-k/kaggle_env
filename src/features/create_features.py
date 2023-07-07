import pandas as pd


def one_hot_encode(train, submission, column_list):
    train["is_train_data"] = 1
    submission["is_train_data"] = 0
    train_submission = pd.get_dummies(pd.concat([train, submission], axis=0), columns=column_list)
    result_train = train_submission[train_submission["is_train_data"] == 1].drop(["is_train_data"], axis=1)
    result_submission = train_submission[train_submission["is_train_data"] == 0].drop(["is_train_data"], axis=1)

    return result_train, result_submission
