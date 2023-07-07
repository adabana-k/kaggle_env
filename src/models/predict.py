import pickle
from pathlib import Path
from src import config
import pandas as pd
import numpy as np
import re


def train_predict(model_folder, X_train, k):
    result_df = pd.DataFrame(columns=["Id", "class_0", "class_1"])
    path_list = Path(config.MODELS_FILE_PATH + model_folder).glob('**/*.pickle')
    for path in path_list:
        fold_num = int(re.findall(r'.+_k([0-9]+)$', path.stem)[0])
        tmp_train = X_train[X_train["kfold"] == fold_num].reset_index(drop=True)
        tmp_id = tmp_train[[config.NO_TRAIN_VALUE]]
        tmp_train = tmp_train.drop([config.NO_TRAIN_VALUE, config.TARGET_VALUE , "kfold"], axis=1)
        loaded_model = pickle.load(open(path, "rb"))
        proba = loaded_model.predict_proba(tmp_train)
        tmp_df = pd.concat([tmp_id, pd.DataFrame(proba, columns=["class_0", "class_1"])], axis=1)
        result_df = pd.concat([result_df, tmp_df], axis=0)

    result_df.sort_values('Id', inplace=True)
    X_train.sort_values('Id', inplace=True)

    return pd.concat([X_train, result_df.reset_index(drop=True)], axis=1)


def submission_predict(model_folder, X_test):
    result_df = pd.DataFrame(columns=["Id", "class_0", "class_1"])
    path_list = Path(config.MODELS_FILE_PATH + model_folder).glob('**/*.pickle')
    for path in path_list:
        tmp_id = X_test[[config.NO_TRAIN_VALUE]]
        tmp_test = X_test.drop([config.NO_TRAIN_VALUE, config.TARGET_VALUE], axis=1)
        loaded_model = pickle.load(open(path, "rb"))
        proba = loaded_model.predict_proba(tmp_test)
        tmp_df = pd.concat([tmp_id, pd.DataFrame(proba, columns=["class_0", "class_1"])], axis=1)
        result_df = pd.concat([result_df, tmp_df], axis=0)

    result_df.sort_values('Id', inplace=True)
    X_test.sort_values('Id', inplace=True)

    return pd.concat([X_test, result_df.reset_index(drop=True)], axis=1)
