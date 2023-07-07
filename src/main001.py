import datetime
from src.data import create_folds
import pandas as pd
from src import config
from src.features import create_features
from src.models import create_xgboost, predict


train = pd.read_csv(config.INPUT_DATA_PATH + "train.csv")
submission = pd.read_csv(config.INPUT_DATA_PATH + "test.csv")

train, submission = create_features.one_hot_encode(train, submission, config.QUALITATIVE_VALUE)
train = create_folds.create_folded_data_file(train, config.FOLD_NUM)

#train = train.drop(config.NO_TRAIN_VALUE, axis=1)
#submission = submission.drop(config.NO_TRAIN_VALUE, axis=1)

train.to_csv(config.INPUT_DATA_PATH + "fixed_train" + datetime.datetime.now().strftime('%Y%m%d%H%M%S') + ".csv")
submission.to_csv(config.INPUT_DATA_PATH + "fixed_submission" + datetime.datetime.now().strftime('%Y%m%d%H%M%S') + ".csv")

# train = pd.read_csv(config.INPUT_DATA_PATH + "fixed_train20230608223644.csv")
create_xgboost.create_model(train, config.TARGET_VALUE)
#result = predict.train_predict("xgb001", train, config.FOLD_NUM)
#result.to_csv(config.OUTPUT_DATA_PATH + "train_predicted" + datetime.datetime.now().strftime('%Y%m%d%H%M%S') + ".csv")
result = predict.submission_predict("xgb001", submission)
result.to_csv(config.OUTPUT_DATA_PATH + "submission.csv")