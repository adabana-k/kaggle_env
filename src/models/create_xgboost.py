from xgboost import XGBClassifier
import pickle
from src import config


# To Do create_model.pyでは訓練データでモデルを作成
#       predict.pyではテストデータを引数にして予測をする
def create_model(train, object_name):
    # 先にIDをドロップしておく
    train = train.drop(config.NO_TRAIN_VALUE, axis=1)
    # 乱数シード
    seed = 42
    k = train["kfold"].max() + 1
    for i in range(k):
        train_y = train[train["kfold"] != i][object_name]
        train_x = train[train["kfold"] != i].drop([object_name, "kfold"], axis=1)
        test_y = train[train["kfold"] == i][object_name]
        test_x = train[train["kfold"] == i].drop([object_name, "kfold"], axis=1)
        # モデル作成
        model = XGBClassifier(
            booster='gbtree'
            , objective='binary:logistic'
            , random_state=seed
            , n_estimators=10000)  # チューニング前のモデル
        # 学習時fitパラメータ指定
        fit_params = {
            # 学習中のコマンドライン出力
            'verbose': 0,
            # 学習時、評価指標がこの回数連続で改善しなくなった時点でストップ
            'early_stopping_rounds': 10,
            # early_stopping_roundsの評価指標
            'eval_metric': 'logloss',
            # early_stopping_roundsの評価指標算出用データ
            'eval_set': [
                (train_x.values, train_y.values)
                , (test_x.values, test_y.values)]}
        model.set_params(
            eval_metric=fit_params["eval_metric"]
            , early_stopping_rounds=fit_params["early_stopping_rounds"]
        )
        model.fit(
            train_x.values
            , train_y.values
            , eval_set=fit_params["eval_set"]
            , verbose=fit_params["verbose"]
        )
        file_name = config.MODELS_FILE_PATH + "/xgb001/xgb_k" + str(i) + ".pickle"
        pickle.dump(model, open(file_name, "wb"))
