import datetime
from typing import List

INPUT_DATA_PATH = "../input/"
OUTPUT_DATA_PATH = "../output/"
MODELS_FILE_PATH = "../models/"

FOLD_NUM = 3
TARGET_VALUE = "Class"
# QUALITATIVE_VALUE = ["EJ", "Beta", "Gamma", "Delta", "Epsilon", "Epsilon_is_null", "Epsilon_Year", "Epsilon_Month"]
QUALITATIVE_VALUE: List[str] = ["EJ"]
NO_TRAIN_VALUE = "Id"
