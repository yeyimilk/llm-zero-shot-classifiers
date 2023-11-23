
import pandas as pd
from handle_data import data_to_sequence

encoding = "ISO-8859-1"

INPUT_PATH = "../dataset"

def get_data_by_name(name, num_words):
    data = pd.read_csv(f"{INPUT_PATH}/{name}_train.csv", encoding=encoding)
    x_train, y_train = data['text'], data['label']
    data = pd.read_csv(f"{INPUT_PATH}/{name}_test_mini.csv", encoding=encoding)
    x_test, y_test = data['text'], data['label']
    x_train, x_test, size = data_to_sequence(x_train, x_test, num_words)
    
    return x_train, y_train.to_numpy(), x_test, y_test.to_numpy(), size