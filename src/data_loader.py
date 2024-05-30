
import pandas as pd
from handle_data import data_to_sequence
from llm_data_loader import get_test_data_by_name

encoding = "ISO-8859-1"

INPUT_PATH = "../dataset"

def get_data_by_name(name, num_words):
    data = pd.read_csv(f"{INPUT_PATH}/{name}_train_mini.csv", encoding=encoding)
    x_train, y_train = data['text'], data['label']
   
    x_test, y_test = get_test_data_by_name(name)
    x_train, x_test, size = data_to_sequence(x_train, x_test, num_words)
    
    return x_train, y_train.to_numpy(), x_test, y_test, size