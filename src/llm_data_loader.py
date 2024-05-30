import pandas as pd
from config import cfig
from prompt import get_prompt
import json

encoding = "ISO-8859-1"

INPUT_PATH = "../dataset"

def get_raw_train_data_by_name(name):
    data = pd.read_csv(f"{INPUT_PATH}/{name}_train_mini.csv", encoding=encoding)
    x_test, y_test = data['text'], data['label']
    return x_test, y_test.to_numpy()

def get_test_data_by_name(name):
    data = pd.read_csv(f"{INPUT_PATH}/{name}_test_mini.csv", encoding=encoding)
    x_test, y_test = data['text'], data['label']
    return x_test, y_test.to_numpy()

def get_all_test_data():
    names = ['Corona_NLP_new', 'ecommerceDataset', 'sms_spam', 'financial_sentiment']
    names = names[cfig.args.ds: len(names)]
    
    data = {}
    
    for name in names:
        x_test, y_test = get_test_data_by_name(name)
        data[f'{name}'] = {
            'x_test': x_test,
            'y_test': y_test
        }
    return data


def make_ds_for_finetuning():
    ds_labels = {
            'Corona_NLP_new': ['negative', 'neutral', 'positive'],
            'ecommerceDataset': ['household', 'books', 'clothing & accessories', 'electronics'],
            'sms_spam': ['normal', 'spam'],
            'financial_sentiment': ['negative', 'neutral', 'positive']
    }
    vicuna_obj_list = []
    llama3_obj_list = []
    
    for ds_name in ds_labels.keys():
        x_test, y_test = get_raw_train_data_by_name(ds_name)
        
        all_prompts = get_prompt()
        basic_prompt = all_prompts[ds_name]['basic']
        
        for i, content in enumerate(x_test):
            output = '{"value":"' + list(ds_labels[ds_name])[y_test[i]] + '"}'
            vicuna_obj_list.append({
                "id": f"{ds_name}_{i}",
                "conversations": [{
                    "from": "human",
                    "value": f"{basic_prompt}\n{content}"
                }, {
                    "from": "gpt",
                    "value": output
                }]
            })
            
            
            content = f"{basic_prompt}\n{x_test[i]}"
            llama3_obj_list.append({
                "instruction": content,
                "input": "",
                "output": output
            })
            
            
        with open(f"../dataset/vicuna/basic_{ds_name}.json", "w") as final:
            json.dump(vicuna_obj_list, final)
        
        with open(f"../dataset/llama3_{ds_name}.json", "w") as final:
            json.dump(llama3_obj_list, final)

if __name__ == '__main__':
    make_ds_for_finetuning()
    
    