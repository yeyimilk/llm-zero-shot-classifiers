
import json
import pandas as pd
from handle_dataset import SMS_LABELS, CORONA_LABELS, ECOMMERCE_LABELS, FINANCIAL_LABELS
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix        
        
def get_results(model, name):
    f = f"ai_results/{model}/{name}.txt"
    with open(f, 'r') as file:
        json_data = json.loads(file.read())
    
    keys = []
    results = []
    errors = []
    for key in json_data.keys():
        values = json_data[key]
        for v_k in values:
            keys.append(int(v_k))
            results.append(values[v_k])
    for i in range(1, 151, 1):
        if i not in keys:
            errors.append(i)
    
    return keys, results, errors


def print_resutls(model, name, y_true, y_pred):
    print(f"==={model} {name}====")
    print("{:.4f} & {:.4f}".format(accuracy_score(y_true, y_pred), f1_score(y_true, y_pred, average='macro')))
    print(confusion_matrix(y_true, y_pred))
    print("\n")

def gpts(model, ds, name, label_map):
    keys, results, errors = get_results(model, name)
    
    data = pd.read_csv(ds)
    y_true = data['label'].tolist()
    
    if len(y_true) != len(results):
        print(f"Error: {name} has {len(y_true)} records, but {len(results)} results")
        return 
    
    y_pred = []
    for r in results:
        if r.lower() in label_map.keys():
            y_pred.append(label_map[r.lower()])
        else:
            # print(f"{name} {model} has error on {r}")
            y_pred.append(1)
    
    print_resutls(model, name, y_true, y_pred)

def llamas(model, ds, name, label_map):
    f = f"ai_results/{model}/{name}.txt"
    with open(f, 'r') as file:
        json_data = json.loads(file.read())
    
    data = pd.read_csv(ds)
    y_true = data['label'].tolist()
    y_pred = []
    for key in json_data.keys():
        values = json_data[key]
        try:
            if type(values) == str:
                lables = label_map.keys()
                values = values.lower()
                count = 0
                for l in lables:
                    if l in values:
                        count += 1
                pred = None
                if count == 1:
                    for l in lables:
                        if l in values:
                            pred = label_map[l]
                if pred is not None:            
                    y_pred.append(pred)
                else:
                    raise Exception("not found in str")
                        
            elif type(values) == dict:
                v = values[list(values.keys())[0]]
                v = label_map[v.lower()]
                v = v if v is not None else 1 # default to neural
                y_pred.append(v)
            else:
                raise Exception("not supported type")
        except Exception as e:
            # default to neural
            y_pred.append(1)
    print_resutls(model, name, y_true, y_pred)
    

def run_to_get_reports(model, ds, name, label_map):
    label_map = {k.lower(): v for k, v in label_map.items()}
    
    if 'llama' in model:
        llamas(model, ds, name, label_map)
    else:
        gpts(model, ds, name, label_map)


def for_4_dataset():
    models = ['gpt3_5', 'gpt4', 'llama2']
    ds = ['../dataset/sms_spam_test_mini.csv', '../dataset/Corona_NLP_1_test_mini.csv',
        '../dataset/ecommerceDataset_test_mini.csv', '../dataset/financial_sentiment_test_mini.csv']
    names = ['sms', 'tweets', 'ecommerce', 'financial']
    CORONA_LABELS = {'Negative': 0, 'Neutral': 1, 'Positive': 2}
    label_maps = [SMS_LABELS, CORONA_LABELS, ECOMMERCE_LABELS, FINANCIAL_LABELS]
    
    for i in range(len(ds)):
        for model in models:
            run_to_get_reports(model, ds[i], names[i], label_maps[i])

def for_tweets():
    models = ['gpt3_5', 'gpt4', 'llama2']
    ds = '../dataset/Corona_NLP_1_test_mini.csv'
    name = 'tweets'
    CORONA_LABELS = {'Negative': 0, 'Neutral': 1, 'Positive': 2}
    
    for i in range(5):
        for model in models:
            run_to_get_reports(model, ds, f"{name}_new_{i}", CORONA_LABELS)
            run_to_get_reports(model, ds, f"{name}_clean_{i}", CORONA_LABELS)
            run_to_get_reports(model, ds, f"{name}_covid_{i}", CORONA_LABELS)
            run_to_get_reports(model, ds, f"{name}_covid_{i}", CORONA_LABELS)

if __name__ == '__main__':
    for_4_dataset()
    for_tweets()