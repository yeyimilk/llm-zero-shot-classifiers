
import json
import pandas as pd
from handle_dataset import SMS_LABELS, CORONA_LABELS, ECOMMERCE_LABELS, FINANCIAL_LABELS
from sklearn.metrics import accuracy_score, f1_score        
        
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


def gpts(model, ds, name, label_map):
    keys, results, errors = get_results(model, name)
    
    data = pd.read_csv(ds)
    y_true = data['label'].tolist()
    
    if len(y_true) != len(results):
        print(f"Error: {name} has {len(y_true)} records, but {len(results)} results")
        return 
    
    y_pred = [label_map[r.lower()]for r in results]
    print(f"==={model} {name}====")
    print("{:.4f} & {:.4f}".format(accuracy_score(y_true, y_pred), f1_score(y_true, y_pred, average='macro')))
    print("\n")

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
                v = v if v is not None else -1
                y_pred.append(v)
            else:
                raise Exception("not supported type")
        except Exception as e:
            # print(f"{name} has error on {key}, {values} {e}")
            y_pred.append(-1)
    
    print(f"==={model} {name}====")
    print("{:.4f} & {:.4f}".format(accuracy_score(y_true, y_pred), f1_score(y_true, y_pred, average='macro')))
    print("\n")

def run_to_get_reports(model, ds, name, label_map):
    label_map = {k.lower(): v for k, v in label_map.items()}
    
    if 'llama' in model:
        llamas(model, ds, name, label_map)
    else:
        gpts(model, ds, name, label_map)
    

# if __name__ == '__main__':
#     models = ['gpt3_5', 'gpt4', 'llama2']
#     ds = ['../dataset/sms_spam_test_mini.csv', '../dataset/Corona_NLP_1_test_mini.csv',
#         '../dataset/ecommerceDataset_test_mini.csv', '../dataset/financial_sentiment_test_mini.csv']
#     names = ['sms', 'tweets', 'ecommerce', 'financial']
#     label_maps = [SMS_LABELS, CORONA_LABELS, ECOMMERCE_LABELS, FINANCIAL_LABELS]
    
#     for model in models:
#         for i in range(len(ds)):
#             run_to_get_reports(model, ds[i], names[i], label_maps[i])

if __name__ == '__main__':
    # models = ['gpt3_5', 'gpt4']
    models = ['llama2']
    ds = ['../dataset/Corona_NLP_1_test_mini.csv']
    names = []
    for i in range(1, 6, 1):
        names.append(f"tweets_new_{i}")
        # if i > 0:
        #     names.append(f"tweets_new_{i}")
        # else:
        #     names.append(f"tweets_new")
            
    # for i in range(5):
    #     if i > 0:
    #         names.append(f"tweets_clean_{i}")
    #     else:
    #         names.append(f"tweets_clean")
    for i in range(1, 6, 1):
        names.append(f"tweets_clean_{i}")
        # if i > 0:
        #     names.append(f"tweets_clean_{i}")
        # else:
        #     names.append(f"tweets_clean")
    label_maps = [CORONA_LABELS]
    
    for model in models:
        for i in range(len(names)):
            run_to_get_reports(model, ds[0], names[i], label_maps[0])
    
    
