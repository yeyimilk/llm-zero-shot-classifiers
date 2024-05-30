import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

encoding = 'ISO-8859-1'
MINI_TRAIN_SIZE = 10000
MINI_TEST_SIZE = 800

SMS_LABELS = {'normal': 0, 'spam': 1}
CORONA_LABELS = {'Extremely Negative': 0, 'Negative': 0, 'Neutral': 1, 'Positive': 2, 'Extremely Positive': 2}
ECOMMERCE_LABELS = {'Household': 0, 'Books': 1, 'Clothing & Accessories': 2, 'Electronics': 3}
FINANCIAL_LABELS = {'positive': 2, 'neutral': 1, 'negative': 0}

def validate_dataset(data, allowed_values):
    data = data[data['label'].isin(allowed_values) & 
            data['label'].notna() & 
            data['text'].notna() & 
            (data['text'].apply(lambda x: isinstance(x, str)) &
            (data['text'].str.len() > 0)) & 
            (data['label'].str.len() > 0)]
    return data    

def get_data(souce, renames, label_maps):
    data = pd.read_csv(souce, encoding=encoding)
    if isinstance(renames, dict):
        data.rename(columns=renames, inplace=True)
    elif isinstance(renames, list):
        data.columns = renames
    
    data = validate_dataset(data, label_maps.keys())
    data['label'] = data['label'].map(label_maps).astype(np.int64)
    return data

def create_mini(data, count, dest):
    if count > len(data):
        data.to_csv(dest, index=False)
    else:
        _, df_mini = train_test_split(data, test_size=count/len(data),
                                        random_state=42, stratify=data['label'])
        df_mini.to_csv(dest, index=False)
    
    
def create_corona():
    
    renames = {'Sentiment': 'label', 'OriginalTweet': 'text'}
    new_name = 'Corona_NLP_new'
    data = get_data("../dataset/Corona_NLP_train.csv",
                    renames=renames,
                    label_maps=CORONA_LABELS)
    data.to_csv(f"../dataset/{new_name}_train.csv", index=False)
    create_mini(data, MINI_TRAIN_SIZE, f"../dataset/{new_name}_train_mini.csv")
    
    data = get_data("../dataset/Corona_NLP_test.csv",
                    renames=renames,
                    label_maps=CORONA_LABELS)
    data.to_csv(f"../dataset/{new_name}_test.csv", index=False)
    create_mini(data, MINI_TEST_SIZE, f"../dataset/{new_name}_test_mini.csv")
    

def create_ecommerce():   
    create_splited_data_set(souce="../dataset/ecommerceDataset.csv", 
                            renames=['label', 'text'],
                            label_maps=ECOMMERCE_LABELS,
                            dest="../dataset/ecommerceDataset",
                            train_mini=MINI_TRAIN_SIZE,
                            test_mini=MINI_TEST_SIZE)
    

def create_splited_data_set(souce, renames, label_maps, dest, train_mini, test_mini):
    data = get_data(souce, renames, label_maps)
    
    train, test = train_test_split(data, test_size=0.2,
                                    random_state=42, stratify=data['label'])
    train.to_csv(f'{dest}_train.csv', index=False)
    test.to_csv(f'{dest}_test.csv', index=False)
    
    
    if train_mini is not None:
        create_mini(train, train_mini, f'{dest}_train_mini.csv')

    create_mini(test, test_mini, f'{dest}_test_mini.csv')

def create_sms():
    create_splited_data_set(souce="../dataset/sms_spam.csv",
                         renames={'label': 'label', 'text': 'text'},
                         label_maps=SMS_LABELS,
                         dest="../dataset/sms_spam",
                         train_mini=MINI_TRAIN_SIZE,
                         test_mini=MINI_TEST_SIZE)
    
    
def create_financial_sentiment():
    create_splited_data_set(souce="../dataset/financial_sentiment.csv",
                         renames=['label', 'text'],
                         label_maps=FINANCIAL_LABELS,
                         dest="../dataset/financial_sentiment",
                         train_mini=MINI_TRAIN_SIZE,
                         test_mini=MINI_TEST_SIZE)
    

if __name__ == "__main__":
    create_corona()
    create_ecommerce()
    create_sms()
    create_financial_sentiment()