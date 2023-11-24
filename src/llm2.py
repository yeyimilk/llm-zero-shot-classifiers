import requests
import time
import json
import pandas as pd
import os
from handle_data import text_cleaner

def send_to_llama2(prompt, system_prompt):
    max_retry_attempts = 3

    print(f"ai task prompt:\n{prompt}")

    for i in range(max_retry_attempts):
        try:
            results = requests.post('https://www.llama2.ai/api', json = {
                "prompt": prompt,
                "systemPrompt": system_prompt,
                "temperature": 0.01,
                "topP": 0.9,
                "maxTokens": 800,
                "version": "02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3",
            }, timeout=20)
            try:
                results = results.json()
                return results
            except Exception as e:
                if results.text:
                    return results.text
                raise Exception("not json and not text response returned from llama2.ai")
        except Exception as e:
            print(f"An unexpected error occurred on {i} time: {e}")
    
    # Return 'None' after three failed attempts
    return 'None'

    
def excute_ai_task(prompt, records, name):
    results = {}
    count = 0
    text = ''
    for i, record in enumerate(records):
        count += 1
        text += f"{i}. {record}\n\n"
        
        if count == 1: # since llam2 does not support json format, make one by one would be easier for downstream tasks
            task_prompt = f"{prompt}\n\n-------content------\n{text}"
            result = send_to_llama2(task_prompt, prompt)
            results[f"{i}"] = result
            count = 0
            text = ''
            print(f"ai task {name} results: {result}")
        
        if i > 0 and i % 5 == 0:
            print(f"ai task {name} in progress: {i+1}/{len(records)}")
            time.sleep(5)
        
    
    directory = f"ai_results/llama2"
    if not os.path.exists(directory):
        os.makedirs(directory)

    file_path = os.path.join(directory, f"{name}.txt")
    try:
        with open(file_path, "w") as f:
            f.write(json.dumps(results))
    except IOError as e:
        print(f"An error occurred: {e}")

def do_tasks(address, name, saved_name=None):
    f = f"./prompts/{name}.txt"
    with open(f, 'r') as file:
        prompt = file.read()
    records = pd.read_csv(address)
    records = records['text'].tolist()
    saved_name = name if saved_name is None else saved_name
    excute_ai_task(prompt, records, saved_name)
    time.sleep(30)
    
def run_baseline(i):
    sms_file = '../dataset/sms_spam_test_mini.csv'
    tweets_file = '../dataset/Corona_NLP_1_test_mini.csv'
    eco_file = '../dataset/ecommerceDataset_test_mini.csv'
    fin_file = '../dataset/financial_sentiment_test_mini.csv'
    
    # do_tasks(sms_file, 'sms')
    do_tasks(tweets_file, 'tweets', f'tweets_new_{i}')
    # do_tasks(eco_file, 'ecommerce')
    # do_tasks(fin_file, 'financial')
              
def clean_up_tweets(i):
    f = f"./prompts/tweets.txt"
    with open(f, 'r') as file:
        prompt = file.read()
    sms_file = '../dataset/Corona_NLP_1_test_mini.csv'
    records = pd.read_csv(sms_file)
    records = records['text'].apply(text_cleaner).tolist()
    excute_ai_task(prompt, records, f'tweets_clean_{i}')
    time.sleep(30)
        
if __name__ == "__main__":
    for i in range(2, 6, 1):
        run_baseline(i)
        clean_up_tweets(i)