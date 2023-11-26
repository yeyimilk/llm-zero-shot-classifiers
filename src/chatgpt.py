
import openai
import pandas as pd
import time
import json
import os
import random
from handle_data import text_cleaner

def retry_with_exponential_backoff(
    func,
    initial_delay: float = 3,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = 3,
    errors: tuple = (openai.error.RateLimitError,),
):
    """Retry a function with exponential backoff."""

    def wrapper(*args, **kwargs):
        num_retries = 0
        delay = initial_delay

        while True:
            try:
                return func(*args, **kwargs)

            except errors as e:
                num_retries += 1

                if num_retries > max_retries:
                    raise Exception(f"Maximum number of retries ({max_retries}) exceeded.")

                delay *= exponential_base * (1 + jitter * random.random())

                time.sleep(delay)

            except Exception as e:
                raise e

    return wrapper

@retry_with_exponential_backoff
def completions_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)

def send_to_gpt(prompt, conversation_id, model='gpt-3.5-turbo-1106'):
    max_retry_attempts = 3

    print(f"ai task prompt:\n{prompt}")

    for i in range(max_retry_attempts):
        try:
            messages = []
            if conversation_id is None:
                messages = [{"role": "system", "content": "start"}]
            else:
                messages = [{"role": "system", "content": "continue"}]
            
            messages.append({'role':'user', 'content': prompt})
            
            response = completions_with_backoff(
                    model=model,
                    messages=messages,
                    max_tokens=1000,
                    temperature=0.01,
                    response_format = {"type": 'json_object'},
                    top_p=0.9
            )
            content = response['choices'][0]['message']['content']
            id = response['id']
            content = json.loads(content)
            return id, content
        except Exception as e:
            print(f"An unexpected error occurred on {i+1} time: {e}")
            time.sleep(10)
    
    # Return 'None' after three failed attempts
    return None, {}

    
def excute_ai_task(prompt, records, name, model):
    results = {}
    count = 0
    text = ''
    conversation_id = ''
    for i, record in enumerate(records):
        count += 1
        text += f"{i+1}. {record}\n\n"
        
        if count == 5:
            task_prompt = f"{prompt}\n\n-------content------\n{text}"
            id, content = send_to_gpt(task_prompt, conversation_id, model)
            # if id != None:
            #     conversation_id = id
            results[i] = content
            count = 0
            text = ''
            print(f"ai task {name}, {conversation_id} results: {content}")
        
        if i > 0 and i % 5 == 0:
            print(f"ai task {name} in progress: {i+1}/{len(records)}")
            inter = 10 if 'gpt-4' in model else 5
            time.sleep(inter)
    
    directory = f"ai_results/gpt4"
    if '3.5' in model:
        directory = f"ai_results/gpt3_5"
    
    if not os.path.exists(directory):
        os.makedirs(directory)

    file_path = os.path.join(directory, f"{name}.txt")
    try:
        with open(file_path, "w") as f:
            f.write(json.dumps(results))
    except IOError as e:
        print(f"An error occurred: {e}")

def do_tasks(address, name, model, file_name=None, should_clean=False):
    f = f"./prompts/{name}.txt"
    with open(f, 'r') as file:
        prompt = file.read()
    records = pd.read_csv(address)
    records = records['text'].tolist()
    if should_clean:
        records = [text_cleaner(r) for r in records]
    saved_file_name = file_name if file_name is not None else name
    excute_ai_task(prompt, records, saved_file_name, model)   
    time.sleep(60)        
              
  
def run_baseline():
    for model in ['gpt-4-1106-preview', 'gpt-3.5-turbo-1106']:
        sms_file = '../dataset/sms_spam_test_mini.csv'
        tweets_file = '../dataset/Corona_NLP_1_test_mini.csv'
        eco_file = '../dataset/ecommerceDataset_test_mini.csv'
        fin_file = '../dataset/financial_sentiment_test_mini.csv'
        
        do_tasks(sms_file, 'sms', model)
        do_tasks(tweets_file, 'tweets', model)
        do_tasks(eco_file, 'ecommerce', model)
        do_tasks(fin_file, 'financial', model)         
              
              
def run_tweets():
    for i in range(5):
        for model in ['gpt-4-1106-preview', 'gpt-3.5-turbo-1106']:
            tweets_file = '../dataset/Corona_NLP_1_test_mini.csv'
            prompt_file = 'tweets'
            file_name = 'tweets'
            do_tasks(tweets_file, prompt_file, model, file_name=f"{file_name}_new_{i}", should_clean=False)
            do_tasks(tweets_file, prompt_file, model, file_name=f"{file_name}_clean_{i}", should_clean=True)
    
    for i in range(5):
        for model in ['gpt-4-1106-preview', 'gpt-3.5-turbo-1106']:
            tweets_file = '../dataset/Corona_NLP_1_test_mini.csv'
            prompt_file = 'tweets_covid'
            file_name = 'tweets_covid'
            do_tasks(tweets_file, prompt_file, model, file_name=f"{file_name}_{i}", should_clean=False)
            do_tasks(tweets_file, prompt_file, model, file_name=f"{file_name}_clean_{i}", should_clean=True)
            

              
if __name__ == "__main__":
    openai.api_key = os.getenv('OPENAI_API_KEY') or 'YOUR_OPEN_AI_KEY'
    run_baseline()
    run_tweets()