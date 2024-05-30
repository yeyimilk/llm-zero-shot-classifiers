
import openai
import time
import random
import os
from tqdm import tqdm
from .llm_utils import excute_llm
from utils.utils import save_to_file
def retry_with_exponential_backoff(
    func,
    initial_delay: float = 3,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = 3,
    errors: tuple = (None, None),
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
    try:
        return openai.ChatCompletion.create(**kwargs)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def send_to_gpt(messages, model):
    max_retry_attempts = 3

    for i in range(max_retry_attempts):
        try:
            response = completions_with_backoff(
                    model=model,
                    messages=messages,
                    response_format = {"type": 'json_object'},
                    max_tokens=4096,
            )
            print(response)
            content = response['choices'][0]['message']['content']
            return content
        except BaseException as e:
            print(f"An unexpected error occurred on {i+1} time: {e}")
            time.sleep(10)
    
    return {'value': 'error'}

def get_prompt_message(prompt, info):
    prompt = f"{prompt}\n{info}"
    
    prompt_message = [{
        "role": "user",
        "content": [
            {
                'type': 'text',
                'text': prompt
            }
        ]
    }]
    
    return prompt_message
    
def excute_ai_task(prompt, test_list, model):
    results = {}
    
    for i, info in tqdm(enumerate(test_list), f'Processing {model} Task'):     
        prompt_message = get_prompt_message(prompt, info)
        result = send_to_gpt(prompt_message, model)
        results[f"{i}"] = result
        
        i = i + 1
        print(f"{model}, {i} ==> {result}")
        
        if i % 60 == 0:
            time.sleep(5)
    
    return results
    
def run_chatgpt():
    # I would suggest your to create batch request file to reduce cost and improve the stabilities
    # I created one sample for your reference within dataset folder: batch_request_gpt35_basic.jsonl
    # for more details, please refer to https://beta.openai.com/docs/api-reference/batch-requests
    openai.api_key = os.getenv("OPENAI_API_KEY") or "YOUR OPENAI API KEY"
    excute_llm(['gpt-4-1106-preview', 'gpt-3.5-turbo-1106'], excute_ai_task)