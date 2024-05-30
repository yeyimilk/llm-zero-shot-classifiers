import google.generativeai as genai
import time
from tqdm import tqdm
from .llm_utils import excute_llm
from utils.utils import save_to_file

knn_size = 5

def send_gen(content, model='gemini-pro'):
    attempt = 0
    
    while attempt < 3:
        try:
            model = genai.GenerativeModel(model)
            response = model.generate_content(content, stream=True)
            response.resolve()
            return response.text
        except BaseException as e:
            print(f"\n{attempt} {content} with error: {e}\n")
            
        time.sleep(2 ** (attempt + 1))
        attempt += 1
    
    return {'value': 'error'}
 

def generate_content(prompt, info):
    return f"{prompt}\n{info}"   
    
def excute_ai_task(prompt, test_list, model):
    results = {}
    for i, info in tqdm(enumerate(test_list)):
        content = generate_content(prompt, info)
        result = send_gen(content, model)
        results[f"{i}"] = result
        
        i = i + 1
        # control the request rate to avoid the rate limit
        if i % 10 == 0:
            time.sleep(10)
        
        save_to_file(results, f"llm/{model}/gemini_tmp", 'json', with_t=False)
        
    return results


def run_gemini():
    # make sure your GOOGLE_API_KEY is set and you may adjust the request rate to avoid the rate limit
    GOOGLE_API_KEY = 'YOUR_GOOGLE_API_KEY'
    genai.configure(api_key=GOOGLE_API_KEY)
    excute_llm(['gemini-pro'], excute_ai_task)
