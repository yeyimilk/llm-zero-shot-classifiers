
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from .llm_utils import excute_llm, generate_content

def load_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cuda:0", trust_remote_code=True).cuda()
    model = model.eval()
    return model, tokenizer

@torch.inference_mode()
def excute_ai_task(prompt, test_list, model_path):
    model, tokenizer = load_model(model_path)
    
    results = {}
    for i, info in tqdm(enumerate(test_list)):
        content = generate_content(prompt, info)
        result, history  = model.chat(tokenizer, content, history=None)
        results[f"{i}"] = result
        
        print(f"{model_path}, {i} ==> {result}")
        i = i + 1
        break
        
    return results
    
    
def run_qwen():
    excute_llm(['Qwen/Qwen-7B-Chat',
                'Qwen/Qwen-14B-Chat'], 
               excute_ai_task)