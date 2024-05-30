

from tqdm import tqdm
from .llm_utils import excute_llm

from llama import Llama

def excute_ai_task(prompt, test_list, model_path):
    results = {}
    YOUR_PATH = '/mnt/beegfs/home/zwang2022/projects/llama3'
    path_to_model = f'{YOUR_PATH}/Meta-Llama-3-8B-Instruct'
    generator = Llama.build(
        ckpt_dir=path_to_model,
        tokenizer_path=f"{path_to_model}/tokenizer.model",
        max_seq_len=4096,
        max_batch_size=6,
    )
    
    for i, info in tqdm(enumerate(test_list)):
        dialogs = [[{'role': 'user', 'content': f"{prompt}\n{info}"}]]
        result = generator.chat_completion(
            dialogs,
            max_gen_len=2048,
            temperature=0.1,
            top_p=0.9,
        )
        result = result[0]['generation']['content']
        results[f"{i}"] = result
        
        print(f"{model_path}, {i} ==> {result}")
        i = i + 1
        
    return results
    
    
def run_llama3():
    excute_llm(['meta-llama/Meta-Llama-3-8B-instruct'], 
               excute_ai_task)
