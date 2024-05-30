
from llm_data_loader import get_all_test_data
from utils.utils import save_to_file
from prompt import get_prompt
from tqdm import tqdm
from config import cfig

def generate_content(prompt, info):
    return f"{prompt}\n{info}"   


def run_prompt_strategy(model, ds_key, p_key, excute_func, test_data, prompts):
    save_file_name = f"llm/{model}/{ds_key}/{p_key}"
    prompt = prompts[f"{p_key}"]
    test_list = test_data[f"{ds_key}"]['x_test']                
    results = excute_func(prompt, test_list, model)
    
    save_to_file(results, save_file_name, 'json', with_t=True)

def excute_llm(models, excute_func):
    all_prompts = get_prompt()
    
    for model in models:
        test_data = get_all_test_data()
        for ds_key in tqdm(test_data, f'Processing {model}...'):
            print(f"\tProcessing {model}, {ds_key}, {cfig.args.prompt_type}...")
            prompts = all_prompts[f"{ds_key}"]
            
            if cfig.args.prompt_type == 'all':
                for p_key in prompts:
                    run_prompt_strategy(model, ds_key, p_key, excute_func, test_data, prompts)
            else:
                run_prompt_strategy(model, ds_key, cfig.args.prompt_type, excute_func, test_data, prompts)