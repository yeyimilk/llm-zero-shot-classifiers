
from transformers import AutoTokenizer
from tqdm import tqdm
from llm.llm_utils import generate_content
from prompt import get_prompt
from llm_data_loader import get_all_test_data
from peft import AutoPeftModelForCausalLM
from utils.utils import save_to_file

def load_model_and_tokenizer(path_to_adapter):
    model = AutoPeftModelForCausalLM.from_pretrained(
        path_to_adapter, # path to the output directory
        trust_remote_code=True,
        device_map={"": "cuda:0"}
    ).eval()

    tokenizer = AutoTokenizer.from_pretrained(path_to_adapter, trust_remote_code=True)

    return model, tokenizer

if __name__ == '__main__':
    
    data = get_all_test_data()
    prompts = get_prompt()
    
    abs_path = "YOUR_FINETUNED_PATH/Qwen/output"
    
    for ds_name in data:        
        x_test = data[ds_name]['x_test']
        prompt = prompts[ds_name]['basic']
        
        model_path = f"{abs_path}/Qwen-7B-Chat_basic_{ds_name}"
        model, tokenizer = load_model_and_tokenizer(model_path)
        
        results = {}
        for i, info in tqdm(enumerate(x_test)):
            content = generate_content(prompt, info)
            result, _  = model.chat(tokenizer, content, history=None)
            results[f"{i}"] = result

        save_file_name = f"llm/finetune/qwen/{ds_name}"
        save_to_file(results, save_file_name, 'json', with_t=True)