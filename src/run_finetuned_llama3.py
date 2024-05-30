from unsloth import FastLanguageModel
from transformers import TextStreamer
from llm_data_loader import get_all_test_data
from prompt import get_prompt
from utils.utils import save_to_file
from tqdm import tqdm
from config import run_args

def load_model_and_tokenizer(ds_name):
    max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
    dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = f"unsloth_llama-3-8b-Instruct-bnb-4bit_{ds_name}", # YOUR MODEL YOU USED FOR TRAINING
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )
    FastLanguageModel.for_inference(model) # Enable native 2x faster inference
    
    return model, tokenizer


def get_prompt_content(prompt, content, tokenizer):
    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

    ### Instruction:
    {}

    ### Input:
    {}

    ### Response:
    {}"""
    
    inputs = tokenizer(
        [
            alpaca_prompt.format(
                f"{prompt}\n{content}", # instruction
                "", # input
                "", # output - leave this blank for generation!
            )
        ], return_tensors = "pt").to("cuda")

    return inputs

if __name__ == '__main__':
    run_args()
    
    data = get_all_test_data()
    prompts = get_prompt()
    
    for ds_name in data:
        x_test = data[ds_name]['x_test']
        
        model, tokenizer = load_model_and_tokenizer(ds_name)
        prompt = prompts[ds_name]['basic']
        
        results = {}
        for i in tqdm(range(len(x_test)), f'Running {ds_name}...'):
            inputs = get_prompt_content(prompt, x_test[i], tokenizer)
            text_streamer = TextStreamer(tokenizer)
            result = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 128)
            inputs_ids = inputs.data['input_ids']
            
            result = tokenizer.decode(result[0][inputs_ids.shape[-1]: ], skip_special_tokens=True)
            result = result.strip(' \t\n')
            results[f"{i}"] = result
        save_file_name = f"llm/finetune/llama3/{ds_name}/basic"
        save_to_file(results, save_file_name, 'json', with_t=True)