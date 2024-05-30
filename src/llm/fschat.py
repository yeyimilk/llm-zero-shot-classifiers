import torch
import json
from fastchat.model import load_model, get_conversation_template
from tqdm import tqdm
from .llm_utils import excute_llm, generate_content

def generate_result(msg, model_path, model, tokenizer):
    conv = get_conversation_template(model_path)
    conv.append_message(conv.roles[0], msg)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    # Run inference
    inputs = tokenizer([prompt], return_tensors="pt").to('cuda')
    output_ids = model.generate(
        **inputs,
        do_sample=False,
        temperature=0.1,
        repetition_penalty=1.0,
        max_new_tokens=128,
    )

    if model.config.is_encoder_decoder:
        output_ids = output_ids[0]
    else:
        output_ids = output_ids[0][len(inputs["input_ids"][0]) :]
    outputs = tokenizer.decode(
        output_ids, skip_special_tokens=True, spaces_between_special_tokens=False
    )
    
    return outputs

@torch.inference_mode()
def excute_ai_task(prompt, test_list, model_path):
    model, tokenizer = load_model(model_path)
    
    results = {}
    for i, info in tqdm(enumerate(test_list)):
        content = generate_content(prompt, info)
        result = generate_result(content, model_path, model, tokenizer)
        results[f"{i}"] = result
        
        print(f"{model_path}, {i} ==> {result}")
        i = i + 1
        
    return results
    
    
def run_fschat():
    excute_llm(['lmsys/vicuna-7b-v1.5',
                'lmsys/vicuna-13b-v1.5'], 
               excute_ai_task)
