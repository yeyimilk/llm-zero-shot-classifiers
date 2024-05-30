
def get_prompt_from_file(file_path):
    f = open(file_path, "r")
    prompt = f.read()
    f.close()
    return prompt


def get_prompt():
    prompts = {}
    
    names = ['Corona_NLP_new', 'ecommerceDataset', 'sms_spam', 'financial_sentiment']
    types = ['basic', 'few_shot']
    
    for name in names:
        prompts[f"{name}"] = {}
        for t in types:
            prompts[f"{name}"][f"{t}"] = get_prompt_from_file(f'./prompts/{name}/{t}.txt')
    
    return prompts