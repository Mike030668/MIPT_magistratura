from transformers import GenerationConfig
import random

# roles
ROLES = ["Stan", "Kyle", "Cartman", "Randy", "Butters"]

def generate_answer(query, model, tokenizer, max_new_tokens, device, role):

    encoding = tokenizer(query, return_tensors="pt").to(device)
    generation_config = GenerationConfig(max_new_tokens=max_new_tokens,
                                         pad_token_id = tokenizer.eos_token_id,repetition_penalty=1.3,
                                         eos_token_id = tokenizer.eos_token_id)

    outputs = model.generate(input_ids=encoding.input_ids, generation_config=generation_config)
    text_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

    ans = text_output.split(f"your role: {role}\n\n")[1].split("answer:")[1:]
    res = []
    for txt in ans:
       txt = txt.replace("[/INST]", " ").replace("\n", " ")
       if len(txt) > 5: res.append(txt.split(".")[0])
    
    if len(res) == 1:
      return res[0]
    elif len(res)>1:   
      return random.choice(res)

    else: return "I can't undanstand you, please answer me again"

def get_prompt(query, context, role = "Kyle"):
    prompt = f"<s>[INST]"
    prompt += f'Use the given context to guide your an about the query like indicated in your role'
    prompt += f"query: {query}\n\n"
    prompt += f"context: {context}\n\n"
    prompt += f"your role: {role}\n\n"
    prompt += f'answer:[/INST]</s>'

    return prompt