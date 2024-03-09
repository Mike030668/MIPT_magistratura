from transformers import GenerationConfig

# roles
ROLES = ["Stan", "Kyle", "Cartman", "Randy", "Butters"]

def generate_answer(query, model, tokenizer, max_new_tokens, device):

    encoding = tokenizer(query, return_tensors="pt").to(device)
    generation_config = GenerationConfig(max_new_tokens=max_new_tokens,
                                         pad_token_id = tokenizer.eos_token_id,repetition_penalty=1.3,
                                         eos_token_id = tokenizer.eos_token_id)

    outputs = model.generate(input_ids=encoding.input_ids, generation_config=generation_config)
    text_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text_output.split("answer:[/INST]")[1].split("\n\n")[0]


def get_prompt(query, context, role = "Kyle"):
    prompt = f"<s>[INST]"
    prompt += f'Use the given context to guide your an about the query like indicated in your role'
    prompt += f"query: {query}\n\n"
    prompt += f"context: {context}\n\n"
    prompt += f"your role: {role}\n\n"
    prompt += f'answer:[/INST]</s>'

    return prompt