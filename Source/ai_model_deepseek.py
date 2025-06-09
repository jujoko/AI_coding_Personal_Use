import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-1.3b-instruct", trust_remote_code=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AutoModelForCausalLM.from_pretrained(
    "deepseek-ai/deepseek-coder-1.3b-instruct",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32  # CPU는 bfloat16 지원 안 함
).to(device)

def get_AI_answer(prompt:str):
    messages=[
        { 'role': 'user', 'content': f"{prompt}"}
    ]
    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(device)
    # tokenizer.eos_token_id is the id of <|EOT|> token
    outputs = model.generate(inputs, max_new_tokens=512, num_return_sequences=1, eos_token_id=tokenizer.eos_token_id)
    answer = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
    return answer
