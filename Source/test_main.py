# 예비 모델 : mistralai/Ministral-8B-Instruct-2410
# Mistral AI Research License -> 연구 목적 외 상업적 사용 금지.
#pip install --upgrade vllm / Make sure you install vLLM >= v0.6.4:
#pip install --upgrade mistral_common / Also make sure you have mistral_common >= 1.4.4 installed:
from vllm import LLM
from vllm.sampling_params import SamplingParams

#모델 로딩 & 출력
model_name = "mistralai/Ministral-8B-Instruct-2410"

sampling_params = SamplingParams(max_tokens=8192)

# note that running Ministral 8B on a single GPU requires 24 GB of GPU RAM
# If you want to divide the GPU requirement over multiple devices, please add *e.g.* `tensor_parallel=2`
llm = LLM(model=model_name, tokenizer_mode="mistral", config_format="mistral", load_format="mistral")

prompt = "Do we need to think for 10 seconds to find the answer of 1 + 1?"

messages = [
    {
        "role": "user",
        "content": prompt
    },
]

outputs = llm.chat(messages, sampling_params=sampling_params)

print(outputs[0].outputs[0].text)
