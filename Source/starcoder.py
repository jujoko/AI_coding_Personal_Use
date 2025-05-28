from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# ✅ 모델 이름
model_name = "bigcode/starcoder2-3b"

# ✅ 2. 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    use_fast=True
)

# ✅ 3. 모델 로드 (4bit, device_map 자동 할당, CPU 오프로드 가능)
# 기존 코드 유지하면서 이 부분만 추가하세요
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16,
    max_memory={"cpu": "12GiB"}  
)

# ✅ 4. 테스트용 응답 함수 (텍스트 프롬프트 입력 → 텍스트 출력)
def generate_response(prompt: str):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=512)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ✅ 5. 실행 예시
if __name__ == "__main__":
    print("🧠 StarCoder2-3B model loaded. Type 'exit' to quit.")
    while True:
        prompt = input("💬 Prompt: ")
        if prompt.lower() in ["exit", "quit", "종료"]:
            break
        response = generate_response(prompt)
        print("\n🧠 답변:\n", response)