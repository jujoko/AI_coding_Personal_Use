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
#GPU를 사용할 수 있으면 사용하도록 변경 -> 속도 더 빨라짐.
device = "cuda"
if torch.cuda.is_available():
    device_map = "auto"
    max_memory = {0: "20GiB"}  # GPU 0번에 최대 20GiB 사용 허용
    torch_dtype = torch.float16
else:
    device_map = {"": "cpu"}
    max_memory = {"cpu": "12GiB"}
    torch_dtype = torch.float32

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map=device_map,
    torch_dtype=torch_dtype,
    max_memory=max_memory
).to(device)

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