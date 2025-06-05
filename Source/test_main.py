import Debug # Debugging utility class
from starcoder import generate_response
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# ✅ 모델 이름

# ✅ 2. 토크나이저 로드

D = Debug.Debug()  # Create an instance of Debug class
model_name = "bigcode/starcoder2-3b"
# CUDA 사용 가능한지 확인
if torch.cuda.is_available():
    device_map = "auto"
    max_memory = {0: "20GiB"}  # 0번 GPU에 최대 메모리 제한 설정 (원한다면 생략 가능)
    torch_dtype = torch.float16  # GPU에서는 float16 사용
else:
    device_map = {"": "cpu"}
    max_memory = {"cpu": "12GiB"}
    torch_dtype = torch.float32  # CPU에서는 float32 사용

# 모델 로드
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map=device_map,
    torch_dtype=torch_dtype,
    max_memory=max_memory
)


tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    use_fast=True
)
D.get_code()  # Get code from user input
D.check_grammar()  # Check grammar of the code
# generate_response()  # Generate response using the model

# # Example code to run pylint on
# code = """import math
# import math  # duplicate import (W0404)

# def BadFunctionName():  # invalid function name (C0103)
#     x = 10  # unused variable (W0612)
#     return

# def conflict_example(x):  # argument name reused (W0621)
#     x = 5
#     return x

# def undefined_var_example():
#     return y  # undefined variable (E0602)

# def unnecessary_else(x):
#     if x > 0:
#         return "positive"
#     else:  # unnecessary else after return (R1705)
#         return "non-positive"

# def useless_return():  # useless return (R1711)
#     return

# def broken_syntax_example():  # syntax is valid here, but let's assume we test a wrong one
#     pass

# class NoInitClass:  # no __init__ method (E1101)
#     pass

# obj = NoInitClass()
# obj.name = "test"  # dynamic attribute assignment without __init__

# print("실행됨!")  # top-level script execution (W1514)

# def __main__():
#     BadFunctionName()
#     conflict_example(10)
#     undefined_var_example()
#     unnecessary_else(0)
#     useless_return()
#     broken_syntax_example()
#     print(obj.name)

# __main__()"""


