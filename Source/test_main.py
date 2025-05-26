# 예비 모델 : mistralai/Ministral-8B-Instruct-2410
# Mistral AI Research License -> 연구 목적 외 상업적 사용 금지.
#pip install --upgrade vllm / Make sure you install vLLM >= v0.6.4:
#pip install --upgrade mistral_common / Also make sure you have mistral_common >= 1.4.4 installed:
from vllm import LLM
from vllm.sampling_params import SamplingParams
import Debug # Debugging utility class


