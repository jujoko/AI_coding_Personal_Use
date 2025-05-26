class Debug: # Debugging utility class #디버깅 역할을 할 클래스.
    def __init__(self, code_file=None): # code_file: str, code file directory
        self.code = ""  # code: str, code to debug
        if code_file is not None:
            with open(code_file, "r") as f: 
                self.code = f.read()    
        self.code_file = code_file  
        
        #예시 모델.
        from vllm import LLM
        from vllm.sampling_params import SamplingParams
        model_name = "mistralai/Ministral-8B-Instruct-2410"
        self.llm_params = SamplingParams(max_tokens=8192)
        self.llm =  LLM(model=model_name, tokenizer_mode="mistral", config_format="mistral", load_format="mistral")
    
    def get_code(self): #디버깅 할 코드가 파일 형식이 아니라면 쓰는 메소드
        code = input()  # Get code from user input
        self.code = code
        
    def check_grammar(self):
        import ast
        error=ast.parse(self.code)
        print(error)
        
