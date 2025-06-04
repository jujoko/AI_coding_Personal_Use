class Debug: # Debugging utility class #디버깅 역할을 할 클래스.
    def __init__(self, code_file=None): # code_file: str, code file directory
        self.code = ""  # code: str, code to debug
        if code_file is not None:
            with open(code_file, "r") as f: 
                self.code = f.read()    
        self.code_file = code_file  
    def get_code(self): #code를 입력 받는 함수
        code = ""
        print("검사받을 코드를 한 행씩 입력해주세요. (종료하려면 '!STOP' 입력)")
        while True:
            tmp = input()
            if not tmp.strip():    #빈 줄 건너뛰기.
                continue
            if tmp.strip()[0] == "!":   # '!'로 시작하는 경우, 명령어로 인식.
                if tmp.strip()[1:] == "STOP":   # '!STOP' 입력 시, 입력 종료.
                    break
                if tmp.strip()[1:] == "CHECK":
                    print(code)
            else :
                code += tmp + "\n"
        self.code = code
        
    def check_grammar(self):
        import ast
        error=ast.parse(self.code)
        print(error)
        
