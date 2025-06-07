import os
class Debug: # Debugging utility class #디버깅 역할을 할 클래스.
    def __init__(self, code:str=None): # code_file: str, code file directory
        if code == None:
            code = ""
        else :
            if code.endswith(".py") and os.path.isfile(code):   #code가 .py 파일이라면 .py의 코드를를 읽어오기.
                with open(code, "r", encoding="utf-8") as code_file:
                    self.code = code_file.read()
            else:   # code가 파일이 아니라면, 저장.
                self.code = code

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
        
    def check_grammar(self, output_file_path: str = "pylint_output.txt") -> None:
        if not self.code:
            print("No code to check.")
            return
        else:
            print("Checking code grammar...")
            run_pylint_to_file(self.code, output_file_path)
            print(f"Grammar check completed. Output saved to {output_file_path}")
            with open(output_file_path, "r", encoding="utf-8") as file:
                print(file.read())
        

     
    
def run_pylint_to_file(code: str, output_file_path: str = "pylint_output.txt") -> None:
    """Run pylint on the provided code and save the output to a file."""
    import sys
    from pylint.lint import Run
    import os
    tmp_file_path = "tmp_code.py"
    #무조건 문자열만 받아야 함. .py 파일 받지 않음.
    with open(tmp_file_path, "w", encoding="utf-8") as tmp_file:
            tmp_file.write(code)
        
    # Run pylint on the temporary file
    try:
        Run([tmp_file_path, f"--output={output_file_path}", "--msg-template='{line}:{column}/{msg_id}: {msg}'"], exit=False)
        _revise_code_with_pylint (tmp_file_path, output_file_path)
        #transfer tmp_file to output_file
        with open(output_file_path, "w", encoding="utf-8") as output_file:
            with open(tmp_file_path, "r", encoding="utf-8") as tmp_file:
                output_file.write(tmp_file.read())
    except KeyboardInterrupt:
        sys.exit(1)
    finally:
        # Clean up the temporary file
        os.remove(tmp_file_path)
    
# pylint_output 파일과 temp_code 파일의 내용을 비교. temp_code 파일의 내용에 pylint_output 파일의 내용을 추가하는 함수
def _revise_code_with_pylint(code_file_path: str, output_file_path: str = "pylint_output.txt") -> None:
    """Revise the code based on pylint output."""
    with open(output_file_path, "r", encoding="utf-8") as file:
        output_lines = file.readlines()
    for i, line in enumerate(output_lines):
        if i == 0: # 첫번째 라인 스킵
            continue
        if not line.strip():  # 빈 줄은 건너뜀
            break
        _add_string_to_line_in_file(code_file_path, int(line.split(':')[0])-1, f"{i}번째 답변") #pylint_output.line은 1부터 시작. python 코드 line은 0부터 시작.

def _add_string_to_line_in_file(file_path: str, line_number: int, add_string: str) -> None:
    """Add a string to a specific line in a file."""
    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()

    if 0 <= line_number < len(lines):
        lines[line_number] = lines[line_number].rstrip() + " #" + add_string + "\n"

    with open(file_path, "w", encoding="utf-8") as file:
        file.writelines(lines)
   