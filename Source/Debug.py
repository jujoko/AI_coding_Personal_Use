class Debug: # Debugging utility class
    def __init__(self, code_file=None):
        if code_file is not None:
            with open(code_file, "r") as f: 
                self.code = f.read() 
        self.code_file = code_file
    def get_code(self, code):
        self.code_file = code
