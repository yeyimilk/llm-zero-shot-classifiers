
from llm.chatgpt import run_chatgpt
from llm.fschat import run_fschat
from llm.qwen import run_qwen
from llm.gemini import run_gemini
from config import run_args

if __name__ == "__main__":
    run_args()
    run_gemini()
    run_chatgpt()
    run_fschat()
    run_qwen()