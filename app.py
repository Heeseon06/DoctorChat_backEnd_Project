from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from fastapi.middleware.cors import CORSMiddleware
from model import get_input_token_length, run

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 또는 허용할 특정 도메인 리스트를 넣으세요
    allow_credentials=True,
    allow_methods=["*"],  # 모든 메서드를 허용합니다. 필요에 따라 ["POST"] 등으로 제한할 수 있습니다.
    allow_headers=["*"],  # 모든 헤더를 허용합니다. 필요에 따라 특정 헤더만 허용할 수 있습니다.
)


DEFAULT_SYSTEM_PROMPT = """아래는 매우 전문적인 의사와 환자의 진료 기록이다."""
MAX_MAX_NEW_TOKENS = 2048
DEFAULT_MAX_NEW_TOKENS = 1024
MAX_INPUT_TOKEN_LENGTH = 1500
class InputData(BaseModel):
    message: str
    history: list
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    temperature: float = 0.8
    top_p: float = 0.95
    top_k: int = 50
    repetition_penalty: float = 1.2
@app.post("/generate")
async def generate(input_data: InputData):
    if input_data.max_new_tokens > MAX_MAX_NEW_TOKENS:
        raise HTTPException(status_code=400, detail="max_new_tokens too large")
    try:
        history = input_data.history[:-1]
        generator = run(input_data.message, history, DEFAULT_SYSTEM_PROMPT, input_data.max_new_tokens, input_data.temperature, input_data.top_p, input_data.top_k, input_data.repetition_penalty)
        response = ""
        for resp in generator:
            response = resp
        response = response.strip('</끝>')
        print(response)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)