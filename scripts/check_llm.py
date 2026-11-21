from pydantic import BaseModel
from llm import structured_llm


class Ping(BaseModel):
    message: str
    ok: bool


print(structured_llm(Ping).invoke("Reponds avec message='pong' et ok=true."))