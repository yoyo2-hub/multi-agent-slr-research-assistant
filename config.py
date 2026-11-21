import os
from dotenv import load_dotenv

load_dotenv()

OPENALEX_API_KEY = os.getenv("OPENALEX_API_KEY")
OPENALEX_EMAIL = os.getenv("OPENALEX_EMAIL", "")
YEAR_MIN = int(os.getenv("YEAR_MIN", "2018"))

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-20b")