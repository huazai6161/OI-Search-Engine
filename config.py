from dotenv import load_dotenv
import os
from pathlib import Path

# Load .env file
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Project paths
ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"
QUESTIONS_DIR = DATA_DIR / "questions"
VECTOR_STORE_PATH = DATA_DIR / "vector_store"

# Embedding model
EMBEDDING_MODEL = "text-embedding-3-large"