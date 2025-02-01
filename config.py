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

# Completion model
COMPLETION_MODEL = "gpt-4o-mini"

# LeetCode concepts
LEETCODE_CONCEPTS = [
    "Array", "String", "Hash Table", "Dynamic Programming",
    "Math", "Depth-First Search", "Binary Search", "Binary Tree",
    "Two Pointers", "Breadth-First Search", "Tree", "Stack",
    "Greedy", "Backtracking", "Design", "Graph", "Linked List",
    "Heap", "Sliding Window", "Union Find", "Divide and Conquer",
    "Trie", "Recursion", "Queue"
]