import faiss
import numpy as np
import pickle
import json
from typing import List, Dict
from pathlib import Path
from openai import OpenAI
from config import OPENAI_API_KEY, VECTOR_STORE_PATH, EMBEDDING_MODEL

class SimilaritySearcher:
    def __init__(self, api_key = OPENAI_API_KEY):
        self.client = OpenAI(api_key=api_key)
        
        # Load the FAISS index
        self.index = faiss.read_index(VECTOR_STORE_PATH / "questions.index")
        
        # Load metadata
        with open(VECTOR_STORE_PATH / "metadata.pkl", 'rb') as f:
            self.metadata = pickle.load(f)
            
        with open(VECTOR_STORE_PATH / "concept_mapping.json", 'r') as f:
            self.concept_mapping = json.load(f)
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for the search text"""
        response = self.client.embeddings.create(
            input=text,
            model=EMBEDDING_MODEL
        )
        return np.array(response.data[0].embedding, dtype=np.float32)
    
    def search(self, query: str, k: int = 5, concepts: List[str] = None) -> List[Dict]:
        """Search for similar questions
        
        Args:
            query (str): The query text
            k (int): Number of results to return
            concepts (List[str], optional): Filter by concepts
            
        Returns:
            List[Dict]: List of similar questions
        """
        # Get embedding for the query
        query_embedding = self._get_embedding(query)
        query_embedding = query_embedding.reshape(1, -1)
        
        # Search in FAISS index
        distances, indices = self.index.search(query_embedding, k * 2)  # Get more results for filtering
        
        # Get metadata for the results
        results = []
        for idx in indices[0]:
            if idx < len(self.metadata):  # Ensure valid index
                doc = self.metadata[idx]
                # Filter by concepts if provided
                if concepts and not any(c in doc.get('concepts', []) for c in concepts):
                    continue
                results.append(doc)
                if len(results) >= k:  # Stop after getting k filtered results
                    break
        
        return results[:k]  # Ensure we return at most k results
    
if __name__ == "__main__":
    retriever = SimilaritySearcher("")

    list = retriever.search("现在政府决定在公路上增设一些路标，使得公路的“空旷指数”最小。他们请求你设计一个程序计算能达到的最小值是多少。请注意，公路的起点和终点保证已设有路标，公路的长度为整数，并且原有路标和新设路标都必须距起点整数个单位距离。")

    print(list)