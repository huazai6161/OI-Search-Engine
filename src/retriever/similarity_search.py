import faiss
import numpy as np
import pickle
import json
from typing import List, Dict
from pathlib import Path
from openai import OpenAI
from src.indexer.document_processor import DocumentProcessor
from config import OPENAI_API_KEY, VECTOR_STORE_PATH, EMBEDDING_MODEL

class SimilaritySearcher:
    def __init__(self, api_key = OPENAI_API_KEY):

        self.document_processor = DocumentProcessor()
        self.client = OpenAI(api_key=api_key)

        # print(str(VECTOR_STORE_PATH / "questions.index"))
        
        # Load the FAISS index
        self.question_index = faiss.read_index(str(VECTOR_STORE_PATH / "questions.index"))
        self.concept_index = faiss.read_index(str(VECTOR_STORE_PATH / "concepts.index"))
        self.summary_index = faiss.read_index(str(VECTOR_STORE_PATH / "summary.index"))
        
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
    
    def search(self, question: str, k: int = 5, solution = "", concepts: List[str] = None) -> List[Dict]:
        """Search for similar questions
        
        Args:
            query (str): The query text
            k (int): Number of results to return
            concepts (List[str], optional): Filter by concepts
            
        Returns:
            List[Dict]: List of similar questions
        """
        # Get concepts
        extracted_concepts = self.document_processor._extract_concepts(question, solution)

        # Get summary
        summary = self.document_processor._extract_summary(question, solution)
        
        # Get embeddings
        sorted_extracted_concepts = sorted(extracted_concepts)
        concepts_embedding = self._get_embedding(' '.join(sorted_extracted_concepts))
        concepts_embedding = concepts_embedding.reshape(1, -1)

        summary_embedding = self._get_embedding(summary)
        summary_embedding = summary_embedding.reshape(1, -1)

        question_embedding = self._get_embedding(question)
        question_embedding = question_embedding.reshape(1, -1)
        
        # Search in FAISS index
        question_distances, question_indices = self.question_index.search(question_embedding, k)  # Get more results for filtering
        concept_distances, concept_indices = self.concept_index.search(concepts_embedding, k)
        summary_distances, summary_indices = self.summary_index.search(summary_embedding, k)
        
        # Get metadata for the results
        question_results = []
        for idx in question_indices[0]:
            if idx < len(self.metadata):  # Ensure valid index
                doc = self.metadata[idx]
                # Filter by concepts if provided
                if concepts and not any(c in doc.get('concepts', []) for c in concepts):
                    continue
                question_results.append(doc)
                if len(question_results) >= k:  # Stop after getting k filtered results
                    break
        
        return question_results[:k]  # Ensure we return at most k results
    
if __name__ == "__main__":
    retriever = SimilaritySearcher()

    list = retriever.search("现在政府决定在公路上增设一些路标，使得公路的“空旷指数”最小。他们请求你设计一个程序计算能达到的最小值是多少。请注意，公路的起点和终点保证已设有路标，公路的长度为整数，并且原有路标和新设路标都必须距起点整数个单位距离。")

    print(list)