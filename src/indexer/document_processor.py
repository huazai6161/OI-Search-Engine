import os
from typing import Dict, List, Tuple
from pathlib import Path
from openai import OpenAI
import json
from config import OPENAI_API_KEY, EMBEDDING_MODEL, DATA_DIR

class DocumentProcessor:
    def __init__(self):
        """Initialize the document processor
        
        Args:
            api_key (str): OpenAI API key
            embedding_model (str): Name of the embedding model to use
        """
        self.api_key = OPENAI_API_KEY
        self.embedding_model = EMBEDDING_MODEL
        self.client = OpenAI(api_key=self.api_key)
        
    def _extract_question_and_solution(self, content: str) -> Tuple[str, str]:
        """Extract question and solution from file content
        
        Args:
            content (str): Full content of the Python file
            
        Returns:
            Tuple[str, str]: (question, solution)
        """
        # Split content at class Solution
        parts = content.split("class Solution")
        if len(parts) != 2:
            return "", ""
            
        question = parts[0].strip()
        solution = "class Solution" + parts[1].strip()
        return question, solution
    
    def _extract_concepts(self, question: str, solution: str) -> List[str]:
        """Extract LeetCode concepts using OpenAI
        
        Args:
            question (str): The question text
            solution (str): The solution code
            
        Returns:
            List[str]: List of concepts
        """
        with open(DATA_DIR / 'IOI_outline/NOI.json', 'r', encoding='utf-8') as file:
            syllabus = json.load(file)

        prompt = f"""知识点大纲：{syllabus}
        题目描述：{question}
        题解：{solution}
        请阅读题目、题解、以及知识点大纲，分析该题目所考察的大纲知识点（最多5个），提取这些知识点，并用','分隔输出。不要输出其他内容。
        """
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "你是一个信息学竞赛专家。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=1000
        )
        
        concepts = response.choices[0].message.content.strip().split(',')

        print(concepts)

        return [concept.strip() for concept in concepts]
    
    def process_file(self, file_path: Path) -> Dict:
        """Process a single LeetCode question file
        
        Args:
            file_path (Path): Path to the Python file
            
        Returns:
            Dict: Processed document information
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract question number from filename
        question_number = os.path.basename(file_path).replace('.md', '')
        
        # Extract question and solution
        question, solution = content, ""
        
        # Get concepts
        concepts = self._extract_concepts(question, solution)
        
        # Get embedding for the question
        question_embedding = self._get_embedding(question + f"\nconcept: {','.join(concepts)}")
        
        return {
            'id': question_number,
            'file_path': str(file_path),
            'question': question,
            'solution': solution,
            'concepts': concepts,
            'embedding': question_embedding
        }
    
    def _get_embedding(self, text: str) -> List[float]:
        """Get OpenAI embedding for text
        
        Args:
            text (str): Text to get embedding for
            
        Returns:
            List[float]: Embedding vector
        """
        response = self.client.embeddings.create(
            input=text,
            model=self.embedding_model
        )
        
        return response.data[0].embedding