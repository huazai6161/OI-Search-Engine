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

    def _extract_summary(self, question: str, solution: str) -> str:

        prompt = f"""题目描述：{question}
题解：{solution}
请总结以上算法竞赛题目的考察点，并以简明专业的语言描述。你的总结应包括：  
1. 题目涉及的主要算法和数据结构。
2. 题目要求考生掌握的关键技巧（题解有可能提到）。
3. 总结应简洁、直接，符合竞赛题解风格，不要使用冗余的描述。
4. 输出格式：“主要算法和数据结构：xxx；关键技巧：xxx。”
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

        content = response.choices[0].message.content

        return content
    
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
        请阅读题目、题解、以及知识点大纲，分析该题目所考察的最重要的大纲知识点（最多5个），提取这些知识点，并用','分隔输出。不要输出其他内容。
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
        question = content

        # solution
        solution_path = str(file_path).replace('/statement/', '/solution/')
        solution_path = Path(solution_path)

        print(solution_path)

        with open(solution_path, 'r', encoding='utf-8') as f:
            solution = f.read()
        
        # Get concepts
        concepts = self._extract_concepts(question, solution)

        # Get summary
        summary = self._extract_summary(question, solution)
        
        # Get embeddings
        question_embedding = self._get_embedding(question)
        sorted_concepts = sorted(concepts)
        concepts_embedding = self._get_embedding(' '.join(sorted_concepts))
        summary_embedding = self._get_embedding(summary)
        
        return {
            'id': question_number,
            'file_path': str(file_path),
            'question': question,
            'solution': solution,
            'concepts': concepts,
            'summary': summary,
            'question_embedding': question_embedding,
            'conscepts_embedding': concepts_embedding,
            'summary_embedding': summary_embedding
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