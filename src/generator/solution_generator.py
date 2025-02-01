from openai import OpenAI
from typing import List, Dict
from config import COMPLETION_MODEL
from config import OPENAI_API_KEY

class SolutionGenerator:
    def __init__(self, api_key: str):
        """Initialize the solution generator"""
        self.client = OpenAI(api_key=api_key)
    
    def generate(self, question: str, similar_questions: List[Dict]) -> str:
        """Generate solution based on similar questions"""
        # Prepare context from similar questions
        context = self._prepare_context(similar_questions)
        
        prompt = f"""参考题解:{context}
        
        你是一位信息竞赛专家。请根据一道新的题目和相似的参考题目,生成详细的解答。请按以下格式:

        1. 问题理解
        - 分析问题的关键点
        - 识别约束条件和边界情况

        2. 解题思路  
        - 详细解释解题策略
        - 分析为什么这个方法是最优的

        3. C++代码实现

        4. 复杂度分析
        - 时间复杂度及其解释
        - 空间复杂度及其解释

        新题目:
        {question}

        请按照以上格式生成解答:
        """
        
        response = self.client.chat.completions.create(
            model=COMPLETION_MODEL,
            messages=[
                {"role": "system", "content": "你是一位信息学竞赛教练。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=2000
        )
        
        return response.choices[0].message.content
    
    def _prepare_context(self, similar_questions: List[Dict]) -> str:
        """Prepare context from similar questions"""
        context = ""
        for i, q in enumerate(similar_questions, 1):  # Use top 2 similar questions
            context += f"\nReference {i}:\n"
            context += f"Question: {q['question']}\n"
            context += f"Solution:\n{q['solution']}\n"
            context += f"Concepts: {', '.join(q['concepts'])}\n"
            context += f"Summary: {q['summary']}\n"
            context += "-" * 80 + "\n"
        return context

def main():
    # Test question
    test_question = "Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target."
    
    # Sample similar questions for testing
    similar_questions = [
        {
            "question": "Given a sorted array of integers, find two numbers that sum to a target value",
            "solution": """def twoSum(nums, target):
    left, right = 0, len(nums)-1
    while left < right:
        curr_sum = nums[left] + nums[right]
        if curr_sum == target:
            return [left, right]
        elif curr_sum < target:
            left += 1
        else:
            right -= 1
    return []""",
            "concepts": ["Two Pointers", "Binary Search"]
        },
        {
            "question": "Find three numbers in an array that sum to zero",
            "solution": """def threeSum(nums):
    nums.sort()
    result = []
    for i in range(len(nums)-2):
        if i > 0 and nums[i] == nums[i-1]:
            continue
        left, right = i+1, len(nums)-1
        while left < right:
            total = nums[i] + nums[left] + nums[right]
            if total == 0:
                result.append([nums[i], nums[left], nums[right]])
                while left < right and nums[left] == nums[left+1]:
                    left += 1
                while left < right and nums[right] == nums[right-1]:
                    right -= 1
                left += 1
                right -= 1
            elif total < 0:
                left += 1
            else:
                right -= 1
    return result""",
            "concepts": ["Two Pointers", "Sorting"]
        }
    ]

    # Initialize generator
    generator = SolutionGenerator(OPENAI_API_KEY)
    
    # Generate solution
    try:
        solution = generator.generate(test_question, similar_questions)
        print("Generated Solution:")
        print("-" * 80)
        print(solution)
    except Exception as e:
        print(f"Error generating solution: {e}")

if __name__ == "__main__":
    main()