import streamlit as st
import openai
from openai import OpenAI
import os
from pathlib import Path
from src.retriever.similarity_search import SimilaritySearcher
from src.generator.solution_generator import SolutionGenerator
from config import *

# Page config
st.set_page_config(
    page_title="Code Solution Tutor",
    page_icon="🧮",
    layout="wide"
)

def render_sidebar():
    """Render the sidebar with filtering options"""
    st.sidebar.header("Settings")
    
    # Filter by concepts
    if st.sidebar.checkbox("Filter by Concepts"):
        selected_concepts = st.sidebar.multiselect(
            "Select Concepts",
            options=LEETCODE_CONCEPTS
        )
        return {"concepts": selected_concepts}
    return {"concepts": []}

def initialize_components():
    """Initialize necessary components"""
    if "searcher" not in st.session_state:
        st.session_state.searcher = SimilaritySearcher(OPENAI_API_KEY)
    if "solution_generator" not in st.session_state:
        st.session_state.solution_generator = SolutionGenerator(OPENAI_API_KEY)
    
    return st.session_state.searcher, st.session_state.solution_generator

def display_solution(solution: str, similar_questions: list):
    """Display the solution and similar questions in columns"""
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Generated Solution")

        st.markdown(solution)
    
    with col2:
        st.subheader("Similar Questions")
        for idx, question in enumerate(similar_questions):
            with st.expander(f"Question {question['id']}"):
                st.markdown(question['question'])
                st.markdown("**Concepts:**")
                st.markdown(", ".join(question['concepts']))
                
                if st.button(f"Show Solution {idx}", key=f"sol_{idx}"):
                    st.code(question['solution'], language="python")
        
        st.subheader("Concepts Used")
        all_concepts = set()
        for q in similar_questions:
            all_concepts.update(q['concepts'])
        st.markdown(", ".join(sorted(all_concepts)))

def main():
    st.title("OI Search Engine 🧮")
    st.markdown("""
    Enter your question below, and I'll help you find similar questions and generate a solution!
    """)
    
    # Initialize components
    searcher, solution_generator = initialize_components()
    
    # Sidebar filters
    filters = render_sidebar()
    
    # Main input area
    question = st.text_area(
        "Enter your question:",
        height=200,
        placeholder="Paste here...(markdown format)"
    )
    
    col1, col2 = st.columns([1, 5])
    with col1:
        search_button = st.button("Generate Solution")
    with col2:
        if search_button:
            with st.spinner('🔍 Searching similar questions and generating solution...'):
                try:
                    # Search similar questions
                    similar_questions = searcher.search(question, concepts=filters["concepts"])
                    
                    # Generate solution
                    solution = solution_generator.generate(
                        question=question,
                        similar_questions=similar_questions
                    )
                    
                    # Display results
                    display_solution(solution, similar_questions)
                    
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    st.error("Please try again or contact support if the problem persists.")

if __name__ == "__main__":
    main()