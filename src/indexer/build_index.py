import faiss
import numpy as np
import json
import pickle
from pathlib import Path
from document_processor import DocumentProcessor
from tqdm import tqdm

def build_index():


    processor = DocumentProcessor()
    
    # Process all question files
    all_docs = []
    question_embeddings = []
    concept_embeddings = []
    summary_embeddings = []
    
    # Walk through all directories in questions
    ROOT_DIR = Path(__file__).parent.parent.parent
    DATA_DIR = ROOT_DIR / "data"
    QUESTIONS_DIR = DATA_DIR / "questions" / "Luogu" / "statement"

    print(QUESTIONS_DIR)

    cnt = 0
    for file_path in tqdm(QUESTIONS_DIR.glob("*.md")):
        print(f"Processing {file_path}")
        try:
            doc_info = processor.process_file(file_path)
            all_docs.append({
                'id': doc_info['id'],
                'file_path': doc_info['file_path'],
                'question': doc_info['question'],
                'solution': doc_info['solution'],
                'concepts': doc_info['concepts'],
                'summary': doc_info['summary']
            })
            question_embeddings.append(doc_info['question_embedding'])
            concept_embeddings.append(doc_info['conscepts_embedding'])
            summary_embeddings.append(doc_info['summary_embedding'])

            # cnt += 1
            # if cnt > 2:
            #     break
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    # Create FAISS index
    question_embedding_dim = len(question_embeddings[0])
    question_index = faiss.IndexFlatL2(question_embedding_dim)
    question_embeddings_np = np.array(question_embeddings).astype('float32')
    question_index.add(question_embeddings_np)

    concept_embedding_dim = len(concept_embeddings[0])
    concept_index = faiss.IndexFlatL2(concept_embedding_dim)
    concept_embeddings_np = np.array(concept_embeddings).astype('float32')
    concept_index.add(concept_embeddings_np)

    summary_embedding_dim = len(summary_embeddings[0])
    summary_index = faiss.IndexFlatL2(summary_embedding_dim)
    summary_embeddings_np = np.array(summary_embeddings).astype('float32')
    summary_index.add(summary_embeddings_np)
    
    # Save index and metadata
    VECTOR_STORE_PATH = DATA_DIR / "vector_store"
    VECTOR_STORE_PATH.mkdir(parents=True, exist_ok=True)
    faiss.write_index(question_index, str(VECTOR_STORE_PATH / "questions.index"))
    faiss.write_index(concept_index, str(VECTOR_STORE_PATH / "concepts.index"))
    faiss.write_index(summary_index, str(VECTOR_STORE_PATH / "summary.index"))
    
    with open(VECTOR_STORE_PATH / "metadata.pkl", 'wb') as f:
        pickle.dump(all_docs, f)
    
    # Create concept mapping
    concept_to_questions = {}
    for doc in all_docs:
        for concept in doc['concepts']:
            if concept not in concept_to_questions:
                concept_to_questions[concept] = []
            concept_to_questions[concept].append(doc['id'])
    
    with open(VECTOR_STORE_PATH / "concept_mapping.json", 'w') as f:
        json.dump(concept_to_questions, f, ensure_ascii=False)

if __name__ == "__main__":
    build_index()