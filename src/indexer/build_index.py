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
    embeddings = []
    
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
                'concepts': doc_info['concepts']
            })
            embeddings.append(doc_info['embedding'])

            cnt += 1
            if cnt > 2:
                break
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    # Create FAISS index
    embedding_dim = len(embeddings[0])
    index = faiss.IndexFlatL2(embedding_dim)
    embeddings_np = np.array(embeddings).astype('float32')
    index.add(embeddings_np)
    
    # Save index and metadata
    VECTOR_STORE_PATH = DATA_DIR / "vector_store"
    VECTOR_STORE_PATH.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(VECTOR_STORE_PATH / "questions.index"))
    
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