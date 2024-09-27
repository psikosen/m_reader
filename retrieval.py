# Retrieval and Chunking

import re
import faiss
import numpy as np
import streamlit as st
from embedding import deserialize_embedding, get_embedding
from database import Chunk, ImageEntry
from openaicli import client
#increase chunks if larger docs or more docs
def chunk_texts_with_page_numbers(texts, chunk_size=500, chunk_overlap=50):
    from langchain.text_splitter import RecursiveCharacterTextSplitter   
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = []
    for doc in texts:
        pages = re.split(r'Page\s+(\d+):\n', doc)
        for i in range(1, len(pages), 2):
            page_number = int(pages[i])
            page_text = pages[i+1]
            split_chunks = splitter.split_text(page_text)
            for split_chunk in split_chunks:
                chunks.append({
                    'page_number': page_number,
                    'text': split_chunk
                })
    return chunks

def retrieve_chunks_from_db(db_session):
    try:
        all_chunks = db_session.query(Chunk).order_by(Chunk.id).all()
        chunks = [
            {
                'page_number': chunk.page_number,
                'text': chunk.text,
                'embedding': chunk.embedding
            } for chunk in all_chunks
        ]
        return chunks
    except Exception as e:
        st.error(f"Failed to retrieve chunks from the database: {e}")
        return []

def get_images_for_query(db_session, query):
    images = db_session.query(ImageEntry).filter(ImageEntry.caption.ilike(f"%{query}%")).all()
    return images

def get_images_for_pages(db_session, page_numbers):
    images = db_session.query(ImageEntry).filter(ImageEntry.page_number.in_(page_numbers)).all()
    return images

def create_faiss_index(embeddings, dimension, index_path):
    try:
        index = faiss.IndexFlatL2(dimension)  # good need to find 1 for larger data setas 
        index.add(np.array(embeddings).astype('float32'))
        faiss.write_index(index, index_path)
        st.success("FAISS index created and saved.")
        return index
    except Exception as e:
        st.error(f"Failed to create FAISS index: {e}")
        return None

def load_or_create_faiss_index(embeddings, dimension, index_path="daf_manuals_faiss.index"):
    if os.path.exists(index_path):
        try:
            index = faiss.read_index(index_path)
            st.info("FAISS index loaded from disk.")
        except Exception as e:
            st.error(f"Failed to load FAISS index: {e}")
            index = create_faiss_index(embeddings, dimension, index_path)
    else:
        index = create_faiss_index(embeddings, dimension, index_path)
    return index
