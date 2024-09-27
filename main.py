#  Main Application

import os
import streamlit as st
from database import init_db
from pdf_ingestion import load_and_process_manuals
from retrieval import load_or_create_faiss_index, retrieve_chunks_from_db
from answering import answer_question
from image_display import display_images
from embedding import deserialize_embedding

def main():
    st.title("DAF Manuals Q&A")

    # Initialize Session 
    if 'index' not in st.session_state:
        st.session_state.index = None
    if 'chunks' not in st.session_state:
        st.session_state.chunks = []
    if 'images' not in st.session_state:
        st.session_state.images = []
    if 'loaded' not in st.session_state:
        st.session_state.loaded = False
    if 'history' not in st.session_state:
        st.session_state.history = []
 
    db_session = init_db()

    # try to to load existing FAISS index and chunks from the database
    if os.path.exists("daf_manuals_faiss.index"):
        st.session_state.chunks = retrieve_chunks_from_db(db_session)
        if st.session_state.chunks:
            embeddings = [deserialize_embedding(chunk['embedding']) for chunk in st.session_state.chunks]
            if all(len(emb) > 0 for emb in embeddings):
                dimension = len(embeddings[0])
                st.session_state.index = load_or_create_faiss_index(embeddings, dimension)
                if st.session_state.index is not None:
                    st.sidebar.success("FAISS index and text chunks loaded successfully!")
                else:
                    st.sidebar.error("Failed to load FAISS index.")
            else:
                st.sidebar.warning("Some embeddings failed to deserialize. Please reprocess the manuals.")
        else:
            st.sidebar.warning("No text chunks found in the database.")
    else:
        st.sidebar.info("FAISS index not found.")
 
    st.session_state.loaded = True

    # Sidebar: Load Manuals
    st.sidebar.header("Load Manuals")
    manuals_dir = st.sidebar.text_input("Path to DAF manuals directory:", value="./manuals")
    if st.sidebar.button("Load and Process Manuals"):
        if os.path.isdir(manuals_dir):
            with st.spinner("Loading and processing manuals..."):
                try:
                    texts, images = load_and_process_manuals(manuals_dir, db_session)
                    if not texts:
                        st.warning("No text extracted from the manuals.")
                    st.session_state.chunks = retrieve_chunks_from_db(db_session)
                    if not st.session_state.chunks:
                        st.warning("No text chunks found after processing.")
                    else:
                        embeddings = [deserialize_embedding(chunk['embedding']) for chunk in st.session_state.chunks]
                        if all(len(emb) > 0 for emb in embeddings):
                            dimension = len(embeddings[0])
                            index = load_or_create_faiss_index(embeddings, dimension)
                            if index:
                                st.session_state.index = index
                        else:
                            st.warning("Some embeddings failed to deserialize. Please reprocess the manuals.")
                except Exception as e:
                    st.sidebar.error(f"An error occurred while processing manuals: {e}")
                    return   

            st.sidebar.success("Manuals loaded and indexed successfully!")

            if images:
                st.sidebar.header("Extracted Images")
                display_images(images, images_per_page=10)
            else:
                st.sidebar.info("No images extracted from the manuals.")
        else:
            st.sidebar.error("Invalid directory path.")

    # Enter query
    query = st.text_input("Enter your question about DAF manuals:")

    if st.button("Ask"):
        if not query.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Generating answer..."):
                try:
                    response = answer_question(
                        query,
                        st.session_state.index,
                        st.session_state.chunks,
                        k=5,
                        db_session=db_session
                    )
                    if response:
                        st.success("Answer:")
                        st.write(response)
                    else:
                        st.info("No answer could be generated.")
                except Exception as e:
                    st.error(f"An error occurred while generating the answer: {e}")

    # Display  History
    if st.session_state.history:
        st.markdown("---")
        st.header("History")
        for idx, interaction in enumerate(st.session_state.history, start=1):
            with st.expander(f"Q{idx}: {interaction['query']}"):
                st.write(f"A{idx}: {interaction['answer']}")

if __name__ == "__main__":
    main()   
