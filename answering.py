#  Answering Questions

import re
import numpy as np
import streamlit as st
from faq import get_cached_faq, cache_faq
from embedding import get_embedding, deserialize_embedding
from retrieval import retrieve_chunks_from_db
from openaicli import client

modelname ="/Qwen1.5-0.5BQwen-Chat-GGUF"

def clean_llm_response(response):
    sentences = re.split(r'(?<=[.!?]) +', response)
    seen = set()
    cleaned_sentences = []
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence and sentence not in seen:
            cleaned_sentences.append(sentence)
            seen.add(sentence)
    return ' '.join(cleaned_sentences)

def answer_question(query, index, chunks, k=5, db_session=None):
    if db_session:
        cached_answer = get_cached_faq(db_session, query)
        if cached_answer:
            st.info("Retrieved answer from FAQ cache.")
            st.session_state.history.append({
                'query': query,
                'answer': cached_answer
            })
            return cached_answer

    if index is None or not chunks:
        st.warning("FAISS index or text chunks are not available. Proceeding without context.")
        context = ""
    else:
        query_embedding = get_embedding(query)
        if query_embedding is None:
            return ""

        try:
            query_embedding_np = np.array([query_embedding]).astype('float32')
            D, I = index.search(query_embedding_np, k)
            relevant_chunks = [chunks[i]['text'] for i in I[0]]
            relevant_page_numbers = [chunks[i]['page_number'] for i in I[0]]
        except Exception as e:
            st.error(f"FAISS search failed: {e}")
            return ""

        context = "\n\n".join(relevant_chunks)

    messages = [
        {"role": "system", "content": "You are an assistant specialized in DAF technical manuals and maintenance procedures. Provide clear, concise, and accurate answers to user queries based on the provided context. If no context is provided, try to answer the question to the best of your ability."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
    ]

    try:
        completion = client.chat.completions.create(
            model=modelname,
            messages=messages,
            temperature=0.2,
            max_tokens=32000
        )
        answer = completion.choices[0].message.content.strip()
        cleaned_answer = clean_llm_response(answer)

        if db_session:
            cache_faq(db_session, query, cleaned_answer)

        st.session_state.history.append({
            'query': query,
            'answer': cleaned_answer
        })

    except Exception as e:
        st.error(f"Chat completion failed: {e}")
        cleaned_answer = ""

    return cleaned_answer
