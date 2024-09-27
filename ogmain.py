import os
import re
import faiss
import numpy as np
import streamlit as st
from openai import OpenAI
from PIL import Image
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pdfplumber
import fitz   
import io
import math
from sqlalchemy import create_engine, Column, Integer, String, Text, Index
from sqlalchemy.orm import declarative_base, sessionmaker
from openai import OpenAI

# ---------------------------- Database Setup ---------------------------- # 
Base = declarative_base()

class Chunk(Base):
    __tablename__ = 'chunks'
    id = Column(Integer, primary_key=True, autoincrement=True)
    pdf_file = Column(String, nullable=False)
    page_number = Column(Integer, nullable=True)
    text = Column(Text, nullable=False)
    embedding = Column(Text, nullable=False)   

class FAQ(Base):
    __tablename__ = 'faqs'
    id = Column(Integer, primary_key=True, autoincrement=True)
    question = Column(Text, nullable=False, unique=True)
    answer = Column(Text, nullable=False)

class ImageEntry(Base):
    __tablename__ = 'images'
    id = Column(Integer, primary_key=True, autoincrement=True)
    pdf_file = Column(String, nullable=False)
    page_number = Column(Integer, nullable=False)
    image_path = Column(String, nullable=False)  
    caption = Column(Text, nullable=True)    #doenst work perfect yet

# Create indexes for faster retrieval
Index('idx_pdf_file', Chunk.pdf_file)
Index('idx_page_number', Chunk.page_number)
Index('idx_question', FAQ.question)
Index('idx_image_pdf_file', ImageEntry.pdf_file)
Index('idx_image_page_number', ImageEntry.page_number)
Index('idx_image_caption', ImageEntry.caption)

def init_db(db_path="sqlite:///manuals.db"):
    engine = create_engine(db_path)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    return Session()

# ------------------------ Embedding Serialization ----------------------- #

def serialize_embedding(embedding):
    return ",".join(map(str, embedding))

def deserialize_embedding(embedding_str):
    try:
        return list(map(float, embedding_str.split(',')))
    except ValueError:
        return []

# ------------------------ LM STUDIO Client Setup --------------------------- #
 
 
client = OpenAI(base_url="http://localhost:8002/v1", api_key="lm-studio")

# ------------------- Document Ingestion and Processing ------------------ #

def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_number, page in enumerate(pdf.pages, start=1):
                page_text = page.extract_text()
                if page_text:
                    text += f"Page {page_number}:\n{page_text}\n"
    except Exception as e:
        st.error(f"Failed to extract text from {pdf_path}: {e}")
    return text

def extract_images_from_pdf(pdf_path, pdf_file, db_session):
    images = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_number, page in enumerate(pdf.pages, start=1):
                extracted = False
                page_text = page.extract_text()
                for image_number, image in enumerate(page.images, start=1):
                    if 'bbox' in image:
                        img_bbox = image['bbox']
                        try:
                            img_obj = page.within_bbox(img_bbox).to_image()
                            img_pil = img_obj.to_pil_image().convert("RGB")

                            image_dir = os.path.join("extracted_images", os.path.splitext(os.path.basename(pdf_path))[0])
                            os.makedirs(image_dir, exist_ok=True)
                            image_filename = f"page_{page_number}_image_{image_number}.jpg"
                            image_path = os.path.join(image_dir, image_filename)

                            img_pil.save(image_path)

                            # Attempt to extract caption
                            caption = extract_caption(page_text, img_bbox)

                            image_entry = ImageEntry(
                                pdf_file=os.path.basename(pdf_path),
                                page_number=page_number,
                                image_path=image_path,
                                caption=caption
                            )
                            images.append(img_pil)
                            db_session.add(image_entry)
                            extracted = True
                        except Exception as e:
                            st.warning(f"Failed to process image {image_number} on page {page_number}: {e}")
                    else:
                        available_keys = list(image.keys())
                        st.warning(
                            f"Image {image_number} on page {page_number} does not have a 'bbox' and was skipped. "
                            f"Available keys: {available_keys}"
                        )

                if not extracted:
                    st.info(f"No images with 'bbox' found on page {page_number} with pdfplumber. Trying PyMuPDF...")
                    images += extract_images_with_pymupdf(pdf_path, pdf_file, page_number, db_session, page_text)

            db_session.commit()
    except Exception as e:
        st.error(f"Failed to extract images from {pdf_path} with pdfplumber: {e}")
    return images

def extract_caption(page_text, img_bbox):
    captions = re.findall(r'(Figure\s+\d+[-.]\d+.*?)(?=\n|$)', page_text, re.IGNORECASE)
    if captions:
        return captions[0].strip()
    return None

def extract_images_with_pymupdf(pdf_path, pdf_file, page_number, db_session, page_text):
    images = []
    try:
        doc = fitz.open(pdf_path)
        page = doc[page_number - 1]   
        image_list = page.get_images(full=True)
        for image_index, img in enumerate(image_list, start=1):
            xref = img[0]
            try:
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                img_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")

                image_dir = os.path.join("extracted_images", os.path.splitext(os.path.basename(pdf_path))[0])
                os.makedirs(image_dir, exist_ok=True)
                image_filename = f"page_{page_number}_image_{image_index}.jpg"
                image_path = os.path.join(image_dir, image_filename)

                img_pil.save(image_path)

                # Try to extract caption (not working yet)
                caption = extract_caption(page_text, None)

                image_entry = ImageEntry(
                    pdf_file=os.path.basename(pdf_path),
                    page_number=page_number,
                    image_path=image_path,
                    caption=caption
                )
                images.append(img_pil)
                db_session.add(image_entry)
            except Exception as e:
                st.warning(f"Failed to extract image {image_index} on page {page_number} with PyMuPDF: {e}")
        db_session.commit()
    except Exception as e:
        st.error(f"Failed to extract images with PyMuPDF from {pdf_path}: {e}")
    return images

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'(.)\1+', r'\1', text)   
    return text.strip()

def load_and_process_manuals(manuals_dir, db_session):
    manuals_text = []
    manuals_images = []
    for file in os.listdir(manuals_dir):
        if file.lower().endswith(".pdf"):
            pdf_path = os.path.join(manuals_dir, file)
            st.info(f"Processing {file}...")
 
            text = extract_text_from_pdf(pdf_path)
            cleaned = clean_text(text)
            manuals_text.append(cleaned)
 
            images = extract_images_from_pdf(pdf_path, file, db_session)

            if images:
                st.success(f"Extracted {len(images)} images from {file}.")
            else:
                st.info(f"No images found in {file}.")

            manuals_images.extend(images)

            # Split text into chunks with page #
            chunks = chunk_texts_with_page_numbers([cleaned])

            # Insert chunks into sqll
            for chunk in chunks:
                embedding = get_embedding(chunk['text'])
                if embedding:
                    embedding_str = serialize_embedding(embedding)
                    chunk_entry = Chunk(
                        pdf_file=file,
                        page_number=chunk['page_number'],
                        text=chunk['text'],
                        embedding=embedding_str
                    )
                    db_session.add(chunk_entry)
    db_session.commit()
    return manuals_text, manuals_images

def chunk_texts_with_page_numbers(texts, chunk_size=500, chunk_overlap=50):
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

# ----------------------------- Embedding --------------------------------- #

def get_embedding(text, model="nomic-ai/nomic-embed-text-v1.5-GGUF"):
    text = text.replace("\n", " ")
    try:
        response = client.embeddings.create(input=[text], model=model)
        embedding = response.data[0].embedding
        return embedding
    except Exception as e:
        st.error(f"Failed to get embedding for text: {e}")
        return None

def create_faiss_index(embeddings, dimension, index_path):
    try:
        index = faiss.IndexFlatL2(dimension)  # Suitable for small datasets need to find one for big ones
        index.add(np.array(embeddings).astype('float32'))
        faiss.write_index(index, index_path)
        st.success("FAISS index created and saved to disk.")
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

# -------------------------- FAQ Caching ---------------------------------- #

def get_cached_faq(db_session, question):
    faq = db_session.query(FAQ).filter(FAQ.question == question).first()
    if faq:
        return faq.answer
    return None

def cache_faq(db_session, question, answer):
    faq = FAQ(question=question, answer=answer)
    db_session.add(faq)
    try:
        db_session.commit()
    except Exception as e:
        db_session.rollback()
        st.error(f"Failed to cache FAQ: {e}")

# ---------------------------- Retrieval ----------------------------------- #

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

# ------------------------- Answering Questions --------------------------- #

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
            model="/Qwen1.5-0.5BQwen-Chat-GGUF",
            messages=messages,
            temperature=0.2,
            max_tokens=500
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

# ------------------------- Image Display --------------------------------- #

def display_images(images, images_per_page=10):
    total_images = len(images)
    total_pages = math.ceil(total_images / images_per_page)
    if total_pages == 0:
        st.info("No images to display.")
        return

    page = st.sidebar.number_input("Image Page", min_value=1, max_value=total_pages, value=1, step=1)
    start_idx = (page - 1) * images_per_page
    end_idx = start_idx + images_per_page
    for idx, img in enumerate(images[start_idx:end_idx], start=start_idx + 1):
        st.image(img, caption=f"Extracted Image {idx}", use_column_width=True)

# ----------------------------- Main App ----------------------------------- #

def main():
    st.title("DAF Manuals Q&A")
 
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

    #  rty load existing FAISS index and chunks from the database
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
                    return  # Exit  

            st.sidebar.success("Manuals loaded and indexed successfully!")

            if images:
                st.sidebar.header("Extracted Images")
                display_images(images, images_per_page=10)
            else:
                st.sidebar.info("No images extracted from the manuals.")
        else:
            st.sidebar.error("Invalid directory path.")
 
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

    # Display History
    if st.session_state.history:
        st.markdown("---")
        st.header("History")
        for idx, interaction in enumerate(st.session_state.history, start=1):
            with st.expander(f"Q{idx}: {interaction['query']}"):
                st.write(f"A{idx}: {interaction['answer']}")

if __name__ == "__main__":
    main()
