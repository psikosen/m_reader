# Document Ingestion and Processing

import os
import re
import pdfplumber
import fitz   
from PIL import Image
import io
import streamlit as st
from database import ImageEntry
from embedding import serialize_embedding
from openaicli import client

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
#need to mk faster
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

 
            from retrieval import chunk_texts_with_page_numbers  
            chunks = chunk_texts_with_page_numbers([cleaned])

            # Insert chunks into sqlits
            from retrieval import get_embedding   
            for chunk in chunks:
                embedding = get_embedding(chunk['text'])
                if embedding:
                    embedding_str = serialize_embedding(embedding)
                    from database import Chunk  
                    chunk_entry = Chunk(
                        pdf_file=file,
                        page_number=chunk['page_number'],
                        text=chunk['text'],
                        embedding=embedding_str
                    )
                    db_session.add(chunk_entry)
    db_session.commit()
    return manuals_text, manuals_images
