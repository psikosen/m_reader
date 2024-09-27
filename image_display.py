#  Image Display

import math
import streamlit as st

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
