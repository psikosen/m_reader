
## Table of Contents

- [Features](#features) 
- [Installation](#installation) 
  - [Conda Setup](#conda-setup)
  - [Install Dependencies](#install-dependencies)
- [Usage](#usage)
  - [Running the Application](#running-the-application)  

 
## Features

- **PDF Text Extraction:** Extracts and cleans text from DAF manuals in PDF format.
- **Image Extraction:** Retrieves images from PDFs, saves them locally, and manages captions.
- **Embedding Generation:** Creates vector embeddings for text chunks using a local LM Studio server.
- **FAISS Indexing:** Implements a FAISS index for efficient similarity search and retrieval.
- **FAQ Caching:** Caches frequently asked questions and their answers for faster responses. 
- **History Tracking:** Maintains a history of user queries and corresponding answers. 

## Installation

 
### Conda Setup


1. **Create a Conda Environment**
 

   ```bash
   conda create -n daft-qa-env python=3.8 or 3.10
   ```

3. **Activate the Environment**

   ```bash
   conda activate daft-qa-env
   ```

### Install Dependencies
 

   ```bash
   pip install -r requirements.txt
   ``` 

## Usage

### Running the Application

1. **Ensure the Conda Environment is Activated**

   ```bash
   conda activate daft-qa-env
   ```

2. **Run the Streamlit App**

   ```bash
   streamlit run main.py
   ```
   run the ogmain if this is too slow (split up for organization)

   This command will launch the Streamlit application in your default web browser. If it doesn't open automatically, navigate to the URL provided in the terminal (usually `http://localhost:8501`).
 