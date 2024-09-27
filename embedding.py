import re
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8002/v1", api_key="lm-studio")

def serialize_embedding(embedding):
    return ",".join(map(str, embedding))

def deserialize_embedding(embedding_str):
    try:
        return list(map(float, embedding_str.split(',')))
    except ValueError:
        return []

def get_embedding(text, model="nomic-ai/nomic-embed-text-v1.5-GGUF"):
    text = text.replace("\n", " ")
    try:
        response = client.embeddings.create(input=[text], model=model)
        embedding = response.data[0].embedding
        return embedding
    except Exception as e:
        return None
