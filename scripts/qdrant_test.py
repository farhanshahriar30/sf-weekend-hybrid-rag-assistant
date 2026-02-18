import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv()

url = os.getenv("QDRANT_URL")
api_key = os.getenv("QDRANT_API_KEY")

if not url:
    raise RuntimeError("QDRANT_URL missing from .env")
if not api_key:
    raise RuntimeError("QDRANT_API_KEY missing from .env")

client = QdrantClient(url=url, api_key=api_key)
print(client.get_collections())
