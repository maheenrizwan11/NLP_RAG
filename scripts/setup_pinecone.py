import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

existing = [idx.name for idx in pc.list_indexes()]

if "rag-qa-nlp" not in existing:
    pc.create_index(
        name="rag-qa-nlp",
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    print("Index created: rag-qa-nlp")
else:
    print("Index already exists")

index = pc.Index("rag-qa-nlp")
print(index.describe_index_stats())
