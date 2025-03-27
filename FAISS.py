from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load a model to generate embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")

# Example documents
docs = ["cat", "dog", "apple", "banana", "elephant", "car", "robot", "technology", "machine learning", "artificial intelligence"]


# Convert text to embeddings
embeddings = model.encode(docs)

# Create FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings))

# Search for a query
query_embedding = model.encode(["AI revolution"])
distances, indices = index.search(np.array([query_embedding[0]]), k=1)

print("Most similar document:", docs[indices[0][0]])
print("Distances:", distances)
print("Indices:", indices)
for i, idx in enumerate(indices[0]):
    print(f"Rank {i+1}: {docs[idx]} (Distance: {distances[0][i]})")

