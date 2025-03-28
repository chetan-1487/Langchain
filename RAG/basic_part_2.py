import os
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
#Define the persisent directory
current_dir=os.path.dirname(os.path.abspath(__file__))
persistent_directory=os.path.join(current_dir,"db","chroma_db")

# Define the embedding model
# embeddings=OpenAIEmbeddings(model="text-embedding-3-small")
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en"
)

# Load the existing vector store with the embedding function
db=Chroma(persist_directory=persistent_directory,embedding_function=embeddings)

#Define the user's Questions
query="What role do comfort and familiarity play in the early chapters of The Lord of the Rings?"

#Retrive relevant documents based on query
retriever=db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k":1,"score_threshold":0.5},
)
relevant_docs=retriever.invoke(query)

#Display the relevant results with metadata
print("\n--- Relevant Documents ---")
for i, doc in enumerate(relevant_docs,1):
    print(f"Document {i}:\n{doc.page_content}\n")
    if doc.metadata:
        print(f"Source: {doc.metadata.get('source','Unknown')}\n")