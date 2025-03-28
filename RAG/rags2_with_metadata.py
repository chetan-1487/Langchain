import os
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain_huggingface import HuggingFaceEmbeddings


#Define the persistant directory
current_dir=os.path.dirname(os.path.abspath(__file__))
db_dir=os.path.join(current_dir,"db")
persistent_directory=os.path.join(db_dir,"chroma_db_with_metadata")

#Define the embedding model
# embeddings=OpenAIEmbeddings(model="text-embedding-3-small")
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en"
)

#Load the existing vector store with the embedding function
db=Chroma(persist_directory=persistent_directory,embedding_function=embeddings)

#Define the user's question
query="What role do comfort and familiarity play in the early chapters of The Lord of the Rings?"

#Retrieve relevant documents based on the query
retriever=db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k":1,"score_threshold":0.0}
)
relevant_docs=retriever.invoke(query)

#Display the relevant result with metadata
print("\n--- Relevant documents ---")
for i,doc in enumerate(relevant_docs,1):
    print(f"Document {i}:\n{doc.page_content}\n")
    print(f"Source: {doc.metadata['source']}\n")