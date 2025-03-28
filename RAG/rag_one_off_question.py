import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage,SystemMessage
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

#Define the persistant directory
current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory=os.path.join(current_dir,"db","chroma_db_with_metadata")

#Define the embedding model
# embeddings=OpenAIEmbeddings(model="text-embedding-3-small")
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en"
)

# load the existing vector store with embedding function
db=Chroma(persist_directory=persistent_directory,embedding_function=embeddings)

#Define the user's question
query="What role do comfort and familiarity play in the early chapters of The Lord of the Rings?"

#Retrieve relevant documents based on query
retriever=db.as_retriever(
    search_type="similarity",
    search_kwargs={"k":1}
)
relevant_docs=retriever.invoke(query)

#Display the relevant results with metadata
# print("\n--- Relevant Documents ---")
# for i, doc in enumerate(relevant_docs,1):
#     print(f"Document {i}:\n{doc.page_content}\n")
#     if doc.metadata:
#         print(f"Source: {doc.metadata.get('source','Unknown')}\n")

combined_input=(
    "Here are some documents that might help answer the question: "
    + query
    + "\n\nRelevant Document:\n"
    + "\n\n".join([doc.page_content for doc in relevant_docs])
    + "\n\nPlease provide a rough answer based only on the provided documents. Id the answer is not found in the documents, respond with 'I'm not sure'."
)

#Create a ChatOpenAI model
model=ChatOpenAI(model="gpt-4o-mini")

#Define the messages for the model
messages=[
    SystemMessage(content="You are a helpful assistant"),
    HumanMessage(content=combined_input)
]

#Invoke the model with the combined input
result=model.invoke(messages)

#Display the full result and content only
print("\n--- Generated Response ---")
print(result)
print("Content only:")
print(result.content)   