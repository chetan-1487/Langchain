import os
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.llms import HuggingFaceHub

# Set up Hugging Face API key
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "your_huggingface_api_key"

# Load documents for classification and retrieval
loader = TextLoader("documents.txt")  # Load documents from a text file
documents = loader.load()

# Split text into chunks for embedding
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)

# Generate embeddings and store in FAISS vector database
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_db = FAISS.from_documents(chunks, embedding_model)

# Initialize the chatbot model
chat_model = HuggingFaceHub(repo_id="facebook/blenderbot-3B", model_kwargs={"temperature": 0.7})

def classify_text(user_input):
    """Classify user input using Hugging Face embeddings."""
    query_embedding = embedding_model.embed_query(user_input)
    docs = vector_db.similarity_search_by_vector(query_embedding, k=3)
    return docs

def chatbot(user_input):
    """Chatbot with retrieval-augmented generation (RAG)"""
    context_docs = classify_text(user_input)
    context = "\n".join([doc.page_content for doc in context_docs])
    
    messages = [
        SystemMessage(content="You are an AI assistant helping with information retrieval."),
        HumanMessage(content=f"User query: {user_input}\nRelevant context: {context}")
    ]
    
    response = chat_model(messages)
    return response.content

# Example usage
if __name__ == "__main__":
    user_query = input("Ask something: ")
    print("Bot:", chatbot(user_query))