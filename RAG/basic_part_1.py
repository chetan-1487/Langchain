import os
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

#define the directory containing the text file and persistant directory
Current_dir=os.path.dirname(os.path.abspath(__file__))
file_path=os.path.join(Current_dir,"documents","lord_of_the_rings.txt")
persistant_directory=os.path.join(Current_dir,"db","chroma_db")

# check if the Chroma vector store already exists
if not os.path.exists(persistant_directory):
    print("Persistant does not exist. Initializing vector store...")

    #Ensure the text file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"The file {file_path} does not exis. Please check the path."
        )

    #Read the text content from the file
    loader=TextLoader(file_path)    
    documents=loader.load()

    #Split the document into chunks
    text_splitter=CharacterTextSplitter(chunk_size=1000,chunk_overlap=0)
    docs=text_splitter.split_documents(documents)
    
    # Display information about the split documents
    print("\n--- document chunk information ---")
    print(f"Number of document chunks: {len(docs)}")

    #Creating embedding
    print("\n--- creating embeddings ---")
    # embedding=OpenAIEmbeddings(
    #     model="text-embedding-3-small"
    # ) # Update to a valid emnedding model if needed
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en"
    )
    print("\n-- Finished creaing embedding ---")

    #Create the vector store and persist it automatically
    print('\n-- Creating vector store --')
    db=Chroma.from_documents(
        docs, embeddings, persist_directory=persistant_directory
    )
    print("\n--- Finished creating vector store ---")
else:
    print("Vector store already exists. No need to initialize.")