from dotenv import load_dotenv
from google.cloud import firestore
from langchain_google_firestore import FirestoreChatMessageHistory
from langchain_openai import ChatOpenAI

load_dotenv()

#setup firestore Firestore
PROJECT_ID="langchain-e08bf"
SESSION_ID=""
COLLECTION_ID="chat_messages"

#Initialize Firestore Client
print("Initializing Firestore Client")
client=firestore.Client(project=PROJECT_ID)

#Initialize Firestore chat message history
print("Initializing Firestore Chat Message History")
chat_history=FirestoreChatMessageHistory(
    session_id=SESSION_ID,
    collection=COLLECTION_ID,
    client=client
)
print("cha History Initialized")
print("current chat History: ",chat_history.messages)

model=ChatOpenAI(model="gpt-4o-mini")

print("start chatting with the AI. Type 'exit' to quit")
while True:
    user_input=input("You: ")
    if user_input.lower()=="exit":
        break
    chat_history.add_user_message(user_input)
    ai_response=model.invoke(chat_history.messages)
    chat_history.add_ai_message(ai_response.content)
    print("AI: ",ai_response.content)
