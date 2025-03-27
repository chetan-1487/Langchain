from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage 

load_dotenv()

# create a chatopenai model
model=ChatOpenAI(model="gpt-4o-mini")

chat_history=[] # use a list to store messages

# Set an initial message (optional)

system_message=SystemMessage(content="You are a helpul AI assistant")
chat_history.append(system_message)

#chat loop
while True:
    query=input("You: ")
    if query.lower()=="exit":
        break
    chat_history.append(HumanMessage(content=query)) #Add user message to chat history

    result=model.invoke(chat_history)
    response=result.content
    chat_history.append(AIMessage(content=response)) # add AI message

    print(f"AI: {response}")

# --------------------Message history--------------------
print("Message history:")
print(chat_history)