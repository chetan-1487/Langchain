from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()

messages=[
    SystemMessage(content="Solve the following math problem"),
    HumanMessage(content="What is the square root of 497?")
]

#----------------Langchain openai chat model-----------------

llm=ChatOpenAI(model="gpt-4o-mini")

answer=llm.invoke(messages)
print(answer.content)

# -----------------Anthropic Chat model-----------------

model=ChatAnthropic(model="claude-3-opus-20240229")
result=model.invoke(messages)
print(f"answer from anthropic:{result.content}")


# -----------------Google Chat model-----------------

model=ChatGoogleGenerativeAI(model="gemini-1.5-flash")
result=model.invoke(messages)
print(f"answer from google:{result.content}")
