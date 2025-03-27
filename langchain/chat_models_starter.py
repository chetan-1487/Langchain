from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()

llm=ChatOpenAI(model="gpt-4o-mini")

messages=[
    SystemMessage("You are an expert in social expert media content strategy"),
    HumanMessage("Give a short tip to create engaging posts on Instagram")
]

answer=llm.invoke(messages)
print(answer.content)


# import getpass
# import os

# if not os.environ.get("OPENAI_API_KEY"):
#   os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

# from langchain.chat_models import init_chat_model

# model = init_chat_model("gpt-4o-mini", model_provider="openai")