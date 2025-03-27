from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()

model=ChatOpenAI(model="gpt-4o-mini")

#Define Prompt templates (no need for separate Runnable files)
prompt_template=ChatPromptTemplate.from_messages(
    [
        ("system","You are a facts expert who knows facts about {animal}."),
        ("human","Tell me {fact_count} facts."),
    ]
)

# Create the combined chain using LangChain Expression Language (LCEL)
chain=prompt_template| model | StrOutputParser()

# Run the chain
result=chain.invoke({"animal":"cat","fact_count":2})

print(result)