from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain.schema.runnable import RunnableLambda

load_dotenv()

model=ChatOpenAI(model="gpt-4o-mini")

# Define prompt model
animal_facts_template = ChatPromptTemplate.from_messages(
    [
        {"role": "system", "content": "You like telling facts and you tell facts about {animal}."},
        {"role": "user", "content": "Tell me {fact_count} facts."},
    ]
)

# Define the prompt template for translation to French
translate_template = ChatPromptTemplate.from_messages(
    [
        {"role": "system", "content": "You are a translator and convert the provided text into {language}."},
        {"role": "user", "content": "Translate the following text to {language}: {text}"},
    ]
)

# Define additional processing steps using RunnableLambda
count_words = RunnableLambda(lambda x: {"text": f"Word count: {len(x.content.split())}\n{x.content}", "language": "French"})

# Create the combined chain using LangChain Expression Language (LCEL)
chain = animal_facts_template | model | count_words | translate_template | model | StrOutputParser()


result=chain.invoke({"animal":"cat","fact_count":2})

print(result)