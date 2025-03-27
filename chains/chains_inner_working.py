from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnableSequence
from langchain_openai import ChatOpenAI

load_dotenv()

model=ChatOpenAI(model="gpt-4o-mini")

prompt_template=ChatPromptTemplate.from_messages(
    [
        ("system","You are a facts expert who knows facts about {animal}."),
        ("human","Tell me {fact_count} facts."),
    ]
)

format_prompt=RunnableLambda(lambda x:prompt_template.format_prompt(**x))
invoke_model=RunnableLambda(lambda x: model.invoke(x.to_messages()))
parse_output=RunnableLambda(lambda x: x.content)

# Create the RunnableSequence (equivalent to the LangChain Expression Language (LCEL) expression)
chain= RunnableSequence(first=format_prompt,middle=[invoke_model],last=parse_output)

# Run the chain
response=chain.invoke({"animal":"cat","fact_count":2})

print(response)