from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableBranch
from langchain_openai import ChatOpenAI

load_dotenv()

model=ChatOpenAI(model="gpt-4o-mini")

positive_feedback_template=ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant"),
        ("human", "Generate a thank you note for this positive feedback: {feedback}."),
    ]
)

negative_feedback_template=ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant"),
        ("human","Generate a response addressing this negative feedback: {feedback}.")
    ]
)

neutral_feedback_template=ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant"),
        ("human","Generate a request for more details for this neutral feedback: {feedback}.")
    ]
)

escalate_feedback_template=ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant."),
        ("human","Generate a message to escalate this feedback to a human agent: {feedback}.")
    ]
)


#Define the feedback classification template
classification_template=ChatPromptTemplate.from_messages(
    [
        ("system", "you are a helpful assistant"),
        ("human", "Classify the feedback as positive, negative, neutral, or escalate: {feedback}.")
    ]
)

# Define the runnable branches for handling feedback
branches=RunnableBranch(
    (
        lambda x: "positive" in x,
        positive_feedback_template | model | StrOutputParser()  # positive feedback chain
    ),
    (
        lambda x: "negative" in x,
        negative_feedback_template | model | StrOutputParser()  # negative feedback chain
    ),
    (
        lambda x: "neutral" in x,
        neutral_feedback_template | model | StrOutputParser()  # neutral feedback chain
    ),
    escalate_feedback_template | model | StrOutputParser()  # escalate feedback chain
)

# Create a classification chain
classification_chain=classification_template | model | StrOutputParser()

# Combine classification and response generation into one chain
chain = classification_chain | branches

review="The product is terrible. It broke after just one use and the quality is very poor."
result=chain.invoke({"feedback": review})
print(result)
