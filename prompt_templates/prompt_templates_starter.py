from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

llm=ChatOpenAI(model="gpt-4o-mini")

# template="Write a {tone} email to {company} expressing interest in the {position} position, mentioning {skill} as a key strength, keep it to 4 lines max"

messages = [
    {"role": "system", "content": "You are a comedian who tells jokes about {topic}."},
    {"role": "human", "content": "Tell me {joke_count} jokes."},
]

prompt_template=ChatPromptTemplate.from_messages(messages)
prompt=prompt_template.invoke({
    "topic":"lawyers",
    "joke_count":3
})
print("\n------ Prompt with system and human messages (Tuple) ------")
print(prompt)
# prompt_template=ChatPromptTemplate.from_template(template)
# prompt=prompt_template.invoke({
#     "tone":"energetic",
#     "company":"Samsung",
#     "position":"AI Engineer",
#     "skill":"Python"
# })

# result=llm.invoke(prompt)
# print(result.content)