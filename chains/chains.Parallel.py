from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnableParallel
from langchain_openai import ChatOpenAI

load_dotenv()

model=ChatOpenAI(model="gpt-4o-mini")

# Define Prompt Template for movie summary
summary_template = ChatPromptTemplate.from_messages(
    [
        {"role": "system", "content": "You are a movie critic"},
        {"role": "human", "content": "Write a summary of the movie {movie_name}"},
    ]
)

# Define Plot analysis step
def analyze_plot(plot):
    plot_template=ChatPromptTemplate.from_messages(
        [
            {"role": "system", "content": "You are a movie critic"},
            {"role": "human", "content": "Analyze the plot: {plot}. what are its strengths and weaknesses?"},
        ]    
    )
    return plot_template.format_prompt(plot=plot)

# Define Character analysis step

def analyze_characters(characters):
    characters_template=ChatPromptTemplate.from_messages(
        [
            {"role": "system", "content": "You are a movie critic"},
            {"role": "human", "content": "Analyze the characters: {characters}. what are their strengths and weaknesses?"},
        ]    
    )
    return characters_template.format_prompt(characters=characters)

#Combine branches into a final verdict
def combine_verdict(plot_analysis, character_analysis):
    return f"Plot Analysis: {plot_analysis}\n\nCharacter Analysis: {character_analysis}"

#Simplify branches with LCEL
plot_branch_chain=(
    RunnableLambda(lambda x:analyze_plot(x)) | model | StrOutputParser()
)

character_branch_chain=(
    RunnableLambda(lambda x:analyze_characters(x)) | model | StrOutputParser()
)

chain=(
    summary_template
    | model
    | StrOutputParser()
    | RunnableParallel(branches={"plot":plot_branch_chain, "characters":character_branch_chain})
    | RunnableLambda(lambda x:combine_verdict(x["branches"]["plot"], x["branches"]["characters"]))
)

result=chain.invoke({"movie_name":"Inception"})

print(result)