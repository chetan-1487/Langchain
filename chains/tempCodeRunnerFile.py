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