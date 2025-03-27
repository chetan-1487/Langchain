import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# Example dataset
data = {
    "text": [
        "Win a free iPhone now!",
        "Congratulations! You won a lottery.",
        "Meeting at 3 PM. Please confirm.",
        "Here’s your bank statement.",
        "Click here to claim your prize!",
        "Your report is attached."
    ],
    "label": ["Spam", "Spam", "Not Spam", "Not Spam", "Spam", "Not Spam"]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Convert text into numbers (Bag of Words method)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df["text"])

# Convert labels to numbers (Spam = 1, Not Spam = 0)
y = df["label"].apply(lambda x: 1 if x == "Spam" else 0)

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Naive Bayes model (simple AI model for text classification)
model = MultinomialNB()
model.fit(X_train, y_train)

# Test on new text
new_text = ["Get a free vacation now!"]
new_text_vectorized = vectorizer.transform(new_text)
prediction = model.predict(new_text_vectorized)

# Show result
print("Spam" if prediction[0] == 1 else "Not Spam")
