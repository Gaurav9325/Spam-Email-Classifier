# Project 1: Spam Email Classifier using Logistic Regression
# We'll use a small, custom dataset and the scikit-learn library.

# 1. Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

print("Libraries imported successfully.")

# 2. Create a dummy dataset
# In a real-world scenario, you would load data from a CSV or text file.
# The 'text' column contains the email content, and the 'label' column is the target (0 for not spam, 1 for spam).
data = {
    'text': [
        "Hey, how are you doing?",
        "Get free prizes now! Click here.",
        "Meeting at 2pm tomorrow?",
        "You've won a free iPhone. Claim your prize!",
        "Can you send me the report?",
        "Urgent: Your account is locked. Verify now.",
        "Just wanted to follow up on our discussion.",
        "Congratulations! You've been selected for a special offer.",
        "Let's grab lunch sometime next week.",
        "This is an important message from your bank.",
        "Re: Your submission",
        "Don't miss out on this amazing deal!",
        "Final Reminder: Your subscription is expiring."
    ],
    'label': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1]
}

df = pd.DataFrame(data)
print("\n--- Sample Dataset ---")
print(df.head())
print("-" * 20)

# 3. Preprocessing: Convert text data into numerical features
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['text'])
y = df['label']

print(f"\nShape of the features matrix: {X.shape}")
print(f"Number of unique words (features): {len(vectorizer.get_feature_names_out())}")
print("-" * 20)

# 4. Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

print("\n--- Data Splitting ---")
print(f"Training set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")
print("-" * 20)

# 5. Train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

print("\nLogistic Regression model trained successfully.")
print("-" * 20)

# 6. Make predictions and evaluate the model
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("\n--- Model Evaluation ---")
print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print("classification_report(y_test, predictions)")
print("-" * 20)

# 7. Make a prediction on a new email
new_email_text = ["Exclusive offer for you! Click now to win."]
new_email_vectorized = vectorizer.transform(new_email_text)
prediction = model.predict(new_email_vectorized)

if prediction[0] == 1:
    print(f"\nThe email '{new_email_text[0]}' is classified as: SPAM")
else:
    print(f"\nThe email '{new_email_text[0]}' is classified as: NOT SPAM")