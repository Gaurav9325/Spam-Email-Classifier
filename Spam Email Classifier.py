# Improved Spam Email Classifier

import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import joblib

# 1. Dataset (small sample, better to use larger dataset later)
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

# 2. Preprocessing
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['text'])
y = df['label']

# Balanced train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# 3. Define models
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "SVM": SVC()
}

# 4. Train, Evaluate, Save
for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print(f"\n--- {name} ---")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.2f}")
    print("Classification Report:")
    print(classification_report(y_test, predictions, zero_division=0))

    # Cross-validation (optional, gives better estimate)
    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    print(f"Cross-validation Accuracy (5-fold): {scores.mean():.2f}")

    # Save model
    filename = f"{name.replace(' ', '_').lower()}_spam_model.pkl"
    joblib.dump(model, filename)
    print(f"Model saved as {filename}")

# 5. Load a saved model & test new email
loaded_model = joblib.load("decision_tree_spam_model.pkl")
new_email = ["Exclusive offer for you! Click now to win."]
new_email_vec = vectorizer.transform(new_email)
pred = loaded_model.predict(new_email_vec)

print("\n--- Loaded Model Prediction ---")
print(f"Email: {new_email[0]}")
print("Prediction:", "SPAM" if pred[0] == 1 else "NOT SPAM")
