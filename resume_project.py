# Import required libraries
import pandas as pd
import re
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# Load dataset
data = pd.read_csv("resume.csv")

# Display first few rows
print(data.head())


# Function to clean text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z ]', '', text)
    return text


# Apply text cleaning
data['cleaned'] = data['Resume_str'].apply(clean_text)


# Convert text data into numerical features using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(data['cleaned'])


# Encode target labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(data['Category'])


# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Train the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)


# Make predictions
y_pred = model.predict(X_test)


# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# Predict category for new resume text
def predict_resume(text):
    cleaned = clean_text(text)
    vector = vectorizer.transform([cleaned])
    pred = model.predict(vector)
    return label_encoder.inverse_transform(pred)[0]


# Refinement function for detailed roles
def refine_prediction(text, base_pred):
    text = text.lower()

    if "machine learning" in text or "data science" in text:
        return "Data Scientist"
    elif "html" in text or "css" in text or "react" in text:
        return "Web Developer"
    elif "java" in text:
        return "Java Developer"
    elif "python" in text:
        return "Python Developer"
    elif "accounting" in text or "finance" in text:
        return "Finance Analyst"
    else:
        return base_pred


# Test with multiple inputs
texts = [
    "python machine learning data analysis",
    "html css javascript react",
    "accounting finance taxation excel"
]

for text in texts:
    base = predict_resume(text)
    final = refine_prediction(text, base)

    print("\nInput:", text)
    print("Base Prediction:", base)
    print("Final Prediction:", final)


# Single test example
text = "python machine learning deep learning"

base = predict_resume(text)
final = refine_prediction(text, base)

print("\nSingle Test")
print("Base:", base)
print("Final:", final)


# Save trained model and components
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")

print("\nModel and components saved successfully.")