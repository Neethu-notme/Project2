import re
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ---------------------------
# TEXT CLEANING FUNCTION
# ---------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z ]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# Load dataset
df = pd.read_csv("your_resume_dataset.csv")  # Must have Content & Category columns

# Apply cleaning
df["Cleaned_Content"] = df["Content"].apply(clean_text)

X = df["Cleaned_Content"]
y = df["Category"]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------------------
# CREATE PIPELINE
# ---------------------------
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("svm", SVC(kernel="rbf", probability=True))
])

# Train
pipeline.fit(X_train, y_train)

# Accuracy
y_pred = pipeline.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save single file
joblib.dump(pipeline, "resume_pipeline.pkl")

print("Pipeline saved successfully!")
