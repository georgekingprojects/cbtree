import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import os

# Example full path components
directory = '/Users/george/Desktop/cbtree/cbtree'
file_name = 'Annotated_data.csv'

# Construct the full path
full_path = os.path.join(directory, file_name)

# Load the data from CSV
df = pd.read_csv(full_path)
# Drop rows with empty values in the 'text' column
df.dropna(subset=['Distorted part'], inplace=True)

# Assuming your CSV has columns for 'text' and 'cognitive_distortion'
X = df['Distorted part'].tolist()  # Extracting text data
y = df['Dominant Distortion'].tolist() 

# Step 2: Feature Extraction
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(X)

# Step 3: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Model Training
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 5: Model Evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Step 6: Prediction for new inputs
def predict_cognitive_distortions(text):
    # Assuming 'vectorizer' and 'model' are already defined

    # Transform the input text
    text_features = vectorizer.transform([text])

    # Predict probabilities for each class
    predicted_probabilities = model.predict_proba(text_features)

    # Get all possible classes
    classes = model.classes_

    # Create a dictionary to store predictions and their probabilities
    predictions = {}
    for i, class_ in enumerate(classes):
        predictions[class_] = predicted_probabilities[0][i]

    return predictions

# Example usage:
new_text = "Everyone probably thinks I'm annoying."
predicted_distortions = predict_cognitive_distortions(new_text)
print("Predicted cognitive distortions and their probabilities:")
for distortion, probability in predicted_distortions.items():
    print(f"{distortion}: {probability}")

